import torch.nn.functional as F
import torch
import torch.nn as nn

class TTA(nn.Module):
    """TTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, model_anchor, use_test_bn=True,
                 use_sabe=True, use_sff=True, admission_policy='released',
                 gate_mode='released', gate_gamma=1.0, fusion_mode='released'):
        super().__init__()
        self.model = model
        self.model_anchor = model_anchor.eval()
        # EXP0_REPRO: keep the released path as the default and expose only
        # the paper Table 5 component switches needed by EXP-0.
        self.use_test_bn = use_test_bn
        self.use_sabe = use_sabe
        self.use_sff = use_sff
        if admission_policy not in {'released', 'rolling_all_40', 'rolling_all_160',
                                    'sft_queue_40', 'domain_reset_oracle'}:
            raise ValueError(admission_policy)
        self.admission_policy = admission_policy
        if gate_mode not in {'released', 'identity', 'lowconf', 'highconf'}:
            raise ValueError(gate_mode)
        self.gate_mode = gate_mode
        self.gate_gamma = float(gate_gamma)
        if fusion_mode not in {'released', 'crsff_identity', 'global_reliability',
                               'class_reliability', 'inverse_class_reliability'}:
            raise ValueError(fusion_mode)
        self.fusion_mode = fusion_mode

        self.num_classes = 4
        self.max_lens = 40
        self.topk = 5
        self.threshold = 0.9
        self.entropy_list = []
        self.pool = Prototype_Pool(0.1,class_num=self.num_classes,max = self.max_lens).cuda()
        # EXP0_5_DIAG: read-only state populated from existing intermediates.
        self.last_diag = {}
        self.last_ccd_diag = {}
        self.accepted_ccd_queue = []
        # EXP2_DIAG: read-only current/query and pre-update memory prototypes.
        self.last_class_prototypes = None
        self.last_class_masses = None
        self.last_class_memory_prototypes = None
        self.last_class_memory_masses = None
        self.last_class_memory_names = []
        self.last_anchor_probability = None
        # EXP3_DIAG: expose the unchanged pre-SABE/SFF bottleneck map for proxy diagnostics.
        self.last_latent_feature_map = None
        # EXP4_GATE: diagnostics for the optional SFF strength modulation.
        self.last_gate_diag = {}
        self.last_released_sff_feature = None
        self.last_gated_sff_feature = None
        # EXP6_CRSFF: read-only diagnostic state for the fusion-weight experiment.
        self.last_crsff_diag = {}

    def reset_admission_history(self):
        self.entropy_list = []
    def forward(self, x, names):
        for _ in range(1):
            outputs = self.forward_and_adapt(x, self.model, names)
        return outputs

    torch.autograd.set_detect_anomaly(True)
    @torch.no_grad() 
    def forward_and_adapt(self, x, model, names):
        layer_fea = 'med'
        self.ww = x.shape[-1]
        bad_num = x.shape[0]
        topk = self.topk
        latent_model = model.get_feature(x, loc = layer_fea)
        latent_feature_map = latent_model
        # EXP3_DIAG: this is the same feature tensor used by Released global retrieval.
        self.last_latent_feature_map = latent_feature_map
        self.last_released_sff_feature = None
        self.last_gated_sff_feature = None
        self.last_crsff_diag = {}
        b,c,w,h = latent_model.shape
        sup_pixel = w
        latent_model = latent_model.reshape(b,c,int(w/sup_pixel),sup_pixel,int(h/sup_pixel),sup_pixel)
        latent_model = latent_model.permute(0,2,4,1,3,5)
        latent_model = latent_model[0].reshape(1*int(w/sup_pixel)*int(h/sup_pixel),c*sup_pixel*sup_pixel)
        latent_model_,fff, out_image, out_mask, len_pool = self.pool.get_pool_feature(latent_model,None,top_k = topk)

        # EXP0_5_DIAG: resolve names before any possible FIFO update shifts indices.
        retrieval_diag = getattr(self.pool, 'last_retrieval_diag', {})
        retrieval_names = []
        for index in retrieval_diag.get('indices', []):
            if index < len(self.pool.name_list):
                retrieval_names.append(self.pool.name_list[index])
            else:
                retrieval_names.append(None)
        history_before = len(self.entropy_list)
        pool_before = {
            'feature': int(self.pool.feature_bank.shape[0]),
            'image': int(self.pool.image_bank.shape[0]),
            'mask': int(self.pool.mask_bank.shape[0]),
            'name': len(self.pool.name_list),
        }
        # EXP2_DIAG: freeze the exact pre-update diagnostic memory view.
        self.last_class_memory_prototypes = self.pool.class_prototype_bank
        self.last_class_memory_masses = self.pool.class_mass_bank
        self.last_class_memory_names = list(self.pool.name_list)

        # EXP0_REPRO: SFF-only retains the source-BN path; SABE-only keeps
        # the enhanced batch but discards the feature-fusion replacement.
        if not self.use_sff:
            latent_model_ = latent_model

        with torch.no_grad():
            if len_pool < topk:
                threshold = len_pool / topk
            else:
                threshold = self.threshold
            # Option 1: Use the adapted model for CCD (works similarly in practice, since CCD mainly selects reliable samples)
            # fine = self.get_fine_ccd(x, model.eval(), self.entropy_list, threshold=threshold)

            # Option 2 (default, consistent with paper): Use the source model (model_anchor) for CCD, which retains source BN statistics
            fine = self.get_fine_ccd(x, self.model_anchor.eval(), self.entropy_list, threshold=threshold)

        # EXP2_DIAG: use the same pre-SABE/SFF feature map and anchor probability.
        self.last_class_prototypes, self.last_class_masses = self._build_class_prototypes(
            latent_feature_map, self.last_anchor_probability)

        # EXP6_CRSFF: only the Top-K internal memory weights may change.
        if self.use_sff and self.fusion_mode != 'released' and out_image is not None:
            latent_model_ = self._apply_crsff_fusion(
                latent_model, latent_model_, fff, latent_feature_map,
                self.last_anchor_probability, self.pool.last_fusion_diag,
                c, w, h, self.fusion_mode)
        elif self.use_sff and out_image is not None:
            self.last_crsff_diag = self._build_crsff_diag(
                self.pool.last_fusion_diag, self.pool.last_retrieval_diag,
                self.pool.class_reliability_bank, self.pool.class_soft_mass_bank,
                self.pool.name_list, self.last_anchor_probability, self.fusion_mode)

        # EXP4_GATE: keep Released and Identity on the exact original feature path.
        self.last_gate_diag = self._empty_gate_diag()
        self.last_gate_diag['global_rate'] = self.pool.last_fusion_diag.get('global_rate', 0.0)

        if fine:
            self.pool.update_feature_pool(latent_model)
            self.pool.update_image_pool(x)
            self.pool.update_mask_pool(model(x).softmax(1))
            self.pool.update_name_pool(names[0])
            self.pool.update_diagnostic_prototype_pool(self.last_class_prototypes,
                                                       self.last_class_masses)
            reliability, soft_mass = self._anchor_class_reliability(self.last_anchor_probability)
            self.pool.update_reliability_pool(reliability, soft_mass)
        self.pool.validate_synchronized_banks()
        if out_image is not None:
            if self.use_sabe:
                out_image = out_image[0]
                x_hised = torch.cat((x, out_image), dim=0)
                latent_model = model.get_feature(x_hised, loc = layer_fea)
            else:
                # EXP0_REPRO: source-BN SFF does not construct an enhanced
                # batch.  The current image is fused at the bottleneck only.
                latent_model = model.get_feature(x, loc = layer_fea)
            latent_model_ = latent_model_.view(bad_num,int(w/sup_pixel),int(h/sup_pixel),c,sup_pixel,sup_pixel)
            latent_model_ = latent_model_.permute(0,3,1,4,2,5)
            latent_model_ = latent_model_.reshape(bad_num,c,w,h)
            # EXP4_GATE: retain exact feature tensors used by Released and gated SFF.
            self.last_released_sff_feature = latent_model_.detach()
            if self.use_sff and self.gate_mode != 'released':
                latent_model_, self.last_gate_diag = self._apply_exp4_gate(
                    latent_feature_map, latent_model_, self.last_anchor_probability,
                    self.pool.last_fusion_diag.get('global_rate', 0.0), self.gate_mode,
                    self.gate_gamma)
            elif self.use_sff:
                self.last_gate_diag = self._diagnose_gate(
                    latent_feature_map, latent_model_, self.last_anchor_probability,
                    self.pool.last_fusion_diag.get('global_rate', 0.0),
                    torch.full((3,), self.pool.last_fusion_diag.get('global_rate', 0.0),
                               device=latent_model_.device), latent_model_, False)
            if self.use_sff:
                latent_model[0:1] = latent_model_
                self.last_gated_sff_feature = latent_model_.detach()
            output = model.get_output(latent_model,loc = layer_fea)[0:1].softmax(1)
        else:
            output = self.model_anchor(x)

        # EXP0_5_DIAG: no value below participates in the output or decision path.
        pool_after = {
            'feature': int(self.pool.feature_bank.shape[0]),
            'image': int(self.pool.image_bank.shape[0]),
            'mask': int(self.pool.mask_bank.shape[0]),
            'name': len(self.pool.name_list),
        }
        ccd_diag = dict(self.last_ccd_diag)
        if self.admission_policy in {'released', 'domain_reset_oracle'}:
            effective_history_before = min(history_before, self.max_lens)
        else:
            effective_history_before = history_before
        effective_history_after = ccd_diag.get('history_len_after', len(self.entropy_list))
        self.last_diag = {
            'ccd': ccd_diag.get('ccd'),
            'ccd_threshold': ccd_diag.get('ccd_threshold'),
            'ccd_margin': (ccd_diag.get('ccd_threshold') - ccd_diag.get('ccd')
                           if ccd_diag.get('ccd_threshold') is not None else None),
            'is_sft': bool(ccd_diag.get('fine', False)),
            'ccd_history_len_before': history_before,
            'ccd_history_len_after': len(self.entropy_list),
            'admission_history_len': effective_history_after,
            'admission_history_len_before': effective_history_before,
            'admission_history_len_after': effective_history_after,
            'admission_history_type': ccd_diag.get('admission_history_type', 'released_local_trimmed_threshold_history'),
            'pool_before': pool_before,
            'pool_after': pool_after,
            # EXP0_5_DIAG: distinguish an actual FIFO write from pool growth.
            'memory_written': bool(ccd_diag.get('fine', False)),
            'pool_grew': pool_after['name'] > pool_before['name'],
            'retrieved_indices': list(retrieval_diag.get('indices', [])),
            'retrieved_similarities': list(retrieval_diag.get('similarities', [])),
            'retrieved_names': retrieval_names,
            'diagnostic_pool_size_before': len(self.last_class_memory_names),
            'diagnostic_pool_size_after': len(self.pool.name_list),
            'gate': dict(self.last_gate_diag),
            'crsff': dict(self.last_crsff_diag),
        }
        return output

    def entropy(self, p, prob=True, mean=True):
        if prob:
            p = F.softmax(p, dim=1)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        if mean:
            return torch.mean(en)
        else:
            return en

    def get_fine_ccd(self, x, model_anchor,entropy_list, threshold = 0.9):
        if self.admission_policy in {'rolling_all_40', 'rolling_all_160', 'sft_queue_40'}:
            return self._get_fine_ccd_controlled(x, model_anchor, threshold)
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    pred1 = model_anchor(x[i:i+1]).softmax(1).detach()
                self.last_anchor_probability = pred1
                pred1 = pred1.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = torch.randperm(pred1.size(0))
                select_point = 200
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                pred1_en =  self.entropy(torch.matmul(pred1.t(), pred1))
                entropy_list.append(pred1_en)
                if len(entropy_list)>self.max_lens:
                    entropy_list = entropy_list[0-self.max_lens:]
        sorted_list = sorted(entropy_list)
        ten_percent_index = int(len(sorted_list) * (1 - threshold))
        if ten_percent_index>0:
            ten_percent_min_value = sorted_list[:ten_percent_index][-1]
            # EXP0_5_DIAG: retain the exact cutoff and decision already used below.
            self.last_ccd_diag = {
                'ccd': float(pred1_en.detach().cpu().item()),
                'ccd_threshold': float(ten_percent_min_value.detach().cpu().item()
                                       if torch.is_tensor(ten_percent_min_value)
                                       else ten_percent_min_value),
                'fine': bool(pred1_en <= ten_percent_min_value),
                'history_len_after': len(entropy_list),
            }
            return pred1_en <= ten_percent_min_value
        else:
            # EXP0_5_DIAG: preserve the original false branch and record no cutoff.
            self.last_ccd_diag = {
                'ccd': float(pred1_en.detach().cpu().item()),
                'ccd_threshold': None,
                'fine': False,
                'history_len_after': len(entropy_list),
            }
            return False

    def _get_fine_ccd_controlled(self, x, model_anchor, threshold):
        values = []
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                pred1 = model_anchor(x[i:i+1]).softmax(1).detach()
                self.last_anchor_probability = pred1
                pred1 = pred1.permute(0,2,3,1)
                pred1 = pred1.reshape(-1, pred1.size(3))
                pred1_rand = torch.randperm(pred1.size(0))
                select_point = 200
                pred1 = F.normalize(pred1[pred1_rand[:select_point]])
                values.append(self.entropy(torch.matmul(pred1.t(), pred1)))

        current = values[-1]
        if self.admission_policy in {'rolling_all_40', 'rolling_all_160'}:
            window = 40 if self.admission_policy == 'rolling_all_40' else 160
            candidate_history = (list(self.entropy_list) + values)[-window:]
            self.entropy_list[:] = candidate_history
            self.accepted_ccd_queue = []
            history_type = f'rolling_all_{window}'
        else:
            candidate_history = list(self.accepted_ccd_queue) + values
            history_type = 'sft_queue_40'

        sorted_list = sorted(candidate_history)
        ten_percent_index = int(len(sorted_list) * (1 - threshold))
        if ten_percent_index > 0:
            ten_percent_min_value = sorted_list[:ten_percent_index][-1]
            fine = bool(current <= ten_percent_min_value)
            cutoff = (float(ten_percent_min_value.detach().cpu().item())
                      if torch.is_tensor(ten_percent_min_value) else float(ten_percent_min_value))
        else:
            fine = False
            cutoff = None

        if self.admission_policy == 'sft_queue_40':
            if fine:
                self.accepted_ccd_queue = (self.accepted_ccd_queue + values)[-40:]
            self.entropy_list[:] = self.accepted_ccd_queue

        self.last_ccd_diag = {
            'ccd': float(current.detach().cpu().item()),
            'ccd_threshold': cutoff,
            'fine': fine,
            'history_len_after': len(self.entropy_list),
            'admission_history_len': len(candidate_history),
            'admission_history_type': history_type,
        }
        return fine

    def _empty_gate_diag(self):
        # EXP4_GATE: keep a stable schema for empty-pool and non-SFF cases.
        return {
            'global_rate': 0.0, 'r_lv': float('nan'), 'r_myo': float('nan'), 'r_rv': float('nan'),
            'r_bar': float('nan'), 'alpha_lv': float('nan'), 'alpha_myo': float('nan'),
            'alpha_rv': float('nan'), 'alpha_map_mean': float('nan'), 'alpha_map_std': float('nan'),
            'alpha_map_min': float('nan'), 'alpha_map_max': float('nan'),
            'fraction_alpha_clipped_0': float('nan'), 'fraction_alpha_clipped_1': float('nan'),
            'scale_map_mean': float('nan'), 'scale_map_max': float('nan'), 'a_fallback': False,
            'correction_norm_released': float('nan'), 'correction_norm_gated': float('nan'),
            'correction_norm_ratio': float('nan'), 'corr_released_lv': float('nan'),
            'corr_released_myo': float('nan'), 'corr_released_rv': float('nan'),
            'corr_gated_lv': float('nan'), 'corr_gated_myo': float('nan'), 'corr_gated_rv': float('nan'),
        }

    def _anchor_class_reliability(self, probability):
        # EXP6_CRSFF: exact EXP-3 Class Confidence and foreground soft mass.
        probability = probability[0].detach().float()
        foreground = probability[1:]
        soft_mass = foreground.sum(dim=(1, 2))
        reliability = foreground.square().sum(dim=(1, 2)) / (soft_mass + 1e-8)
        return reliability, soft_mass

    def _build_crsff_diag(self, fusion_diag, retrieval_diag, reliability_bank,
                          mass_bank, names, anchor_probability, fusion_mode):
        # EXP6_CRSFF: report exact Top-K weights and class-conditioned changes.
        indices = list(fusion_diag.get('topk_indices', []))
        similarities = torch.as_tensor(fusion_diag.get('topk_similarities', []),
                                       device=self.pool.feature_bank.device, dtype=torch.float32)
        if not indices or similarities.numel() == 0 or reliability_bank is None or reliability_bank.shape[0] == 0:
            return {'fusion_mode': fusion_mode, 'pool_size': int(self.pool.feature_bank.shape[0]),
                    'global_rate': float(fusion_diag.get('global_rate', 0.0)), 'topk_indices': indices,
                    'topk_names': [names[i] for i in indices if i < len(names)],
                    'topk_similarities': similarities.detach().cpu().tolist(),
                    'query_class_soft_mass': [], 'released_weights': [], 'memory_class_reliabilities': [],
                    'global_reliability_weights': [], 'class_reliability_weights': [],
                    'inverse_class_reliability_weights': [], 'weight_l1_change_vs_released': [],
                    'weight_entropy_released': 0.0, 'weight_entropy_class': [],
                    'top1_weight_released': 0.0, 'top1_weight_class': []}
        index_tensor = torch.as_tensor(indices, device=reliability_bank.device, dtype=torch.long)
        rel = reliability_bank[index_tensor].detach().float()
        masses = mass_bank[index_tensor].detach().float()
        released = torch.softmax(similarities, dim=0)
        raw_global_rel = (rel * masses).sum(dim=1) / (masses.sum(dim=1) + 1e-8)
        effective_rel = rel
        exp_sim = torch.exp(similarities)
        if fusion_mode in {'released', 'crsff_identity'}:
            if fusion_mode == 'crsff_identity':
                effective_rel = torch.ones_like(rel)
            global_rel = torch.ones_like(raw_global_rel) if fusion_mode == 'crsff_identity' else raw_global_rel
            global_weights = released
            class_weights = released[:, None].expand(-1, 3)
        elif fusion_mode == 'global_reliability':
            global_rel = raw_global_rel
            effective_rel = raw_global_rel[:, None].expand(-1, 3)
            global_weights = exp_sim * global_rel
            global_weights = global_weights / (global_weights.sum() + 1e-8)
            class_weights = global_weights[:, None].expand(-1, 3)
        elif fusion_mode == 'class_reliability':
            global_rel = raw_global_rel
            class_weights = exp_sim[:, None] * rel
            class_weights = class_weights / (class_weights.sum(dim=0, keepdim=True) + 1e-8)
            global_weights = released
        elif fusion_mode == 'inverse_class_reliability':
            global_rel = raw_global_rel
            effective_rel = torch.clamp(1.0 - rel, min=1e-6)
            class_weights = exp_sim[:, None] * effective_rel
            class_weights = class_weights / (class_weights.sum(dim=0, keepdim=True) + 1e-8)
            global_weights = released
        else:
            raise ValueError(fusion_mode)
        inverse_rel = torch.clamp(1.0 - rel, min=1e-6)
        inverse_weights = exp_sim[:, None] * inverse_rel
        inverse_weights = inverse_weights / (inverse_weights.sum(dim=0, keepdim=True) + 1e-8)
        query_mass = self._anchor_class_reliability(anchor_probability)[1]
        def entropy(values):
            return -(values * torch.log(values + 1e-8)).sum(dim=0)
        class_l1 = (class_weights - released[:, None]).abs().sum(dim=0)
        selected_names = [names[i] if i < len(names) else None for i in indices]
        return {
            'fusion_mode': fusion_mode, 'pool_size': int(self.pool.feature_bank.shape[0]),
            'global_rate': float(fusion_diag.get('global_rate', 0.0)), 'topk_indices': indices,
            'topk_names': selected_names, 'topk_similarities': similarities.detach().cpu().tolist(),
            'query_class_soft_mass': query_mass.detach().cpu().tolist(),
            'released_weights': released.detach().cpu().tolist(),
            'memory_class_reliabilities': rel.detach().cpu().tolist(),
            'effective_class_reliabilities': effective_rel.detach().cpu().tolist(),
            'memory_class_soft_masses': masses.detach().cpu().tolist(),
            'memory_global_reliabilities': raw_global_rel.detach().cpu().tolist(),
            'global_reliability_weights': global_weights.detach().cpu().tolist(),
            'class_reliability_weights': class_weights.detach().cpu().T.tolist(),
            'inverse_class_reliability_weights': inverse_weights.detach().cpu().T.tolist(),
            'weight_l1_change_vs_released': class_l1.detach().cpu().tolist(),
            'weight_entropy_released': float(entropy(released).item()),
            'weight_entropy_class': entropy(class_weights).detach().cpu().tolist(),
            'top1_weight_released': float(released.max().item()),
            'top1_weight_class': class_weights.max(dim=0).values.detach().cpu().tolist(),
            'bank_lengths': {'feature': int(self.pool.feature_bank.shape[0]),
                             'image': int(self.pool.image_bank.shape[0]),
                             'mask': int(self.pool.mask_bank.shape[0]),
                             'name': len(self.pool.name_list),
                             'reliability': int(reliability_bank.shape[0]),
                             'soft_mass': int(mass_bank.shape[0])},
        }

    def _apply_crsff_fusion(self, current_flat, released_flat, retrieved_flat,
                            current_feature, anchor_probability, fusion_diag,
                            c, w, h, fusion_mode):
        # EXP6_CRSFF: fixed Top-K, class reliability reweighting, soft query assignment.
        diag = self._build_crsff_diag(
            fusion_diag, self.pool.last_retrieval_diag,
            self.pool.class_reliability_bank, self.pool.class_soft_mass_bank,
            self.pool.name_list, anchor_probability, fusion_mode)
        self.last_crsff_diag = diag
        indices = list(fusion_diag.get('topk_indices', []))
        if not indices or retrieved_flat is None or self.pool.class_reliability_bank.shape[0] == 0:
            return released_flat
        similarities = torch.as_tensor(fusion_diag['topk_similarities'], device=current_flat.device, dtype=current_flat.dtype)
        index_tensor = torch.as_tensor(indices, device=current_flat.device, dtype=torch.long)
        rel = self.pool.class_reliability_bank[index_tensor].to(current_flat.dtype)
        masses = self.pool.class_soft_mass_bank[index_tensor].to(current_flat.dtype)
        exp_sim = torch.exp(similarities)
        released_weights = torch.softmax(similarities, dim=0)
        global_rel = (rel * masses).sum(dim=1) / (masses.sum(dim=1) + 1e-8)
        if fusion_mode == 'crsff_identity':
            global_weights = released_weights
            class_weights = released_weights[:, None].expand(-1, 3)
        elif fusion_mode == 'global_reliability':
            global_weights = exp_sim * global_rel
            global_weights = global_weights / (global_weights.sum() + 1e-8)
            class_weights = global_weights[:, None].expand(-1, 3)
        elif fusion_mode == 'class_reliability':
            class_weights = exp_sim[:, None] * rel
            class_weights = class_weights / (class_weights.sum(dim=0, keepdim=True) + 1e-8)
            global_weights = released_weights
        elif fusion_mode == 'inverse_class_reliability':
            rel_for_class = torch.clamp(1.0 - rel, min=1e-6)
            class_weights = exp_sim[:, None] * rel_for_class
            class_weights = class_weights / (class_weights.sum(dim=0, keepdim=True) + 1e-8)
            global_weights = released_weights
        else:
            raise ValueError(fusion_mode)
        if fusion_mode == 'crsff_identity':
            # Directly preserve the released tensor for the regression-test hash.
            return released_flat
        memory = retrieved_flat[0].view(len(indices), c, w, h)
        current = current_feature[0].to(memory.dtype)
        mem_bg = torch.einsum('k,kchw->chw', global_weights, memory)
        mem_classes = torch.stack([torch.einsum('k,kchw->chw', class_weights[:, i], memory) for i in range(3)])
        probability = F.interpolate(anchor_probability[0].detach().float().unsqueeze(0),
                                    size=(w, h), mode='bilinear', align_corners=False)[0].to(memory.dtype)
        memory_mix = probability[0] * mem_bg + torch.einsum('ihw,ichw->chw', probability[1:], mem_classes)
        rate = float(fusion_diag.get('global_rate', 0.0))
        fused = (1.0 - rate) * current + rate * memory_mix
        return fused.unsqueeze(0).reshape(1, -1)

    def _diagnose_gate(self, current, released, anchor_probability, global_rate,
                       alpha_foreground, gated, fallback):
        # EXP4_GATE: diagnostics are computed from the exact current/released SFF tensors.
        probability = anchor_probability[0].detach()
        probability_feature = F.interpolate(
            probability.unsqueeze(0), size=current.shape[-2:], mode='bilinear', align_corners=False)[0]
        alpha_values = torch.cat((torch.as_tensor([global_rate], device=current.device), alpha_foreground))
        alpha_map = (probability_feature * alpha_values.view(-1, 1, 1)).sum(dim=0)
        released_delta = released - current
        gated_delta = gated - current
        released_norm = torch.linalg.vector_norm(released_delta).item()
        gated_norm = torch.linalg.vector_norm(gated_delta).item()
        diag = self._empty_gate_diag()
        diag.update({
            'global_rate': float(global_rate),
            'r_lv': float('nan'), 'r_myo': float('nan'), 'r_rv': float('nan'),
            'alpha_lv': float(alpha_foreground[0].detach().cpu().item()),
            'alpha_myo': float(alpha_foreground[1].detach().cpu().item()),
            'alpha_rv': float(alpha_foreground[2].detach().cpu().item()),
            'alpha_map_mean': float(alpha_map.mean().item()), 'alpha_map_std': float(alpha_map.std(unbiased=False).item()),
            'alpha_map_min': float(alpha_map.min().item()), 'alpha_map_max': float(alpha_map.max().item()),
            'fraction_alpha_clipped_0': float((alpha_map <= 1e-12).float().mean().item()),
            'fraction_alpha_clipped_1': float((alpha_map >= 1 - 1e-12).float().mean().item()),
            'a_fallback': bool(fallback), 'correction_norm_released': float(released_norm),
            'correction_norm_gated': float(gated_norm),
            'correction_norm_ratio': float(gated_norm / (released_norm + 1e-8)),
        })
        for offset, suffix in enumerate(('lv', 'myo', 'rv'), start=1):
            weights = probability_feature[offset]
            denominator = weights.sum() + 1e-8
            released_class = (weights * torch.linalg.vector_norm(released_delta[0], dim=0)).sum() / denominator
            gated_class = (weights * torch.linalg.vector_norm(gated_delta[0], dim=0)).sum() / denominator
            diag[f'corr_released_{suffix}'] = float(released_class.item())
            diag[f'corr_gated_{suffix}'] = float(gated_class.item())
        if not fallback:
            scale_map = alpha_map / (float(global_rate) + 1e-8)
            diag['scale_map_mean'] = float(scale_map.mean().item())
            diag['scale_map_max'] = float(scale_map.max().item())
        return diag

    def _apply_exp4_gate(self, current, released, anchor_probability, global_rate,
                         gate_mode, gamma):
        # EXP4_GATE: centered soft class modulation is the only changed adaptation operation.
        probability = anchor_probability[0].detach()
        masses = probability[1:].sum(dim=(1, 2))
        reliability = probability[1:].square().sum(dim=(1, 2)) / (masses + 1e-8)
        reliability_bar = (masses * reliability).sum() / (masses.sum() + 1e-8)
        global_tensor = torch.as_tensor(float(global_rate), device=current.device)
        if gate_mode == 'identity':
            alpha_foreground = torch.full((3,), global_tensor.item(), device=current.device)
        elif gate_mode == 'lowconf':
            alpha_foreground = torch.clamp(
                global_tensor + gamma * (reliability_bar - reliability), min=0.0, max=1.0)
        else:
            alpha_foreground = torch.clamp(
                global_tensor - gamma * (reliability_bar - reliability), min=0.0, max=1.0)
        if gate_mode == 'identity':
            gated = released
            fallback = False
        elif float(global_rate) <= 1e-6:
            gated = released
            fallback = True
        else:
            probability_feature = F.interpolate(
                probability.unsqueeze(0), size=current.shape[-2:], mode='bilinear', align_corners=False)[0]
            alpha_values = torch.cat((global_tensor.view(1), alpha_foreground))
            alpha_map = (probability_feature * alpha_values.view(-1, 1, 1)).sum(dim=0)
            scale_map = alpha_map / (global_tensor + 1e-8)
            gated = current + scale_map.unsqueeze(0).unsqueeze(0) * (released - current)
            fallback = False
        diag = self._diagnose_gate(current, released, anchor_probability, global_rate,
                                   alpha_foreground, gated, fallback)
        diag['r_lv'] = float(reliability[0].item())
        diag['r_myo'] = float(reliability[1].item())
        diag['r_rv'] = float(reliability[2].item())
        diag['r_bar'] = float(reliability_bar.item())
        return gated, diag

    def _build_class_prototypes(self, feature_map, anchor_probability):
        # EXP2_DIAG: deterministic semantic weighting for offline diagnostics only.
        feature_map = feature_map[0]
        probability = F.interpolate(anchor_probability, size=feature_map.shape[-2:],
                                    mode='bilinear', align_corners=False)[0]
        feature_flat = feature_map.reshape(feature_map.shape[0], -1).transpose(0, 1)
        prototypes, masses = [], []
        for class_id in (1, 2, 3):
            weights = probability[class_id].reshape(-1)
            mass = weights.sum()
            prototype = (weights.unsqueeze(1) * feature_flat).sum(0) / (mass + 1e-8)
            prototypes.append(F.normalize(prototype, dim=0))
            masses.append(mass)
        return torch.stack(prototypes), torch.stack(masses)

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model



class Prototype_Pool(nn.Module):
    def __init__(self, delta=0.1, class_num=10, max=50):
        super(Prototype_Pool, self).__init__()
        self.class_num=class_num
        self.max_length = max
        self.feature_bank = torch.tensor([]).cuda()
        self.image_bank = torch.tensor([]).cuda()
        self.mask_bank = torch.tensor([]).cuda()
        self.name_list = []
        # EXP2_DIAG: parallel metadata bank; never consumed by model retrieval.
        self.class_prototype_bank = None
        self.class_mass_bank = None
        # EXP4_GATE: expose the exact Released global SFF rate without changing fusion.
        self.last_fusion_diag = {'global_rate': 0.0, 'topk_count': 0}
        # EXP6_CRSFF: metadata banks mirror the actual released pool behavior.
        self.class_reliability_bank = torch.empty((0, 3)).cuda()
        self.class_soft_mass_bank = torch.empty((0, 3)).cuda()
        # EXP0_5_DIAG: retrieval metadata only; it is not consumed by the method.
        self.last_retrieval_diag = {'indices': [], 'similarities': []}
    def get_pool_feature(self, x, mask, top_k = 5):
        if len(self.feature_bank)>0:
            cosine_similarities = torch.nn.functional.cosine_similarity(x.unsqueeze(1), self.feature_bank.unsqueeze(0), dim=2)
            if self.feature_bank.shape[0] >= top_k:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :top_k]
            else:
                outall = cosine_similarities.argsort(dim=1, descending=True)[:, :self.feature_bank.shape[0]]
            # EXP0_5_DIAG: copy the exact selected indices/similarities.
            self.last_retrieval_diag = {
                'indices': outall[0].detach().cpu().tolist(),
                'similarities': cosine_similarities[0][outall[0]].detach().cpu().tolist(),
            }
            rates = cosine_similarities[0][outall[0]].mean(0)
            # EXP4_GATE: this is the same scalar used by the Released fusion equation below.
            self.last_fusion_diag = {
                'global_rate': float(rates.detach().cpu().item()),
                'topk_count': int(outall.shape[1]),
                'topk_indices': outall[0].detach().cpu().tolist(),
                'topk_similarities': cosine_similarities[0][outall[0]].detach().cpu().tolist(),
            }
            weight = rates * torch.exp(cosine_similarities[0][outall[0]]) / torch.sum(torch.exp(cosine_similarities[0][outall[0]]))
            x = x * (1-rates)
            for i in range(min(top_k,self.feature_bank.shape[0])):
                x += self.feature_bank[outall[:,i]]*weight[i]
            return x,self.feature_bank[outall[:,]],self.image_bank[outall[:,]],self.mask_bank[outall[:,]], len(self.feature_bank)
        else:
            # EXP0_5_DIAG: an empty pool has no retrieval ranks.
            self.last_retrieval_diag = {'indices': [], 'similarities': []}
            self.last_fusion_diag = {'global_rate': 0.0, 'topk_count': 0,
                                     'topk_indices': [], 'topk_similarities': []}
            return x,x,None,None, len(self.feature_bank)

    def update_feature_pool(self, feature):
        if self.feature_bank.shape[0] == 0:
            self.feature_bank = torch.cat([self.feature_bank, feature.detach()],dim=0)
        else:
            if self.feature_bank.shape[0] < self.max_length:
                self.feature_bank = torch.cat([self.feature_bank, feature.detach()],dim=0)
            else:
                self.feature_bank = torch.cat([self.feature_bank[-self.max_length:], feature.detach()],dim=0)
    def update_image_pool(self, image):
        if self.image_bank.shape[0] == 0:
            self.image_bank = torch.cat([self.image_bank, image.detach()],dim=0)
        else:
            if self.image_bank.shape[0] < self.max_length:
                self.image_bank = torch.cat([self.image_bank, image.detach()],dim=0)
            else:
                self.image_bank = torch.cat([self.image_bank[-self.max_length:], image.detach()],dim=0)
    def update_mask_pool(self, image):
        if self.mask_bank.shape[0] == 0:
            self.mask_bank = torch.cat([self.mask_bank, image.detach()],dim=0)
        else:
            if self.mask_bank.shape[0] < self.max_length:
                self.mask_bank = torch.cat([self.mask_bank, image.detach()],dim=0)
            else:
                self.mask_bank = torch.cat([self.mask_bank[-self.max_length:], image.detach()],dim=0)
    def update_name_pool(self, image):
        if len(self.name_list) == 0:
            self.name_list.append(image)
        else:
            if len(self.name_list) < self.max_length:
                self.name_list.append(image)
            else:
                self.name_list = self.name_list[-self.max_length:]
                self.name_list.append(image)

    def update_diagnostic_prototype_pool(self, prototypes, masses):
        # EXP2_DIAG: mirror the released FIFO ordering without affecting outputs.
        prototypes = prototypes.detach().unsqueeze(0)
        masses = masses.detach().unsqueeze(0)
        if self.class_prototype_bank is None:
            self.class_prototype_bank = prototypes
            self.class_mass_bank = masses
        elif self.class_prototype_bank.shape[0] < self.max_length:
            self.class_prototype_bank = torch.cat([self.class_prototype_bank, prototypes], dim=0)
            self.class_mass_bank = torch.cat([self.class_mass_bank, masses], dim=0)
        else:
            self.class_prototype_bank = torch.cat([self.class_prototype_bank[-self.max_length:], prototypes], dim=0)
            self.class_mass_bank = torch.cat([self.class_mass_bank[-self.max_length:], masses], dim=0)

    def update_reliability_pool(self, reliability, soft_mass):
        # EXP6_CRSFF: append reliability metadata with the same max=40/actual-41 FIFO behavior.
        reliability = reliability.detach().reshape(1, 3)
        soft_mass = soft_mass.detach().reshape(1, 3)
        if self.class_reliability_bank.shape[0] < self.max_length:
            self.class_reliability_bank = torch.cat([self.class_reliability_bank, reliability], dim=0)
            self.class_soft_mass_bank = torch.cat([self.class_soft_mass_bank, soft_mass], dim=0)
        else:
            self.class_reliability_bank = torch.cat([self.class_reliability_bank[-self.max_length:], reliability], dim=0)
            self.class_soft_mass_bank = torch.cat([self.class_soft_mass_bank[-self.max_length:], soft_mass], dim=0)

    def validate_synchronized_banks(self):
        # EXP6_CRSFF: stop immediately if metadata and released memory banks diverge.
        lengths = [self.feature_bank.shape[0], self.image_bank.shape[0], self.mask_bank.shape[0],
                   len(self.name_list), self.class_reliability_bank.shape[0], self.class_soft_mass_bank.shape[0]]
        if len(set(int(length) for length in lengths)) != 1:
            raise RuntimeError(f'EXP6_CRSFF bank desynchronization: {lengths}')
