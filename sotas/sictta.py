import torch.nn.functional as F
import torch
import torch.nn as nn

class TTA(nn.Module):
    """TTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, model_anchor, use_test_bn=True,
                 use_sabe=True, use_sff=True):
        super().__init__()
        self.model = model
        self.model_anchor = model_anchor.eval()
        # EXP0_REPRO: keep the released path as the default and expose only
        # the paper Table 5 component switches needed by EXP-0.
        self.use_test_bn = use_test_bn
        self.use_sabe = use_sabe
        self.use_sff = use_sff

        self.num_classes = 4
        self.max_lens = 40
        self.topk = 5
        self.threshold = 0.9
        self.entropy_list = []
        self.pool = Prototype_Pool(0.1,class_num=self.num_classes,max = self.max_lens).cuda()
        # EXP0_5_DIAG: read-only state populated from existing intermediates.
        self.last_diag = {}
        self.last_ccd_diag = {}
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

        if fine:
            self.pool.update_feature_pool(latent_model)
            self.pool.update_image_pool(x)
            self.pool.update_mask_pool(model(x).softmax(1))
            self.pool.update_name_pool(names[0])
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
            if self.use_sff:
                latent_model[0:1] = latent_model_
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
        self.last_diag = {
            'ccd': ccd_diag.get('ccd'),
            'ccd_threshold': ccd_diag.get('ccd_threshold'),
            'ccd_margin': (ccd_diag.get('ccd_threshold') - ccd_diag.get('ccd')
                           if ccd_diag.get('ccd_threshold') is not None else None),
            'is_sft': bool(ccd_diag.get('fine', False)),
            'ccd_history_len_before': history_before,
            'ccd_history_len_after': len(self.entropy_list),
            'pool_before': pool_before,
            'pool_after': pool_after,
            # EXP0_5_DIAG: distinguish an actual FIFO write from pool growth.
            'memory_written': bool(ccd_diag.get('fine', False)),
            'pool_grew': pool_after['name'] > pool_before['name'],
            'retrieved_indices': list(retrieval_diag.get('indices', [])),
            'retrieved_similarities': list(retrieval_diag.get('similarities', [])),
            'retrieved_names': retrieval_names,
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
        with torch.no_grad():
            b,c,w,h = x.shape
            for i in range(b):
                with torch.no_grad():
                    pred1 = model_anchor(x[i:i+1]).softmax(1).detach()
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
            weight = rates * torch.exp(cosine_similarities[0][outall[0]]) / torch.sum(torch.exp(cosine_similarities[0][outall[0]]))
            x = x * (1-rates)
            for i in range(min(top_k,self.feature_bank.shape[0])):
                x += self.feature_bank[outall[:,i]]*weight[i]
            return x,self.feature_bank[outall[:,]],self.image_bank[outall[:,]],self.mask_bank[outall[:,]], len(self.feature_bank)
        else:
            # EXP0_5_DIAG: an empty pool has no retrieval ranks.
            self.last_retrieval_diag = {'indices': [], 'similarities': []}
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
