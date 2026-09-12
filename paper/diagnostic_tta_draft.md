# When Reliability Does Not Translate to Safe Adaptation: Diagnosing Granularity Mismatch in Continual Test-Time Medical Image Segmentation

> Working draft for an ICASSP-length paper. Numerical claims are linked to the local experiment reports in `results/`. Citation numbers are placeholders until the bibliography is finalized.

## Abstract

Continual test-time adaptation (CTTA) improves medical image segmentation under distribution shift, but an adaptation mechanism that is beneficial on average can still degrade individual anatomical structures. We present a controlled diagnostic study of reliability-guided single-image CTTA on the M&MS cardiac MRI benchmark. Starting from a reproduced SicTTA pipeline, we isolate admission history, memory retrieval, class reliability, spatial alignment, local correspondence, feature-fusion weighting, normalization statistics, and post-adaptation output selection while preserving the released inference path through prediction-hash regression tests. The adapted prediction improves the anchor prediction by 0.095 Dice on average over ground-truth-present class instances, yet 20.52% of those instances undergo negative adaptation and 11.73% lose more than one Dice point. A post-adaptation confidence delta is strongly associated with adaptation gain (Spearman $\rho=0.707$) and detects one-point harm with an AUROC of 0.749. However, converting this offline ranking signal into online intervention does not yield safe segmentation: class-aware confidence-delta output selection decreases overall Dice by 0.54--0.68 points, with the largest degradation on the thin-walled right ventricle. Excluding RV from the gate reduces but does not reverse the loss ($-0.139$ points overall; $-0.186$ points averaged over LV/MYO). Other plausible interventions—class gating, reliability-weighted retrieval, spatial alignment, local feature matching, and similarity-weighted normalization—also fail to produce practically meaningful gains. These results expose a granularity mismatch: a proxy can predict aggregate class-level utility while remaining unsuitable for localized intervention at anatomically sensitive structures. Our study provides a reproducible evaluation protocol and negative-result evidence for separating proxy predictiveness from intervention utility in medical-image CTTA.

## 1. Introduction

Test-time adaptation (TTA) modifies a source-trained model or its inference behavior using unlabeled target samples. In medical image segmentation, this is attractive because scanner, protocol, and institution shifts are common while target labels are unavailable. Continual and single-image settings are particularly challenging: the method must adapt online, cannot rely on a large target batch, and may accumulate errors through its state.

Most reliability-guided TTA methods implicitly assume that a proxy correlated with prediction quality can safely control adaptation. This assumption contains two distinct claims:

1. **Predictiveness:** the proxy ranks reliable and unreliable outcomes.
2. **Intervention utility:** acting on that ranking improves the final structured prediction.

The second claim does not follow from the first. Segmentation is spatially coupled, classes compete through the output simplex, and thin or boundary-dominated structures may be changed by a very small probability perturbation. A selector with a useful AUROC can therefore reduce Dice when used to suppress, blend, or roll back local predictions.

We study this gap through a sequence of controlled experiments on SicTTA [1]. The released pipeline combines source-free memory admission, global Top-$K$ retrieval, source-aware feature fusion (SFF), and source-aware batch enhancement (SABE). Rather than introducing a large learned module, we test common reliability-driven modifications one at a time and require every identity configuration to reproduce the released prediction hash.

Our contributions are:

- We separate reliability-proxy predictiveness from downstream intervention utility in single-image continual TTA.
- We provide controlled evidence across admission, retrieval, fusion, spatial correspondence, normalization, and post-adaptation selection, showing that several plausible modifications do not yield stable Dice gains.
- We identify a failure mode concentrated in anatomically sensitive structures: a confidence-delta selector modifies less than 0.1% average pixel mass yet reduces right-ventricle Dice by more than one point.
- We release a regression-tested diagnostic protocol in which ground truth is evaluation-only and the released adaptation path is verified by per-slice outputs and SHA-256 prediction hashes.

## 2. Problem Formulation

Let $P^A(x)$ denote the anchor/source prediction and $P^T(x;\mathcal{M})$ the released adapted prediction using continual memory state $\mathcal{M}$. For foreground class $c$, adaptation utility is

$$
G_c(x)=D_c(P^T(x),y)-D_c(P^A(x),y),
$$

where $D_c$ is class Dice and $y$ is used only after inference for evaluation. Negative adaptation occurs when $G_c(x)<0$; severe harm is defined as $G_c(x)<-0.01$.

The class-confidence proxy used throughout the study is

$$
r_c(P)=\frac{\sum_u P_c(u)^2}{\sum_u P_c(u)+\epsilon}.
$$

The post-adaptation confidence delta is

$$
d_c(x)=r_c(P^T(x))-r_c(P^A(x)).
$$

EXP-5 evaluates whether $d_c$ predicts $G_c$. EXP-9 evaluates the stronger and practically relevant claim: whether an online function of $d_c$ can improve the final segmentation without labels.

## 3. Controlled Diagnostic Protocol

### 3.1 Dataset and continual stream

We use the processed 2-D M&MS cardiac MRI benchmark. Domain A is the source domain; target slices arrive without shuffling in the continual order B $\rightarrow$ C $\rightarrow$ D. The stream contains 4,091 slices from 500 patient-frame volumes. Foreground classes are left ventricle (LV), myocardium (MYO), and right ventricle (RV). Unless stated otherwise, primary Stage-1 results use seed 2026 and batch size one.

### 3.2 Invariants

Each diagnostic freezes the source checkpoint, stream order, preprocessing, released memory capacity, Top-$K$, SFF/SABE implementation, and metric code unless that component is the isolated intervention. Diagnostic ground truth never affects admission, retrieval, fusion, normalization, or output selection.

For all identity/released controls, we verify:

- identical slice names and order;
- per-class Dice equality within $10^{-8}$;
- identical prediction SHA-256;
- unchanged CCD decisions, SFT admission, and memory-pool sizes when applicable.

### 3.3 Hypothesis sequence

| Study | Isolated hypothesis | Main observation | Decision |
|---|---|---|---|
| EXP-0/0.5 | The released method and component trends are reproducible; memory behavior matches the nominal configuration. | SABE+SFF is best; CCD storage reaches 4,091 while the pool reaches 41. | Reproduction succeeds; diagnose admission history. |
| EXP-1 | Stale CCD history is the main bottleneck. | Rolling-160 gains only 0.055 points; domain reset gains 0.011 points. | Not the main bottleneck. |
| EXP-2 | Class-wise semantic retrieval yields consistently better memories. | Neighbor-quality changes are $-0.39/+0.73/+1.61$ points for LV/MYO/RV and are not multi-seed stable. | Do not implement prototype memory. |
| EXP-3 | A class-wise proxy predicts anchor segmentation quality. | Class confidence reaches $\rho=0.837/0.851/0.900$ for LV/MYO/RV. | Predictiveness supported. |
| EXP-4 | Prediction reliability should modulate SFF strength. | Best single-seed gain is approximately 0.03 points and disappears across seeds. | No practically meaningful gain. |
| EXP-5 | Reliability proxies predict adaptation utility. | Confidence delta reaches $\rho=0.707$ and harm AUROC 0.749; oracle headroom remains. | Utility signal exists; intervention unproven. |
| EXP-6 | Reliability-weighted memory fusion improves SFF. | Class weighting gains 0.014 points, below the 0.15-point Stage-1 threshold. | Stop expansion. |
| EXP-6.5/7 | Spatial or local correspondence explains SFF failure. | Translation and LocalNN do not improve correspondence reliably. | Stop SFF modification. |
| EXP-8 | Batch-normalization shift explains harmful SABE. | Shift is weakly positively, not negatively, associated with SABE utility. | Stop normalization modification. |
| EXP-9 | Post-adaptation confidence delta can safely select anchor/adapted outputs. | All tested online selectors reduce Dice; RV is most affected. | Stop naive confidence-delta selection. |

## 4. Post-Adaptation Output Selection

The EXP-9 selector executes released SicTTA exactly once and observes $P^A$ and $P^T$, both already available in the inference path. It maintains a class-wise, past-only window $H_c$ of confidence deltas and computes

$$
z_c=\frac{d_c-\operatorname{median}(H_c)}{1.4826\operatorname{MAD}(H_c)+\epsilon}.
$$

We evaluate three label-free rules: a zero-threshold hard gate, a robust lower-tail hard gate, and a robust lower-tail soft gate. Class decisions are converted to a valid spatial convex mixture using the anchor posterior:

$$
g(u)=\sum_c P_c^A(u)g_c,
$$

$$
P^{S}(u)=g(u)P^T(u)+(1-g(u))P^A(u).
$$

The selector is output-only: memory admission and all future SicTTA states remain unchanged. This design tests intervention utility without confounding it with a changed continual trajectory.

## 5. Results

### 5.1 Reproduction and adaptation headroom

The reproduced SABE+SFF model obtains 77.60% domain-macro Dice and 77.90% slice-weighted Dice. Across 11,462 ground-truth-present class instances, mean adapted-minus-anchor gain is 0.095 Dice, but 20.52% are harmed and 11.73% lose more than one point. Class-metric and whole-slice oracles indicate +0.663 and +1.175 points of headroom, respectively. These oracle values quantify recoverable error but do not define a realizable multiclass output.

### 5.2 Proxy predictiveness

Class confidence strongly predicts anchor quality, with Spearman correlations of 0.837, 0.851, and 0.900 for LV, MYO, and RV. Confidence delta is the strongest tested post-adaptation utility signal: $\rho=0.707$, harm AUROC 0.749, and harm AUPRC 0.330. In contrast, the strongest pre-adaptation risk signal reaches only AUROC 0.587.

### 5.3 Predictiveness does not imply intervention utility

| Selector | LV Dice | MYO Dice | RV Dice | Average Dice | $\Delta$ vs. released |
|---|---:|---:|---:|---:|---:|
| Released | 86.281 | 76.303 | 71.125 | 77.903 | 0.000 |
| Zero hard | 85.788 | 75.801 | 70.082 | 77.224 | -0.679 |
| MAD hard | 85.955 | 76.180 | 69.859 | 77.331 | -0.572 |
| MAD soft | 85.983 | 76.191 | 69.917 | 77.364 | -0.539 |
| MAD soft, RV excluded | 85.999 | 76.212 | 71.080 | 77.764 | -0.139 |

MAD soft modifies 10.71% of slices but only 0.094% mean pixel mass. Despite this sparse intervention, RV Dice decreases by 1.208 points. At the case level, its mean change is -0.141 points with a 95% paired-bootstrap confidence interval of $[-0.211,-0.076]$.

To test whether RV alone accounts for the failure, we perform one time-boxed diagnostic in which the RV gate is fixed to the released path. The selector then modifies 5.18% of slices and 0.056% mean pixel mass. LV and MYO change by $-0.282$ and $-0.091$ points, respectively, for an LV/MYO mean of $-0.186$ points. Overall Dice changes by $-0.139$ points, and the case-level 95% paired-bootstrap interval is $[-0.184,-0.073]$. RV sensitivity therefore amplifies the original loss but is not its sole cause.

### 5.4 Granularity mismatch

The observed asymmetry is inconsistent with a simple lack-of-signal explanation. Confidence delta ranks class-level gain, but the intervention acts on spatially coupled probabilities. A small change near an argmax boundary can alter topology or remove a thin component while contributing negligible image-wide probability mass. The RV result is therefore evidence that proxy quality must be evaluated at the same granularity as the intended intervention.

The controlled sequence supports three distinctions:

- **Quality reliability is not utility reliability.** Class confidence predicts anchor Dice but does not identify when adaptation should be suppressed.
- **Utility ranking is not calibrated decision-making.** A useful AUROC does not specify a threshold with positive expected Dice change.
- **Class-level utility is not pixel-level safety.** Independent class-metric replacement overestimates what can be achieved by a valid multiclass probability mixture.

## 6. Discussion

### 6.1 Why the offline signal fails online

First, AUROC is invariant to monotonic score transformations and measures ranking rather than the cost of false interventions. Dice costs are highly asymmetric: reverting one beneficial thin boundary may outweigh correcting multiple small false-positive regions. Second, the class-confidence statistic aggregates over a predicted region and loses boundary location. Third, multiclass probabilities are coupled; locally mixing anchor and adapted predictions changes competition among all classes, even when the gate was triggered by one class.

### 6.2 Implications for TTA evaluation

Reliability-guided TTA should report both proxy-level metrics and intervention-level outcomes. Correlation, AUROC, and oracle headroom are insufficient evidence for a deployable gate. At minimum, studies should include an identity-equivalent released control, a same-budget random intervention, class- and anatomy-stratified outcomes, and case-level paired confidence intervals.

### 6.3 Negative results as design constraints

Our findings do not imply that memory, alignment, normalization, or selective adaptation can never help. They establish narrower constraints for this setting: global or class-aggregated proxies cannot be assumed to support localized intervention; correspondence diagnostics must distinguish label availability from feature-match recoverability; and normalization discrepancy is not equivalent to harmful utility.

## 7. Limitations

The study uses one segmentation architecture, one source domain, one fixed target-domain order, and a single medical dataset. Most Stage-1 no-go decisions use seed 2026, although selected earlier findings were checked across three seeds. The post-adaptation selector pays the cost of both anchor and adapted predictions and does not reduce inference time. Our diagnostics identify associations and intervention failures but do not prove a unique causal mechanism for boundary sensitivity. Finally, oracle class-metric results are not directly realizable as valid multiclass segmentations.

## 8. Conclusion

A reliability proxy can be statistically predictive yet operationally unsafe. In single-image continual cardiac-MRI segmentation, confidence delta strongly ranks adaptation utility, but hard and soft online output selection both reduce Dice, with disproportionate degradation on the right ventricle. Together with controlled negative results for admission, retrieval, fusion, correspondence, and normalization modifications, this study demonstrates a granularity mismatch between aggregate reliability signals and anatomy-sensitive segmentation interventions. Future TTA methods should validate not only whether a proxy predicts error, but whether acting on it improves the structured output under a regression-tested continual protocol.

## Evidence map

| Claim | Local evidence |
|---|---|
| SicTTA reproduction | `results/exp0/exp0_report.md` |
| Admission history | `results/exp1/exp1_report.md` |
| Class retrieval | `results/exp2/exp2_report.md` |
| Reliability proxies | `results/exp3/exp3_report.md` |
| Class gating | `results/exp4/exp4_report.md` |
| Utility and oracle headroom | `results/exp5/exp5_report.md` |
| Reliability-weighted SFF | `results/exp6/exp6_report.md` |
| Spatial/local correspondence | `results/exp6_5/exp6_5_report.md`, `results/exp7/exp7_report.md` |
| SABE normalization | `results/exp8/exp8_report.md` |
| Confidence-delta intervention | `results/exp9/exp9_report.md` |

## References (placeholders)

[1] J. Wu et al., “SicTTA: Single Image Continual Test Time Adaptation for Medical Image Segmentation,” *Medical Image Analysis*, 2026.

[2] D. Wang et al., “Tent: Fully Test-Time Adaptation by Entropy Minimization,” ICLR, 2021.

[3] Y. Zhou, J. Wu, W. Liao, S. Zhang, S. Zhang, and G. Wang, “TEGDA: Test-time Evaluation-Guided Dynamic Adaptation for Medical Image Segmentation,” in *Medical Image Computing and Computer Assisted Intervention (MICCAI)*, 2025, doi:10.1007/978-3-032-04978-0_60. TEGDA motivates confidence calibration because dropout agreement can be dominated by high-agreement interior regions and incorporates border discrepancy into its calibrated ADIC metric; this is related context, not evidence that our EXP-9 errors are spatially concentrated at boundaries.
