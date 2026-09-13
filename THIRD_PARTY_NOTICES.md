# Third-party models

This repository provides benchmark adapters based on the following publications and public implementations. The adapters share this repository's data interface, channel count, cross-validation, and optimization pipeline. They should not be described as byte-for-byte copies of upstream code.

## STAnet

E. Su, S. Cai, L. Xie, H. Li, and T. Schultz, "STAnet: A Spatiotemporal Attention Network for Decoding Auditory Spatial Attention From EEG," IEEE Transactions on Biomedical Engineering, 2022. DOI: [10.1109/TBME.2022.3140246](https://doi.org/10.1109/TBME.2022.3140246). Public implementation: [SCUT-IEL/STAnet](https://github.com/SCUT-IEL/STAnet).

## XANet

S. Pahuja, S. Cai, T. Schultz, and H. Li, "XAnet: Cross-Attention Between EEG of Left and Right Brain for Auditory Attention Decoding," IEEE International Neural Engineering Conference, 2023. DOI: [10.1109/NER52421.2023.10123792](https://doi.org/10.1109/NER52421.2023.10123792). No public author implementation was located when this release was prepared. The included paper-faithful adapter implements bilateral temporal encoding and bidirectional cross-attention using the fixed physical montage described in `docs/MODELS.md`.

## DARNet

S. Yan, C. Fan, H. Zhang, X. Yang, J. Tao, and Z. Lv, "DARNet: Dual Attention Refinement Network with Spatiotemporal Construction for Auditory Attention Detection," NeurIPS, 2024. Public implementation: [fchest/DARNet](https://github.com/fchest/DARNet).

## DBPNet

Q. Ni, H. Zhang, C. Fan, S. Pei, C. Zhou, and Z. Lv, "DBPNet: Dual-Branch Parallel Network with Temporal-Frequency Fusion for Auditory Attention Detection," IJCAI, 2024. DOI: [10.24963/ijcai.2024/345](https://doi.org/10.24963/ijcai.2024/345). Public implementation: [fchest/DBPNet](https://github.com/fchest/DBPNet), adapted from commit `99a153fdbb8276f227ac6393529a24af9afc5dbc`. CSP is fitted inside each training fold, and the frequency maps use fixed coordinates for the custom frontal-periauricular montage.

## ListenNet

C. Fan, X. Yang, H. Zhang, Y. Chen, L. Li, J. Zhou, and Z. Lv, "ListenNet: A Lightweight Spatio-Temporal Enhancement Nested Network for Auditory Attention Detection," 2025. Preprint: [arXiv:2505.10348](https://arxiv.org/abs/2505.10348). Public implementation: [fchest/ListenNet](https://github.com/fchest/ListenNet), adapted from commit `286b8ac4bddc7e6baee870940457809131`. Euclidean alignment is estimated on the inner-training split and applied unchanged to validation and test data.

## MHANet

L. Li, C. Fan, H. Zhang, J. Zhang, X. Yang, J. Zhou, and Z. Lv, "MHANet: Multi-scale Hybrid Attention Network for Auditory Attention Detection," IJCAI, 2025. DOI: [10.24963/ijcai.2025/465](https://doi.org/10.24963/ijcai.2025/465). Public implementation: [fchest/MHANet](https://github.com/fchest/MHANet).

## HCAN

Y. Wen, S. Ma, C. Liu, and Y. Wang, "Hybrid channel attention network for auditory attention detection," Scientific Reports, 2025. DOI: [10.1038/s41598-025-22177-x](https://doi.org/10.1038/s41598-025-22177-x). The adapter follows the architecture and CSP input protocol reported in the paper.

Review the upstream repositories and publication terms before redistributing third-party material beyond these adapters.
