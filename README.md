<img width="564" height="670" alt="图片" src="https://github.com/user-attachments/assets/c1ddf5fa-ff15-49c7-8f3b-822d5e3fca3d" /># EF-SyncNet: An Edge–Frequency Synchronized Multi-scale Network

---

## 📜 License & Acknowledgments

**Copyright (c) 2026. All rights reserved.**

This project is licensed under the Apache License, Version 2.0. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

This implementation is derived from or inspired by the following open-source projects:
- [OpenMMLab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

We sincerely thank the authors for their contribution to the community. Please retain this attribution when redistributing or publishing works based on this project.

---


## 1. Model Architecture
The proposed EF-SyncNet leverages a collaborative synchronization of spatial-domain edge priors and frequency-domain features to achieve robust and fine-grained industrial defect segmentation.
<img width="1071" height="631" alt="EF-SyncNet" src="https://github.com/user-attachments/assets/9a4bc7e8-ae59-4023-8156-dd590ac8bb72" />
*Fig1: Schematic diagram of the proposed EF-SyncNet architecture.

## 2. Core Modules

### 2.1 Edge-Guided Multi-Scale Context Module (EGMC)
This module incorporates gradient priors to capture spatial-domain edge information, effectively alleviating the over-smoothing effect caused by downsampling and preserving the structural integrity of defects.
<img width="1165" height="628" alt="EGMC" src="https://github.com/user-attachments/assets/da88dbec-3a1b-4fde-8bfd-515117288e19" />
*Fig2:Schematic diagram of the proposed EGMC architecture.

### 2.2 Wavelet-Guided Frequency Attention Module (WGFA)
This module performs frequency-domain decoupling via wavelet transform to refine defect texture features while suppressing noise interference in complex backgrounds.
<img width="1163" height="702" alt="WGFA" src="https://github.com/user-attachments/assets/04b5a31f-6e90-4555-911a-9d8980809a55" />
*Fig3:Schematic diagram of the proposed WGFA architecture.

### 2.3 Edge-Frequency Cross-Attention Module (EFCA)
This module utilizes edge priors as spatial anchors to constrain frequency-domain features, ensuring that semantic information is focused within defect boundaries and preventing the dispersion of background semantics.
<img width="655" height="245" alt="EFCA" src="https://github.com/user-attachments/assets/c528849e-b2d5-420c-b9d0-b305939fb5c7" />
*Fig4:Schematic diagram of the proposed EFCA architecture.

### 2.4 Local Frequency Tuning Module (LFT)
This module adaptively calibrates feature representations to eliminate artifacts introduced by multi-scale interpolation in the decoder, further enhancing the local details of the segmentation masks.
<img width="1177" height="531" alt="LFT" src="https://github.com/user-attachments/assets/42afbd68-e0cf-493b-ab21-00124d5238ec" />
*Fig5:Schematic diagram of the proposed LFT architecture.
