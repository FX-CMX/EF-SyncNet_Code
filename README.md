


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
<img width="1085" height="603" alt="Fig4_EF-SyncNet" src="https://github.com/user-attachments/assets/66e3ce30-f5d1-40b1-b361-c488a879d788" />
*Fig1: Schematic diagram of the proposed EF-SyncNet architecture.

## 2. Core Modules

### 2.1 Edge-Guided Multi-Scale Context Module (EGMC)
This module incorporates gradient priors to capture spatial-domain edge information, effectively alleviating the over-smoothing effect caused by downsampling and preserving the structural integrity of defects.
<img width="1166" height="638" alt="Fig5_EGMC" src="https://github.com/user-attachments/assets/98bfbe5c-586e-4fc1-a77c-2d45a36180d5" />
*Fig2:Schematic diagram of the proposed EGMC architecture.

### 2.2 Wavelet-Guided Frequency Attention Module (WGFA)
This module performs frequency-domain decoupling via wavelet transform to refine defect texture features while suppressing noise interference in complex backgrounds.
<img width="662" height="243" alt="Fig7_EFCA" src="https://github.com/user-attachments/assets/21fafe34-4184-4a99-b2d6-8767ddb7d06d" />

*Fig3:Schematic diagram of the proposed WGFA architecture.

### 2.3 Edge-Frequency Cross-Attention Module (EFCA)
This module utilizes edge priors as spatial anchors to constrain frequency-domain features, ensuring that semantic information is focused within defect boundaries and preventing the dispersion of background semantics.
<img width="655" height="245" alt="EFCA" src="https://github.com/user-attachments/assets/c528849e-b2d5-420c-b9d0-b305939fb5c7" />
*Fig4:Schematic diagram of the proposed EFCA architecture.

### 2.4 Local Frequency Tuning Module (LFT)
This module adaptively calibrates feature representations to eliminate artifacts introduced by multi-scale interpolation in the decoder, further enhancing the local details of the segmentation masks.
<img width="1177" height="531" alt="LFT" src="https://github.com/user-attachments/assets/42afbd68-e0cf-493b-ab21-00124d5238ec" />
*Fig5:Schematic diagram of the proposed LFT architecture.
