[![DOI](https://img.shields.io/badge/DOI-10.1148%2Fradiol.242167-blue.svg)](https://doi.org/10.1148/radiol.242167)

# [Large-scale evaluation of machine learning models in identifying follow-up recommendations in radiology reports](https://pubs.rsna.org/doi/10.1148/radiol.242167)

Introduction
=======
This is the official code repository for the manuscript "Large-scale evaluation of machine learning models in identifying follow-up recommendations in radiology reports".


<p align="center">
  <img src="./figures/Figure_1.jpg" alt="Figure 1" width="670"/>
</p>

<p style="max-width: 700px; margin: auto; text-align: justify;">
  <b>Figure 1.</b> Study flowchart. (A) Three sets of radiology reports were gathered: the first one extracted through two regular expressions—sequences of characters that define search patterns in text (Section S1), the second was annotated during dictation, while the third was annotated by three radiologists. All three datasets underwent deduplication to remove duplicate reports. The training set was created by combining the dataset extracted using regular expressions with reports from 20% of patients randomly selected from the two manually annotated datasets, ensuring that each patient was assigned to only one split. The remaining reports from 80% of patients in the radiologist-annotated datasets were divided into validation and internal test sets with a ratio of 1:4, maintaining patient-level separation. (B) Reports lacking a finding section from the training, validation, and internal test sets were excluded from the analysis, as this section was missing. (C) Any reports from the training, validation, and internal test sets that did not include an impression section were eliminated from consideration. (D) The impression sections of 2,000 MIMIC-CXR reports were collected as an external test set. (E) The impression sections 100 CT institutional reports were gathered at a different time point using newer data to serve as a temporal test set.
</p>

License
=======
- **Code**: [Apache 2.0](./LICENSE)
- **Model Weights**: [CC BY-NC 4.0](./MODEL_LICENSE)


Citation
=======
Cite our work:
```
@article{xiao2025large,
  title={Large-Scale Evaluation of Machine Learning Models in Identifying Follow-Up Recommendations in Radiology Reports},
  author={Xiao, Pan and Yu, Xiaobing and Ha, Sung Min and Bani, Abdalla and Mintz, Aaron and Wang, Jieqi and Elbanan, Mohamed and Mokkarala, Mahati and Mattay, Govind and Nazeri, Arash and others},
  journal={Radiology},
  volume={317},
  number={2},
  pages={e242167},
  year={2025},
  publisher={Radiological Society of North America}
}
```

Questions
=======
If you have any questions about our work, please contact us via email (panxiao@wustl.edu)