# Large-scale evaluation of machine learning models in identifying follow-up recommendations in radiology reports

This is the official code repository for the manuscript "Large-scale evaluation of machine learning models in identifying follow-up recommendations in radiology reports".

![alt text](./figures/Figure_1.jpg)
**Study flowchart.** *(A) Three sets of radiology reports were gathered: the first one extracted through two regular expressions (Section S1), the second was annotated during dictation, while the third was annotated by three radiologists. The training set was created by combining the dataset extracted using regular expressions with 20% of reports randomly selected from the two manually annotated corpora. The remaining 80% of radiologist-annotated reports was divided into validation and test sets with a ratio of 1:4. (B) Reports lacking a finding section from the training, validation, and test sets were excluded from the analysis, as this section was missing. (C) Any reports from the training, validation, and test sets that did not include an impression section were eliminated from consideration. (D) The impression sections of 2,000 MIMIC-CXR reports were collected as an external test set. (E) The impression sections 100 CT institutional reports were gathered at a different time point to serve as a temporal test set.*


## TextCNN + Hybrid

1. **First step**. Download [model directory](https://drive.google.com/drive/folders/13-Z3LSTABo1o18RnE0jbIkrzIexn_K7W?usp=drive_link) and put it under './textcnn_hybrid/'. The directory for the model's checkpoint should appear as follows: './textcnn_hybrid/Radreport_Impression_f1/checkpoints/textcnn_impression/...'

2. **Second step**. Set up the environments. `conda env create -f textcnn.yml`.

3. **Three step**. Run `python visualize_lime.py`. Note that `x_test` in the `visualize_lime.py` contains a list of impression sections of radiology reports. So you should put the impression section of your radiology reports into the `x_test` to get the lime visualization results.