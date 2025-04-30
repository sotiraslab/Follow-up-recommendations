# Follow-up-recommendations

## TextCNN + Hybrid

1. **First step**. Download [model directory](https://drive.google.com/drive/folders/13-Z3LSTABo1o18RnE0jbIkrzIexn_K7W?usp=drive_link) and put it under './textcnn_hybrid/'. The directory for the model's checkpoint should appear as follows: './textcnn_hybrid/Radreport_Impression_f1/checkpoints/textcnn_impression/...'

2. **Second step**. Set up the environments. `conda env create -f textcnn.yml`.

3. **Three step**. Run `python visualize_lime.py`. Note that `x_test` in the `visualize_lime.py` contains a list of impression sections of radiology reports. So you should put the impression section of your radiology reports into the `x_test` to get the lime visualization results.