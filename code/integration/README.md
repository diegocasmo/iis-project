# Integration with Final Pipeline 

## Instructions
This folder contains Jupyter notebooks and python file implementing our part in the final integration. We are using ``pca-model.pkl`` model to read the input from Computer Vision group to apply the same amount of dimensionality reduction which we applied to our data while training our model. We were expecting an array with 25 landmarks from CV group. After dimensionality reduction we use ``svm-model.pkl`` to load our trained model and generate output which we need to paas to Emotion Synthesis Group. We are generating output in a dictionary format where each label act as a key and the value denotes the confidence score of the label by our model.

## Results
  - Example of above pipeline can be checked here in the [predict_emotion.ipynb](https://github.com/diegocasmo/iis-project/blob/integration/code/integration/predict_emotion.ipynb) notebook
