# Weed AI

The project pipeline worked as following:


### 1. Training resnet models

To easily implement High-Resolution ResNets, Keras is used. 

The code for model construction is found under `models.py`.

The pipeline for training is given as follows

&nbsp;&nbsp;&nbsp;&nbsp; 1.1 - Models are constructed by calling `models.py`

&nbsp;&nbsp;&nbsp;&nbsp;  1.2 - After training, called by `train_model.py`, trained models are saved as `.h5` in the folder `./Results/Saved_Model_As_h5'`

&nbsp;&nbsp;&nbsp;&nbsp;  1.3 - Training losses are saved in the folder `./Results/Training_Metrics.csv`. 

