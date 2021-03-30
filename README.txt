# Weed AI

The project pipeline worked as following:


### 1. Training resnet models

To easily implement High-Resolution ResNets, Keras is used. 

The code for model construction is found under the file `models.py`.

The pipeline for training is given as follows

1.1 - Models are constructed by calling `models.py`

1.2 - After training, called by `train_model.py`, trained models are saved as `.h5` in the folder `./Results/Saved_Model_As_h5'`

1.3 - Training losses are saved in the folder `./Results/Training_Metrics.csv`. 
2.

### 2. Generating benchmarks with NVIDIA Jetson


2.1 - Use script `from_h5_to_frozen_graph.py` to load the `.h5` model and further convert it into `flex_model`

All converted frozen graphs are going to be saved within `./Models/Frozen_graphs/*`

**Important Remark:** The vanilla ResNet is converted to `flex model` here. Thus, the exported frozen graph is already a `flex model` on full-precision mode (FP32)

2.2 - Call `predict_on_full_images.py` to generate annotations, inference times and confusion matrices. You can give the command line argument `-p FP16` to use half-precision or `-p FP32` for full-precision. All results generated at this stage are saved within `./Results/seeds/*`


#### Generate quick inferences, using half-precision, on one image

For debugging purposes, the script `half_precision_quick_inferences.py` was created to perform quick inferences and test inferences and how long it took for an inference to be complete. 

