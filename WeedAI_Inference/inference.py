import tensorflow as tf
from PIL import Image
import numpy as np
from utils import *


# Inferencing using INT8 Frozen Graph 
IMAGE_FILE = './data/201x201_TestSetMidFull/DSC03534.JPG'
TRT_FROZEN_GRAPH_FILE_INT8 = './tmp/trt_frozen_graph_int8/trt_graph.pb'
a

def main():
    elapsed = benchmark_trt_model_int8(save_inference=True)
    print(elapsed)

if __name__ == "__main__":
    main()