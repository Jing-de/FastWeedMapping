from __future__ import print_function
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Add, \
    Activation, GlobalAveragePooling2D, BatchNormalization, Cropping2D
import keras_resnet.models
import numpy as np
np.set_printoptions(suppress=True)


def setup_model_resnet_manual_highres_center_only(input_shape, num_classes, filtersize1, filtersize2):
    input1 = Input(shape=input_shape,name="input1")

    x = Conv2D(filtersize1, kernel_size=(7,7), padding="same")(input1)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=(2,2),name="pool1")(x)
    x2 = Conv2D(filtersize1, kernel_size=(3,3), padding="same", name="res2a_branch2a")(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x2 = Conv2D(filtersize1, kernel_size=(3,3), padding="same", name="res2a_branch2b")(x2)
    x1 = Conv2D(filtersize1, kernel_size=(1,1), padding="same", name="res2a_branch1")(x)
    x2 = BatchNormalization()(x2)
    x1 = BatchNormalization()(x1)
    x = Add()([x2,x1])
    x = Activation("relu")(x)

    x2 = Conv2D(filtersize1, kernel_size=(3,3), padding="same", name="res2b_branch2a")(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x2 = Conv2D(filtersize1, kernel_size=(3,3), padding="same", name="res2b_branch2b")(x2)
    x2 = BatchNormalization()(x2)
    x = Add()([x,x2])
    x = Activation("relu")(x)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), strides=2, padding="same", name="res3a_branch2a")(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), padding="same", name="res3a_branch2b")(x2)
    x1 = Conv2D(filtersize2, kernel_size=(1,1), strides=2, padding="same", name="res3a_branch1")(x)
    x2 = BatchNormalization()(x2)
    x1 = BatchNormalization()(x1)
    x = Add()([x1,x2])
    x = Activation("relu")(x)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), padding="same", name="res3b_branch2a")(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), padding="same", name="res3b_branch2b")(x2)
    x2 = BatchNormalization()(x2)
    x = Add()([x,x2])
    x = Activation("relu")(x)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), strides=1, padding="same", name="res4a_branch2a")(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), padding="same", name="res4a_branch2b")(x2)
    x1 = Conv2D(filtersize2, kernel_size=(1,1), strides=1, padding="same", name="res4a_branch1")(x)
    x2 = BatchNormalization()(x2)
    x1 = BatchNormalization()(x1)
    x = Add()([x1,x2])
    x = Activation("relu")(x)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), padding="same", name="res4b_branch2a")(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), padding="same", name="res4b_branch2b")(x2)
    x2 = BatchNormalization()(x2)
    x = Add()([x,x2])
    x = Activation("relu")(x)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), strides=1, padding="same", name="res5a_branch2a")(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), padding="same", name="res5a_branch2b")(x2)
    x1 = Conv2D(filtersize2, kernel_size=(1,1), strides=1, padding="same", name="res5a_branch1")(x)
    x2 = BatchNormalization()(x2)
    x1 = BatchNormalization()(x1)
    x = Add()([x1,x2])
    x = Activation("relu")(x)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), padding="same", name="res5b_branch2a")(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x2 = Conv2D(filtersize2, kernel_size=(3,3), padding="same", name="res5b_branch2b")(x2)
    x2 = BatchNormalization()(x2)
    x = Add()([x,x2])
    x = Activation("relu",name="before_pooling")(x)

    x = Cropping2D(cropping=((15,15), (15,15)), input_shape=(None,50,50,32))(x)
    x = GlobalAveragePooling2D(name="pool5")(x)
    predictions = Dense(num_classes, activation="softmax", name="fc2")(x)
    
    model = Model(inputs=[input1], outputs=[predictions])
    lr = 0.01
    return model, lr