import argparse
import time
import numpy as np
import glob
import os
import keras
import tensorflow as tf
from models import setup_model_resnet_manual_highres_center_only
from PIL import Image
from tensorflow.python.client import device_lib
from keras.callbacks import TensorBoard, CSVLogger


# GPU CONFIGURATION
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
print(device_lib.list_local_devices())
use_coordinates = False


# GLOBAL CONSTANTS
TRAINSET_DIR = './ImageTrainSet/201x201_TrainSetLeftRight'
VALIDATION_DIR = ''  # './ImageTrainSet/201x201_ValSetLeftRight_10' # IT WONT BE USED
OUTPUT_DIR = './Tmp'
TRAINING_METRICS_OUTPUT_DIR = './Results/Training_Metrics'
SAVED_MODEL_OUTPUT_DIR = './Results/Saved_Model_As_h5'
NUM_CLASSES = 6


# PARAMETERS
LR = 0.01
DECAY = 0
TRAIN_IMAGE_SIZE = (201, 201, 3)
BATCH_SIZE = 32
NUM_EPOCHS = 100


def get_classes():
    label_names = ["MAT", "MON", "PAP", "SOIL", "VER", "VIO"]
    return label_names


def coordinate_layer(image):
    mesh_x, mesh_y = np.meshgrid(np.arange(-100, 101), np.arange(-100, 101))
    coords = np.sqrt(np.square(mesh_x) + np.square(mesh_y)) / 1000.0
    return coords


def load_images_folder(path, max_images=99999999999):
    """  Load images from path, and discard 
    images that are not in correct shape
    """
    pattern = path + "/*.tif"
    files = sorted(glob.glob(pattern))[0:max_images]
    print("Loading images from path %s..." % (pattern))

    files_loaded = []
    if use_coordinates:
        images = np.zeros((len(files), 201, 201, 4))
    else:
        images = np.zeros((len(files), TRAIN_IMAGE_SIZE[0], TRAIN_IMAGE_SIZE[1], 3))

    discarded = 0
    loaded = 0
    for i in range(0, len(files)):
        if i % 1000 == 0:
            print(i)
        image = np.array(Image.open(files[i])) / 255.0
        if image.shape[0:2] == images[0, :, :, :].shape[0:2]:
            if use_coordinates:
                images[loaded, :, :, 0:3] = image
                images[loaded, :, :, 3] = coordinate_layer(image)
            else:
                images[loaded, :, :, :] = image
            loaded += 1
            files_loaded.append(files[i])
        else:
            # print("Warning: discarding image %s which is of size %s" % (files[i],image.shape))
            discarded += 1

    images = images[0:loaded, :, :, :]
    images = np.float16(images)

    print(
        "Loaded %i images from path %s [discarded %i images which are not of correct size]" % (loaded, path, discarded))
    return images, files_loaded


def train_model(model, lr, train_images, train_labels, val_images=None, val_labels=None):
    train_images, train_labels = augment(train_images, train_labels)
    print(train_images.shape)

    decay = DECAY
    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    lr = LR

    print("Using learning rate %f, decay %f, num_epochs %i" % (lr, decay, num_epochs))
    opt = keras.optimizers.Adam(lr=lr, decay=decay)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=OUTPUT_DIR, histogram_freq=0, write_graph=True, write_images=False)
    csv_logger = CSVLogger(TRAINING_METRICS_OUTPUT_DIR + "/" + MODEL_NAME + "_training_metrics.csv")

    num_classes = NUM_CLASSES
    model.fit(train_images,
              keras.utils.np_utils.to_categorical(train_labels),
              batch_size=batch_size,
              epochs=num_epochs,
              # validation_data=(val_images, keras.utils.np_utils.to_categorical(val_labels)), #it wont be used
              # callbacks=[plot,tensorboard, csv_logger])
              callbacks=[tensorboard, csv_logger])

    return model


def predict_image_set_with_augmentation(images, model):
    augmented_images, _ = augment(images, np.zeros(images.shape[0]))
    predictions = model.predict(augmented_images)
    return np.mean(predictions, axis=0, keepdims=True)


def augment(images, labels=None):
    """ Data augmentation for training, consisting of pi/2 rotation and up-down flip
    """
    global labels_augmented
    if images.shape[0] > 1:
        show_progress = True
    else:
        show_progress = False
    if show_progress:
        print("Augmenting %i images..." % images.shape[0])
    images_augmented = np.zeros((images.shape[0] * 8, images.shape[1], images.shape[2], images.shape[3]), "float16")
    if not labels is None:
        labels_augmented = np.zeros(images.shape[0] * 8)
    for i in range(0, images.shape[0]):
        if i % 1000 == 0 and show_progress:
            print(i)
        image = images[i, :, :, :]
        for j in range(0, 8):
            k = j%4
            new_image = np.rot90(image, k)
            if int(j/4)%2 == 1:
                new_image = np.flipud(new_image)
            images_augmented[i * 8 + j, :, :, :] = new_image
            if not labels is None:
                labels_augmented[i * 8 + j] = labels[i]

    if show_progress:
        print("Done.")
    if not labels is None:
        return images_augmented, labels_augmented
    else:
        return images_augmented


def shuffle(images, labels, files):
    np.random.seed(1)
    perm = np.random.permutation(images.shape[0])
    images = images[perm, :, :, :]
    labels = labels[perm]
    files = [files[i] for i in perm]
    return images, labels, files


def load_images_classes(path, max_num_images=99999999999):
    classes = get_classes()
    images = None
    files = []
    for i in range(0, len(classes)):
        images_class, files_class = load_images_folder(path + "/" + classes[i], max_num_images)
        labels_class = i * np.ones(images_class.shape[0])
        files.extend(files_class)
        if images is None:
            images = images_class
            labels = labels_class
        else:
            images = np.concatenate((images, images_class))
            labels = np.concatenate((labels, labels_class))
    images, labels, files = shuffle(images, labels, files)
    return images, labels, files


def prepare_data(train_path, test_path, max_num_images=99999999999):
    print("Preparing data...")
    train_images, train_labels, _ = load_images_classes(train_path, max_num_images)
    test_images, test_labels, _ = load_images_classes(test_path, max_num_images)
    return train_images, train_labels, test_images, test_labels


def build_and_save_model(save_model_as, train_image_size, num_classes=NUM_CLASSES, filtersize1=2, filtersize2=4, random_seed=1):
    train_images, train_labels, validation_images, validation_labels = prepare_data(TRAINSET_DIR, VALIDATION_DIR)
    model, lr = setup_model_resnet_manual_highres_center_only(train_image_size, num_classes, filtersize1, filtersize2)
    model._get_distribution_strategy = lambda: None
    model = train_model(model, lr, train_images, train_labels, validation_images, validation_labels)
    model.save(SAVED_MODEL_OUTPUT_DIR + "/random_seed_" + str(random_seed) + "_" + save_model_as + ".h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtersize1', type=int)
    parser.add_argument('--filtersize2', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    random_seed = args.seed
    np.random.seed(random_seed)

    MODEL_NAME = 'resnet_manual_highres_center_only_f1_{}_f2_{}'.format(args.filtersize1, args.filtersize2)

    print('Starting model with filtersize 1 ({}) and filtersize 2 ({}).'.format(args.filtersize1, args.filtersize2))
    start = time.time()
    build_and_save_model(MODEL_NAME, TRAIN_IMAGE_SIZE, NUM_CLASSES, args.filtersize1, args.filtersize2, random_seed)
    elapsed = (time.time() - start)
    print("Took %f minutes for training." % (elapsed / 60))

    np.save(TRAINING_METRICS_OUTPUT_DIR + "/" + MODEL_NAME + "_time_used.npy", (elapsed / 60))
