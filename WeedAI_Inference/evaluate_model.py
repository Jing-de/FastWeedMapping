import numpy as np
import sys
import train_model as train_model
import scipy.sparse
import keras
import shutil
import os
import time
    eee


def predict_image_set_with_augmentation(images, model):
    augmented_images,_ = train_model.augment(images,np.zeros(images.shape[0]))
    predictions = model.predict(augmented_images)
    return np.exp(np.mean(np.log(predictions),axis=0,keepdims=True))

        
def save_error(label,prediction,file):
    classes = train_model.get_classes()
    path = "./errors/%s_classified_as_%s" % (classes[label],classes[prediction])
    if not os.path.isdir(path):
        os.makedirs(path)
    shutil.copyfile(file,path+"/"+os.path.basename(file))
    
    
def print_statistics(confusion_matrix):
    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    print("Accuracy = %.4f (on %i test instances)" % (accuracy,np.sum(confusion_matrix)))

    classes = train_model.get_classes()
    for i in range(0,len(classes)):
        recall_class = confusion_matrix[i,i]/np.sum(confusion_matrix[i,:])
        precision_class = confusion_matrix[i,i]/np.sum(confusion_matrix[:,i])
        print("Class %s precision/recall: %0.4f/%0.4f" % (classes[i],precision_class,recall_class))

    print("Confusion matrix (rows = true class, columns = predicted class):\n")
    for i in range(0,6):
        for j in range(0,6):
            s = "%i" % int(confusion_matrix[i,j])
            while len(s) < 6:
                s = " "+s
            sys.stdout.write(s)
        sys.stdout.write("\n")
    
    
def evaluate_model_fast(model_name,path):
    model = keras.models.load_model(model_name)
    images,labels,files = train_model.load_images_classes(path)
    
    predictions = []
    errors = []
    num_classes = len(train_model.get_classes())
    confusion_matrix = np.zeros((num_classes,num_classes))

    print("Evaluating model (fast version) on %i instances..." % images.shape[0])
    start = time.time()    
    prediction_probs = model.predict(images)
    print("Elased time: %f seconds" % (time.time()-start))

    for i in range(0,images.shape[0]):
        predicted_class = np.argmax(prediction_probs[i,:])
        predictions.append(predicted_class)
        label = int(labels[i])
        confusion_matrix[label,predicted_class] += 1
        if label == predicted_class:
            errors.append(0)
        else:
            save_error(label,predicted_class,files[i])
            errors.append(1)

    print_statistics(confusion_matrix)
    
    
def evaluate_model(model_name, path):
    model = keras.models.load_model(model_name)

    images,labels,files = train_model.load_images_classes(path)

    predictions = []
    errors = []
    num_classes = len(train_model.get_classes())
    confusion_matrix = np.zeros((num_classes,num_classes))
    print("Evaluating model...")
    if os.path.isdir("./errors"):
        shutil.rmtree("./errors")    
    
    for i in range(0,images.shape[0]):
        if i % 1000 == 0:
            print(i)
        prediction_probs = predict_image_set_with_augmentation(images[i:i+1,:,:,:],model)[0]
        predicted_class = np.argmax(prediction_probs)
        predictions.append(predicted_class)
        label = int(labels[i])
        confusion_matrix[label,predicted_class] += 1
        if label == predicted_class:
            errors.append(0)
        else:
            #save_error(label,predicted_class,files[i])
            errors.append(1)
    print_statistics(confusion_matrix)
            

if __name__ == "__main__":
    np.random.seed(1)
    model_name = sys.argv[1]
    evaluate_model_fast(model_name,"./data/201x201_TestSetMid_selected")
    #evaluate_model(model_name,"./data/201x201_TestSetMid_selected")
    


