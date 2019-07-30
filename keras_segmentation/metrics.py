import numpy as np
import matplotlib 
matplotlib.use('Agg') # pylint: disable=multiple-statements
import matplotlib.pyplot as plt # plotting

EPS = 1e-12

def get_iou( gt , pr , n_classes ):
    class_wise = np.zeros(n_classes)
    pr = pr.reshape(-1)
    for cl in range(n_classes):
        intersection = np.sum(( gt == cl )*( pr == cl ))
        union = np.sum(np.maximum( ( gt == cl ) , ( pr == cl ) ))
        iou = float(intersection)/( union + EPS )
        #print metrics for each image
        print("Intersection, Union, IOU for class {}: {}, {}, {}".format(cl, intersection, union, iou))
        class_wise[ cl ] = iou
    return class_wise
    
def plot_accuracy(history = None, path = "", validate = False):
    #print total accuracy graph
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['acc'])
    if(validate):
        plt.plot(history.history['val_acc'])
        plt.legend(['train', 'validation'], loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #plt.ylim([0,1])
    plt.savefig(path+'accuracy.png')
    
def plot_loss(history = None, path = "", validate=False):
    #print total loss graph
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'])
    if(validate):
        plt.plot(history.history['val_loss'])
        plt.legend(['train', 'validation'], loc='upper left')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.ylim([0,10])
    plt.savefig(path+'loss.png')
