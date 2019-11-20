import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def preprocess_labels(array):
    """
        array : np array
    
    This function takes as parameter a numpy array. 
    If the elements of this array are integers it trasforms them into a onehot encoded numpy arrays. 
    """
    # Since the elements of the array are of the same nature. 
    # i.e. all of them are either onehot encoded arrays, arrays of 0s and 1s, or integer values.
    # Here we are checking the dimension of the first element of the array.
    if np.ndim(array[0]) == 0:
        # If it is an integer we create a onehot encoded vector for each number.
        # The length of the onehot encoded vector is n_values which is (the biggest number in the array) + 1.
        n_values = np.max(array) + 1
        new = np.eye(n_values)[array]
        return new
    """
    new/array : are np arrays.
    """
    return array

def load_mnist():
    """
    This function loads the MNIST dataset. 
    And preprocesses the labels because they are integers they will be transformed to onehot encoded vectors.
    """
    # Loading the data.
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = keras.datasets.mnist.load_data()

    # Preprocessing the labels.
    y_train_mnist = preprocess_labels(y_train_mnist)
    y_train_mnist = y_train_mnist.astype('float32')
    y_test_mnist = preprocess_labels(y_test_mnist)
    y_test_mnist = y_test_mnist.astype('float32')

    """
    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist : are np arrays.
    """

    return x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist

def load_cifar():
    """
    This function loads the CIFAR dataset. 
    And preprocesses the labels because they are vectors of one integer for each instance.
    They will be transformed to onehot encoded vectors.
    """
    # Loading the data.
    (x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = keras.datasets.cifar100.load_data(label_mode='fine')
        
    # Creating arrays of the same length as the labels vectors.
    test_label_cifar = np.empty([y_test_cifar.shape[0]], dtype=int)
    train_label_cifar = np.empty([y_train_cifar.shape[0]], dtype=int)

    # Puting the labels which are integers into the new vectors.
    for i in range(len(y_train_cifar)):
        # Since le labels are in vectors and they are floats they have to be extracted then casted to integers.
        train_label_cifar[i] = int(y_train_cifar[i][0])

    # Doing the same thing for the test labels.
    for i in range(len(y_test_cifar)):
        test_label_cifar[i] = int(y_train_cifar[i][0])

    # Preprocessing the labels vectors.
    train_label_cifar = preprocess_labels(train_label_cifar)
    train_label_cifar = train_label_cifar.astype('float32')
    test_label_cifar = preprocess_labels(test_label_cifar)
    test_label_cifar = test_label_cifar.astype('float32')

    """
    x_train_cifar, train_label_cifar, x_test_cifar, test_label_cifar : are np arrays.
    """
    
    return x_train_cifar, train_label_cifar, x_test_cifar, test_label_cifar

def load_PA100K():
    """
    This function loads the PA100K dataset.
    The path choosen for the images is ./data/release_data/release_data/
    """
    # Loading the file that contains the name of the images.
    mat = scipy.io.loadmat('annotation.mat')

    # Getting the train validation and test data names and labels.
    test_images_name = mat['test_images_name']
    test_label = mat['test_label']
    train_images_name = mat['train_images_name']
    train_label = mat['train_label']
    val_images_name = mat['val_images_name']
    val_label = mat['val_label']

    # Retrieving the train images and putting them into a list and not an array because they do not have the same shape.
    train_images = []
    for i in range(len(train_images_name)):
        train_images.append(plt.imread('./data/release_data/release_data/'+train_images_name[i][0][0]))

    # Doing the same thing with the test and validation images. 
    # I am considering the validation images with the test images because on the website of the competition 
    # I found that the the test images contain almost 20000 images which is equivallent to the length of the test and validation datasets.
    test_images = []
    for i in range(len(test_images_name)):
        test_images.append(plt.imread('./data/release_data/release_data/'+test_images_name[i][0][0]))

    for i in range(len(val_images_name)):
        test_images.append(plt.imread('./data/release_data/release_data/'+val_images_name[i][0][0]))

    # Concatenating the test and validation labels.
    test_label = test_label + val_label

    # Returning everything.
    """
    train_images,test_images : are lists.
    train_label,test_label : are np arrays.
    """
    return train_images,train_label,test_images,test_label


def load_PA100K_10():
    """
    This function loads 1000 images for the train and the test data from the PA100K dataset.
    """
    # Loading the file that contains the name of the images.
    mat = scipy.io.loadmat('annotation.mat')

    # Getting the train validation and test data names and labels.
    test_images_name = mat['test_images_name']
    test_label = mat['test_label']
    train_images_name = mat['train_images_name']
    train_label = mat['train_label']
    
    train_label = train_label[:1000]
    test_label = test_label[:1000]

    train_images = []
    for i in range(len(train_images_name)):
        train_images.append(plt.imread('/Users/aymene/Desktop/comp-internship/data/release_data/release_data/' + train_images_name[i][0][0]))
        if (i+1)>=1000:
            break

    test_images = []
    for i in range(len(test_images_name)):
        test_images.append(plt.imread('/Users/aymene/Desktop/comp-internship/data/release_data/release_data/' + test_images_name[i][0][0]))
        if (i+1)>=1000:
            break
    
    """
    train_images,train_label,test_images,test_label : are np arrays.
    """
    return train_images,train_label,test_images,test_label