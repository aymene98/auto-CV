import numpy as np
import tensorflow as tf
from tensorflow import keras
import operator
import cv2
from sklearn.metrics import roc_auc_score


class Model(object):
  # Constructor of the model
    def __init__(self):
        # Create attributs of the model.
        self.conv_part = keras.Sequential()
        self.dense_part = keras.Sequential()
        self.model = keras.Sequential()
        self.shape = (0, 0, 0)
        self.done_training = False

    def create_model(self, shape, num_classes):
        """
            shape : the shape of the input images which is a tuple.
            num_classes : the number of classes in the classification problem which is an int.

        This function creates the architecture of the model.
        The architecture of the model is more on the dinamic side of things. the number of convolutions depends on the size of the image.
        If the image is big we can do more convolutions with a big kernal.
        For ample detail on the architecture of the network please refer to the report.
        """

        first = True # This variable is just for checking I should add the input_shape parameter to the conv layer.

        # Constructing the conv part od the CNN
        if (shape[0] > 400) or (shape[1]>400):
            self.conv_part.add(keras.layers.Conv2D(
                    filters=15, kernel_size=7, strides=1, padding='valid', input_shape=shape))
            self.conv_part.add(keras.layers.PReLU())
            self.conv_part.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
            first = False
        else:
            if (shape[0] > 100) or (shape[1]>100):
                self.conv_part.add(keras.layers.Conv2D(
                    filters=15, kernel_size=5, strides=1, padding='valid', input_shape=shape))
                self.conv_part.add(keras.layers.PReLU())
                self.conv_part.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
                first = False

        if first:
            self.conv_part.add(keras.layers.Conv2D(
            filters=30, kernel_size=3, strides=1, padding='same', input_shape=shape))
        else:
            self.conv_part.add(keras.layers.Conv2D(
            filters=30, kernel_size=3, strides=1, padding='same'))

        self.conv_part.add(keras.layers.PReLU())
        self.conv_part.add(keras.layers.BatchNormalization())

        self.conv_part.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.conv_part.add(keras.layers.Conv2D(
            filters=60, kernel_size=3, strides=1, padding='valid'))
        self.conv_part.add(keras.layers.PReLU())
        self.conv_part.add(keras.layers.BatchNormalization())

        if shape[2]==3:
            self.conv_part.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            self.conv_part.add(keras.layers.Conv2D(
                filters=90, kernel_size=3, strides=1, padding='valid'))
            self.conv_part.add(keras.layers.PReLU())
            self.conv_part.add(keras.layers.BatchNormalization())

        self.conv_part.add(keras.layers.Flatten())

        # Constructing the dense part of the CNN.
        if (shape[0] > 100) or (shape[1] > 100):
            self.dense_part.add(keras.layers.Dense(200))
            self.dense_part.add(keras.layers.PReLU())
            self.dense_part.add(keras.layers.BatchNormalization())
            self.dense_part.add(keras.layers.Dropout(0.5))

        self.dense_part.add(keras.layers.Dense(100))
        self.dense_part.add(keras.layers.PReLU())
        self.dense_part.add(keras.layers.BatchNormalization())
        self.dense_part.add(keras.layers.Dropout(0.3))

        self.dense_part.add(keras.layers.Dense(50))
        self.dense_part.add(keras.layers.PReLU())
        self.dense_part.add(keras.layers.BatchNormalization())

        self.dense_part.add(keras.layers.Dense(num_classes, activation='sigmoid'))

        # Adding the conv part and the dense part to the final model.
        self.model.add(self.conv_part)
        self.model.add(self.dense_part)

    def preprocess_train_data(self, dataset):
        """
            dataset : a np array/ list it contains the training images.

        This function preprocesses the training images by resizing them to the average shape. 
        And if they are in a list it puts them in a np array.
        """
        
        # Getting the shape from the first image because i don't know the shape or if the images are rgb or greyscale.
        sum_of_shapes = dataset[0].shape
        s = sum_of_shapes   # For cheking the the 3rd dimension.
        cpt = 0             # For counting the number of images.

        # Summing the shapes of the images.
        for i in dataset:
            sum_of_shapes = tuple(map(operator.add, sum_of_shapes, i.shape))
            cpt += 1

        # Substacting the shape of the first image.
        sum_of_shapes = tuple(
            map(operator.sub, sum_of_shapes, dataset[0].shape))

        # Calculating the average of the shapes of the images.
        self.shape = tuple(ti//cpt for ti in sum_of_shapes)

        # Resizing the images and counting the number of resizes.
        nb_resizes = 0
        for i in range(len(dataset)):
            if (dataset[i].shape == self.shape):
                pass
            else:
                dataset[i] = cv2.resize(dataset[i], (self.shape[1], self.shape[0]), interpolation = cv2.INTER_CUBIC)
                nb_resizes += 1

        
        try:
            # Checking if the image is RGB (i.e. image has 3 dimensions).
            # Checking the number of resizes. If the number of resizes is greater then 0,
            # then the images must have been in a list and they should be copied in a np array.
            print(s[2])
            if nb_resizes > 0:
                # Creating np array.
                x_train = np.empty(
                    [len(dataset), self.shape[0], self.shape[1], s[2]])
                
                # Copying the images from the list to the np array.
                for i in range(len(dataset)):
                    x_train[i] = dataset[i]
                # Updating the shape.
                self.shape = dataset[0].shape
                return x_train

            # Updating the shape of the training images.
            self.shape = dataset[0].shape
            return dataset

        except Exception as e:
            print("**************** Training images are not RGB ****************")

            # Creating a new np array to host the images.
            x_train = np.empty([len(dataset), self.shape[0], self.shape[1]])
                
            # Copying the images from the list to the np array.
            for i in range(len(dataset)):
                x_train[i] = dataset[i]

            # Reshaping the image so that it has the appropriate format.
            x_train = x_train.reshape(x_train.shape[0], self.shape[0], self.shape[1], 1)

            # Updating the shape of the training images.
            self.shape = x_train[0].shape
            return x_train

        return dataset

    def preprocess_test_data(self, test_images):
        """
            test_images : a np array/ list it contains the test images.

        This function preprocesses the test images by resizing them to the average shape (already calculated using the training images).
        And if they are in a list it puts them in a np array.
        """

        # Resizing the test images to the average shape of the training dataset.
        nb_resizes = 0
        for i in range(len(test_images)):
            if test_images[i].shape is not self.shape:
                test_images[i] = cv2.resize(test_images[i], (self.shape[1], self.shape[0]), interpolation = cv2.INTER_CUBIC)
                nb_resizes += 1

        try:
            # Checking for 3rd dimension
            s = test_images[0].shape
            print(s[2])                 # This is to stimulate an error in case of a greyscale image.
            if nb_resizes > 0:
                # Creating np array to host the images. Because if i resized some of the images they must have been in a list.
                x_test = np.empty(
                    [len(test_images), test_images[0].shape[0], test_images[0].shape[1], test_images[0].shape[2]])
                
                # Copying the images from the list to the np array.
                for i in range(len(test_images)):
                    x_test[i] = test_images[i]
            
            return x_test

        except Exception as e:
            print("**************** Test images are not RGB ****************")

            x_test = np.empty([len(test_images), self.shape[0], self.shape[1]])

            for i in range(len(test_images)):
                x_test[i] = test_images[i]

            x_test = x_test.reshape(x_test.shape[0], self.shape[0], self.shape[1], 1)
            return x_test

        return test_images

    def train(self, dataset, labels, remaining_time_budget=None):
        """
            dataset: list/ np array of the training data.
            labels: np array of the training labels.
        This function preprocesses the traning data, creates the model and trains it.
        """
        print("**************** Preprocessing the training data ****************")
        # Preprocessing the train data.
        x = self.preprocess_train_data(dataset)
        
        # Deleting the dataset variable to save some RAM.
        del dataset
        
        # Casting the np array to float to avoid having 0s when deviding by 255.
        x = x.astype('float32')
        x = x / 255
        print("**************** Preprocessing ended ****************")
        
        print("**************** Creating the model ****************")
        self.create_model(self.shape, np.size(labels, 1))

        print("**************** The architecture of the model ****************")
        # Printing the architecture of the model for this specific dataset.
        print(self.conv_part.summary())
        print(self.dense_part.summary())
        print(self.model.summary())

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        print("**************** The shape of the images is : ", self.shape, " ****************")

        self.model.fit(x, labels, epochs=100, batch_size = 100, verbose = 1, validation_split=0.2, shuffle=True,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, min_delta=0.05)])

        print("**************** Training done ****************")

        self.done_training = True

    def test(self, dataset, labels, remaining_time_budget=None):
        """
            dataset: list/ np array of the test data.
            labels: np array of the test labels.
        """
        # Preprocessing the test dataset.
        x = self.preprocess_test_data(dataset)

        # Deleting the dataset variable to save some RAM.
        del dataset

        # Casting the np array to float to avoid having 0s when deviding by 255.
        x = x.astype('float32')
        x = x / 255

        # Evaluating the model on the test data.
        loss, accuracy = self.model.evaluate(x, labels)
        print("The loss is : ", loss)
        print("The accuarcy is : ", accuracy)

        # Calculating the predictions.
        predictions = self.model.predict(x)

        # Calculating the ROC-AUC.
        print("The ROC-AUC is : {:.4f}".format(roc_auc_score(labels,predictions)))
        return predictions
