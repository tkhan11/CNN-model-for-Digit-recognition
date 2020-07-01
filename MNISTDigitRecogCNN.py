import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import time

seconds= time.time()
time_start = time.ctime(seconds)   #  The time.ctime() function takes seconds passed since epoc
print("start time:", time_start,"\n")    # as an argument and returns a string representing time.



# Data preparation

# Load data
dataset = pd.read_csv(".\\MNIST_Dataset.csv")

random_seed = 2

# Split the Dataset into Test and Training datasets 80-20 split
Train_Dataset, Test_dataset = train_test_split(dataset, test_size = 0.2, random_state=random_seed)

Y_train = Train_Dataset['label']
X_train = Train_Dataset.drop(['label'],axis=1)
Y_test = Test_dataset['label']
X_test = Test_dataset.drop(['label'],axis=1)


print("X_train ,X_test ,Y_train ,Y_test :\n",X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# Check the data for null values if any

#X_train.isnull().values.any()
#X_test.isnull().values.any()

# Reshape image in 3 dimensions (height = 28px, width = 28px , depth = 1)
X_train = X_train.values.reshape((-1,28,28,1))
X_test = X_test.values.reshape(-1,28,28,1)

print("After 28*28*1 :  ",X_train.shape,X_test.shape)

# Moreover the CNN converg faster on [0..1] data than on [0..255].

# Normalize the data
X_train = X_train/255.0
X_test = X_test/255.0


#Label encoding
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
Y_Test = to_categorical(Y_test, num_classes = 10)


# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

print("\n**************************\n")
print("After train and validation set split for model fitting :\n X_train, Y_train, X_validation, Y_validation :",X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)

class_labels = ['0', '1', '2' , '3', '4', '5', '6', '7', '8', '9']

#Some examples
#plt.imshow(X_train[6][:,:,0])
#plt.show()


# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

Digit_Recog_CNN_model = Sequential()

Digit_Recog_CNN_model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
Digit_Recog_CNN_model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
Digit_Recog_CNN_model.add(MaxPool2D(pool_size=(2,2)))
Digit_Recog_CNN_model.add(Dropout(0.25))


Digit_Recog_CNN_model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
Digit_Recog_CNN_model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
Digit_Recog_CNN_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
Digit_Recog_CNN_model.add(Dropout(0.25))


Digit_Recog_CNN_model.add(Flatten())
Digit_Recog_CNN_model.add(Dense(256, activation = "relu"))
Digit_Recog_CNN_model.add(Dropout(0.5))
Digit_Recog_CNN_model.add(Dense(10, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
Digit_Recog_CNN_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
epochs = 10 # attained .992 accuracy on 10 epochs
batch_size = 128

# Model Training

history = Digit_Recog_CNN_model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_val, Y_val))

print("\n","****************MODEL EVALUATION ************************\n")

# Model Evaluation on Test data

test_loss,test_acc=Digit_Recog_CNN_model.evaluate(X_test,Y_Test)

print("Evaluated model accuracy on test data :",test_acc)

seconds= time.time()
time_stop = time.ctime(seconds)
print("\n","stop time:", time_stop,"\n")



# Predict the values from the Test dataset
Y_pred = Digit_Recog_CNN_model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_test, Y_pred_classes) 


#Printing Classification Report
Y_Test_as_arrary = Y_test.array                  #Y_test is a series need to be converted into 1-D array for being passed into Classification_report func. 

print(classification_report(Y_Test_as_arrary, Y_pred_classes, target_names = class_labels))

accuracy = accuracy_score(Y_test, Y_pred_classes)
print('Accuracy: %f' % accuracy)

# Training and validation curves

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

plt.show()



# Defining function for plotting confusion matrix  

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 





     
