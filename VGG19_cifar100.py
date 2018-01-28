import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import Adam
import cv2, numpy as np
from keras.datasets import cifar100
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import tensorflow as tf
from keras.utils import multi_gpu_model

tbCallBack = keras.callbacks.TensorBoard(log_dir='./tensorflow/notrans-BN', histogram_freq=0, write_graph=True, write_images=False)

# Parameters
batch_size = 32
num_classes = 100
epochs = 200
data_augmentation = True
weight_decay = 0.0001

#Download and Load the dataset
(X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

#Conversion of the labels to categorical data
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

model = Sequential()
#Implement with heavy dropout
#Kernel penalty of the type - l2
# Block - 1

model.add(Conv2D(64, (3,3), input_shape = (32,32,3), padding = 'same', name = '1_1', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), padding = 'same', name = '1_2', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), name = 'pool_1'))

# Block - 2

model.add(Conv2D(128, (3,3), padding = 'same', name = '2_1', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), padding = 'same', name = '2_2', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), name = 'pool_2'))

# Block - 3

model.add(Conv2D(256, (3,3), padding = 'same', name = '3_1', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(256, (3,3), padding = 'same', name = '3_2', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(256, (3,3), padding = 'same', name = '3_3', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), name = 'pool_3'))

# Block - 4

model.add(Conv2D(512, (3,3), padding = 'same', name = '4_1', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3,3), padding = 'same', name = '4_2', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3,3), padding = 'same', name = '4_3', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3,3), padding = 'same', name = '4_4', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), name = 'pool_4'))

# Block - 5

model.add(Conv2D(512, (3,3), padding = 'same', name = '5_1', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3,3), padding = 'same', name = '5_2', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3,3), padding = 'same', name = '5_3', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(512, (3,3), padding = 'same', name = '5_4', kernel_regularizer=l2(weight_decay)))
model.add(BatchNormalization()) 
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), name = 'pool_5'))

model.add(Flatten())

# Full Connection

model.add(Dense(4096, name = 'fc_1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, name = 'fc_2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('softmax'))

X_train = X_train/255.
X_test  =X_test/255.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#Adam Optimizer
opt = Adam(lr = 0.001, decay = 0.0001)

#Image Preprocessing
datagen = ImageDataGenerator(width_shift_range=0.1,  height_shift_range=0.1, horizontal_flip=True)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    
history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size), steps_per_epoch=X_train.shape[0] // batch_size+1,
                              epochs=epochs,callbacks=[tbCallBack],
                              validation_data=(X_test, Y_test))
#Visualize the train and test accuracies and loss
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()