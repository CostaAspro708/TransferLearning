from statistics import mode
import sys 
import scipy
import numpy as np
from tensorflow import keras
from keras import layers
#from keras.optimizers import SGD
import os
import pathlib
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt



def task_1_3():
    num_classes = 5

    mobilenet_model = keras.applications.MobileNetV2( 
                                                    input_shape=(224,224,3),        
                                                    weights='imagenet'
                                                   )


    # Build the model
    global model
    model = keras.Sequential(
        [
            mobilenet_model,
            layers.Dense(num_classes, activation='softmax'),
        ]
    )

    model.summary()

def task_4():

    data_dir = pathlib.Path("small_flower_dataset")
    print("Number of images: ", len(list(data_dir.glob('*/*.jpg'))))
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name[0] != "."])
    print(CLASS_NAMES)

    image_generator = ImageDataGenerator(validation_split=0.2)
    global train_generator
    train_generator = image_generator.flow_from_directory(
        data_dir,
        target_size = (100,100),
        classes=list(CLASS_NAMES),
        shuffle = True,
        subset="training"
        )
    global test_generator
    test_generator = image_generator.flow_from_directory(
        data_dir,
        target_size = (100,100),
        classes=list(CLASS_NAMES),
        shuffle = True,
        subset="validation"
        )

def task_5_and_6():
    base_learning_rate = 0.01
    base_epochs = 20
    base_optomizer = keras.optimizers.SGD(
        learning_rate=base_learning_rate, momentum=0.00, nesterov=False, name="SGD"
    )

    model.compile(optimizer=base_optomizer,
              loss="categorical_crossentropy",
              metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=base_epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(base_epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.suptitle('Learning rate = 0.01, Momentum = 0')
    plt.show()

## For learning rate = 0.05, epochs = 20, momentum = 0.

def task_7_1():
    task_1_3()
    base_learning_rate = 0.05
    base_epochs = 20
    base_optomizer = keras.optimizers.SGD(
        learning_rate=base_learning_rate, momentum=0.00, nesterov=False, name="SGD"
    )

    model.compile(optimizer=base_optomizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=base_epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(base_epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.suptitle('Learning rate = 0.05, Momentum = 0')
    plt.show()

def task_7_2():
    task_1_3()
    base_learning_rate = 0.1
    base_epochs = 20
    base_optomizer = keras.optimizers.SGD(
        learning_rate=base_learning_rate, momentum=0.00, nesterov=False, name="SGD"
    )

    model.compile(optimizer=base_optomizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=base_epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(base_epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.suptitle('Learning rate = 0.1, Momentum = 0')
    plt.show()

def task_7_3():
    task_1_3()
    base_learning_rate = 0.005
    base_epochs = 20
    base_optomizer = keras.optimizers.SGD(
        learning_rate=base_learning_rate, momentum=0.00, nesterov=False, name="SGD"
    )

    model.compile(optimizer=base_optomizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=base_epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(base_epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.suptitle('Learning rate = 0.005, Momentum = 0')
    plt.show()

def task_8_1():
    task_1_3()
    base_learning_rate = 0.05
    base_epochs = 20
    base_optomizer = keras.optimizers.SGD(
        learning_rate=base_learning_rate, momentum=0.05, nesterov=False, name="SGD"
    )

    model.compile(optimizer=base_optomizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=base_epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(base_epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.suptitle('Learning rate = 0.05, Momentum = 0.05')
    plt.show()

def task_8_2():
    task_1_3()
    base_learning_rate = 0.05
    base_epochs = 20
    base_optomizer = keras.optimizers.SGD(
        learning_rate=base_learning_rate, momentum=0.10, nesterov=False, name="SGD"
    )

    model.compile(optimizer=base_optomizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=base_epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(base_epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.suptitle('Learning rate = 0.05, Momentum = 0.10 ')
    plt.show()

def task_8_3():
    task_1_3()
    base_learning_rate = 0.05
    base_epochs = 20
    base_optomizer = keras.optimizers.SGD(
        learning_rate=base_learning_rate, momentum=0.15, nesterov=False, name="SGD"
    )

    model.compile(optimizer=base_optomizer,
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=base_epochs)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(base_epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.suptitle('Learning rate = 0.05, Momentum = 0.15 ')
    plt.show()


if __name__ == "__main__":
    task_1_3()
    task_4()
    #task_5_and_6()
    #task_7_1()
    #task_7_2()
    #task_7_3()
    task_8_1()
    #task_8_2()
    #task_8_3()
    

