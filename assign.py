from statistics import mode
import sys 
import scipy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from keras import layers
from keras.optimizers import SGD
import os
import pathlib
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator


num_classes = 5

mobilenet_model = keras.applications.MobileNetV2( 
                                                input_shape=(224,224,3),        
                                                weights='imagenet'
                                               )


# Build the model

model = keras.Sequential(
    [
        mobilenet_model,
        layers.Dense(num_classes, activation='softmax'),
    ]
)

model.summary()


data_dir = pathlib.Path("small_flower_dataset")
print("Number of images: ", len(list(data_dir.glob('*/*.jpg'))))
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name[0] != "."])
print(CLASS_NAMES)

image_generator = ImageDataGenerator(validation_split=0.2)

train_generator = image_generator.flow_from_directory(
    data_dir,
    target_size = (100,100),
    classes=list(CLASS_NAMES),
    shuffle = True,
    subset="training"
    )

test_generator = image_generator.flow_from_directory(
    data_dir,
    target_size = (100,100),
    classes=list(CLASS_NAMES),
    shuffle = True,
    subset="validation"
    )

base_learning_rate = 0.000000001
base_optomizer = keras.optimizers.SGD(
    learning_rate=base_learning_rate, momentum=0.0, nesterov=False, name="SGD"
)

model.compile(optimizer=base_optomizer,
              loss="categorical_crossentropy",
              metrics=['accuracy'])


history = model.fit(train_generator, validation_data=test_generator, epochs=20)

## Plotting train and test loss and accuracy

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Model loss and val_loss at 0.000000001 learning rate')
plt.ylabel('loss/val_loss')
plt.xlabel('epoch')
plt.legend()
plt.show()


plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Model accuracy and val_accuracy at 0.000000001 learning rate')
plt.ylabel('accuracy/val_accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()


