import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json

NAME = "landmarks_CNN"

base_model=MobileNet(weights='imagenet',include_top=False) 
preModel=base_model.output
preModel=GlobalAveragePooling2D()(preModel)
preModel=Dense(1024,activation='relu')(preModel) #dense layer 1
preModel=Dense(1024,activation='relu')(preModel) #dense layer 2
preModel=Dense(512,activation='relu')(preModel)
preModel=Dense(256,activation='relu')(preModel) 
predictions=Dense(10,activation='softmax')(preModel) 

model=Model(inputs=base_model.input,outputs=predictions)

for each in model.layers[:15]:
    each.trainable=False
for each in model.layers[15:]:
    each.trainable=True

training=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

train_generator=training.flow_from_directory('Data/train/', 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=3)

model_json = model.to_json()
with open("vgg16model_3.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("vgg16model_3.h5")