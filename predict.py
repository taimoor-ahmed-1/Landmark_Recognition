import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,model_from_json
import numpy as np
json_file = open('MobNetModel_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


loaded_model.load_weights("MobNetModel_3.h5")
print("Model Loaded")
 
loaded_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
print("Model Compiled")

testing=ImageDataGenerator(preprocessing_function=preprocess_input)
pred_generator=testing.flow_from_directory('Data/predict/', 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=5,
                                                 class_mode='categorical',
                                                 shuffle=False)
step_size=pred_generator.n//pred_generator.batch_size

labelsStr = ["TombJehangir","badshahi","centaurus","faisal","minarPak","MizarQ","NoorM","PakMonument","QuaidHouse","SupremeCourt"]
predictions= loaded_model.predict_generator(generator=pred_generator,steps=step_size,verbose=1)

labels = np.argmax(predictions, axis=-1)
fileNames=pred_generator.filenames
count = 0
for each in labels:
    print("file: " + fileNames[count] + " label: " + labelsStr[each])
    count+=1

