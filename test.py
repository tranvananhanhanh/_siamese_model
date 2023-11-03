import cv2
import tensorflow as tf
from siamese import L1Dist
import os
import numpy as np
from predata import preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from train import siamese_model
from predata import preprocess





# Tải mô hình và trọng số đã được huấn luyện
model = tf.keras.models.load_model('siamesemodel.h5')

# Khởi tạo danh sách kết quả

inpa="/Users/jmac/Desktop/siamese/application_data/input_image/A001.jpg"

inp=preprocess(inpa)

results=[]
dt=0.5

for image in os.listdir(os.path.join('application_data','verification_image')):

    if image != '.DS_Store':
        val=preprocess(os.path.join('application_data','verification_image',image))
        result = model.predict(list(np.expand_dims([inp, val], axis=1)))
        
        results.append(result)
        detection = np.sum(np.array(results) > dt)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_image'))) 
        verified = verification > 0.5


 
        

print(verified,results)
    