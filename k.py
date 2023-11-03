import cv2
import os
import random
import numpy as np
import uuid
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus :
  tf.config.experimental.set_memory_growth(gpu, True)




POS_PATH = os.path.join('data','positive')
NEG_PATH = os.path.join('data','negative')
ANC_PATH = os.path.join('data','anchor')



anchor = tf.data.Dataset.list_files(os.path.join(ANC_PATH,'*.jpg')).take(300)
positive = tf.data.Dataset.list_files(os.path.join(POS_PATH,'*.jpg')).take(300)
negative = tf.data.Dataset.list_files(os.path.join(NEG_PATH,'*.jpg')).take(300)


for directory in os.listdir('lfw') :
  if directory != '.DS_Store':
    for file in os.listdir(os.path.join('lfw',directory)) :
      EX_PATH = os.path.join('lfw',directory,file)
      NEW_PATH = os.path.join(NEG_PATH, file)
      os.replace(EX_PATH,NEW_PATH)








