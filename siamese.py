import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow import keras

from predata import train_data


# tạo lớp nhúng trả về lớp cuối cùng trích xuất đặc trưng ảnh
#input tensor kích thước (none,100,100,3) none số lượng hình ảnh 1 batch,3 RBG
#output tensor (none,4096) vecto 1 chiều kích thước 4096
def make_embedding():
    #đầu vào: shape là kích thước,tên đối tượng đầu vào
    inp = Input(shape=(100, 100, 3),name='input_image')
    # first block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)
    # second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)
    # third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)
    # final block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    #làm phẳng đầu ra
    f1 = Flatten()(c4)
    #dense lớp kết nối đầy đủ, kích hoạt hàm sigmoid(tìm ước lượng cực đại và xác xuất) tính toán gia trị d1
    d1 = Dense(4096, activation='sigmoid')(f1)

    model = Model(inputs=[inp], outputs=[d1])
    return model


#build distance layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)#gọi phương thức khởi tạo lớp cơ sở mà lopws ht kế thừa-> đảm bảo thuộc tính cs tạo đúng cách


    def call(self, input_embedding,validation_embedding):
        
        #trả về sự khác nhau 2 tensor
        return tf.math.abs(input_embedding - validation_embedding)
    
#make siamese model



def make_siamese_model():
    # an anchor img
    input_image = Input(name='input_img', shape=(100, 100, 3))
    # validation img in network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))
    # create embedding models
    input_embedding=make_embedding()(input_image)
    validation_embedding=make_embedding()(validation_image)
    # combine siamese distance components
    siamese_layer = L1Dist()
    distances = siamese_layer(input_embedding, validation_embedding)
    # classification layer lớp đầu ra 
    # hàm sigmod giá trj 0->1 giá trị phụ thuộc biến distance
    classifier = Dense(1, activation='sigmoid')(distances)
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model=make_siamese_model()
siamese_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
