import tensorflow as tf
from k import anchor,positive,negative

#prepro-scale-resize
def preprocess(file_path):
    #đọc ảnh từ đường dẫn
    byte_img = tf.io.read_file(file_path)

    #giải mã chuỗi byte thành 1 tensor bd ảnh
    img = tf.io.decode_jpeg(byte_img)

    #đổi kích cỡ 100*100 3 kênh màu
    img = tf.image.resize(img, (100, 100))

    #chuẩn hóa pixel
    img = img / 255.0

    return img



#creat labelled dataset
positives = tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negative = tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
# ống dữ liệu (a,p,1) và (a,n,0)
data = positives.concatenate(negative)


#build train and test partition

def preprocess_twin(input_img, validation_img, label):
  #trả lại các ảnh và nhãn tương ứng
  return(preprocess(input_img),preprocess(validation_img),label)


#build dataloader pipeline

#lặp qua từng phần tử trong data chạy hàm yêu cầu
data = data.map(preprocess_twin)

#lưu trữ dữ liệu đã xử lý trong bộ nhớ cache
data = data.cache()

#xáo trộn các phần tử trong data tạo thành sự ngẫu nhiên 
data = data.shuffle(buffer_size=1024)

#train partition
#lấy 0.7 của data
train_data = data.take(round(len(data)*0.7))
#chia tập thành 16 batch nhỏ 
train_data = train_data.batch(16)
#lấy 8 batch xử lý trước
train_data = train_data.prefetch(8)

#phần test lấy 0.3 còn lại data
test_data=data.skip(round(len(data)*0.7))
test_data=test_data.take(round(len(data)*0.3))
test_data=test_data.batch(16)
test_data=test_data.prefetch(8)