import cv2
import os
import time
from k import ANC_PATH
# Đường dẫn tới thư mục lưu trữ hình ảnh
output_dir = ANC_PATH


# Số lượng ảnh cần chụp
num_images = 300

# Thời gian chờ giữa các lần chụp ảnh (đơn vị: giây)
wait_time = 0.5

# Khởi tạo đối tượng VideoCapture để đọc dữ liệu từ webcam
cap = cv2.VideoCapture(0)

# Kiểm tra xem webcam có khả dụng hay không
if not cap.isOpened():
    raise Exception("Không thể mở webcam.")

# Chụp và lưu ảnh
for i in range(num_images):
    # Đọc khung hình từ webcam
    ret, frame = cap.read()

    if not ret:
        continue

    # Hiển thị khung hình
    cv2.imshow('Webcam', frame)

    # Ghi ảnh vào thư mục lưu trữ
    image_path = os.path.join(output_dir, f'image_{i}.jpg')
    cv2.imwrite(image_path, frame)

    # Chờ 0.5 giây trước khi chụp ảnh tiếp theo
    time.sleep(wait_time)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()