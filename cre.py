import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import shutil
from mtcnn import MTCNN

import tensorflow as tf
from keras import backend as K


# 1. Đường dẫn video và thư mục lưu trữ
cap = cv2.VideoCapture('./data_test/t4.mp4')
output_folder = 'frvideo'
cropFace_folder = 'cropface'

detector = MTCNN()   

def creFolder(opf):
     if not os.path.exists(opf):
        os.makedirs(opf)
        print(f"Đã tạo thư mục: {opf}")

def cutFrame(cap, output_folder, cropFace_folder):

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_original = cap.get(cv2.CAP_PROP_FPS) 
    # số khung hình muốn save
    ws = 2
    save_interval = int(fps_original / ws) # 30 / 10 = 3

    # 2. Tạo thư mục nếu nó chưa tồn tại
    creFolder(output_folder)
    creFolder(cropFace_folder)


    print("w: ",w,",h: ",h,",fps: ",fps_original)

    cnt = 0
    sv_cnt = 0

    # w_new = 150
    # h_new = int(h*w_new/w)

    w_new = w
    h_new = h

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if cnt % save_interval == 0:
            frame1 = cv2.resize(frame, (w_new, h_new))
            file_name_opf = os.path.join(output_folder, f"frame_{sv_cnt:03d}.jpg")
            cv2.imwrite(file_name_opf, frame1)
            file_name_crf = os.path.join(cropFace_folder, f"crop_{sv_cnt:03d}.jpg")
            cv2.imwrite(file_name_crf, cropFace(file_name_opf))
            sv_cnt+=1
            print("saved ",sv_cnt)

        cnt+=1

    cap.release()
    cv2.destroyAllWindows()

    K.clear_session()
    tf.compat.v1.reset_default_graph()


def removeData(output_folder):
    if os.path.exists(output_folder):
        # Xóa toàn bộ thư mục và các file con bên trong
        shutil.rmtree(output_folder)
        print(f"Đã xóa vĩnh viễn thư mục: {output_folder}")
    else:
        print("Thư mục không tồn tại.")

def cropFace(image_path, margin_ratio=0.1):
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh ",os.path.splitext(image_path)[0])
        return None
        
    # OpenCV mặc định đọc ảnh hệ màu BGR, cần chuyển sang RGB cho MTCNN hiểu
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Dò tìm khuôn mặt
    # Kết quả trả về là một danh sách chứa tọa độ các khuôn mặt tìm thấy
    faces = detector.detect_faces(img_rgb)

    if faces:
        # Lấy khuôn mặt đầu tiên (to nhất) trong ảnh
        bounding_box = faces[0]['box']
        x, y, w, h = bounding_box  # Tọa độ x, y góc trên bên trái, cùng chiều rộng (w) và cao (h)

        # 4. Tính toán phần viền mở rộng (BƯỚC QUAN TRỌNG NHẤT)
        x_margin = int(w * margin_ratio)
        y_margin = int(h * margin_ratio)

        # 5. Tính tọa độ cắt mới (dùng hàm max, min để không bị cắt lẹm ra ngoài bức ảnh)
        start_x = max(0, x - x_margin)
        start_y = max(0, y - y_margin)
        end_x = min(img.shape[1], x + w + x_margin)
        end_y = min(img.shape[0], y + h + y_margin)

        # 6. Cắt ảnh theo tọa độ mới
        cropped_face = img[start_y:end_y, start_x:end_x]

        # 7. Đưa về kích thước chuẩn mực cho Mạng CNN (Ví dụ: Mạng Xception / ResNet hay dùng 224x224)
        resized_face = cv2.resize(cropped_face, (224, 224))

        return resized_face

    else:
        # os.path.basename(path) hoặc os.path.splitext(file_name)[0]
        print("Không tìm thấy khuôn mặt nào trong ảnh ",os.path.splitext(image_path)[0])
       
        return None
    

# cutFrame(cap, output_folder, cropFace_folder)
removeData(cropFace_folder)
removeData(output_folder)

