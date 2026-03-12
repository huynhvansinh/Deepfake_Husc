import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
from mtcnn import MTCNN

# =========================
# CẤU HÌNH
# =========================
DATASET_ROOT = "./datasets/FFPP"      # sửa đúng theo thư mục thật
FRAMES_ROOT = "./output_frames"
CROPFACES_ROOT = "./output_cropfaces"
SAVE_PER_SECOND = 2
FACE_SIZE = (224, 224)
MARGIN_RATIO = 0.1

detector = MTCNN()


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def is_video_file(filename):
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    return filename.lower().endswith(video_exts)


def get_all_video_paths(dataset_root):
    video_paths = []
    for root, dirs, files in os.walk(dataset_root):
        if os.path.basename(root).lower() != "videos":
            continue
        for file in files:
            if is_video_file(file):
                video_paths.append(os.path.join(root, file))
    return sorted(video_paths)


def crop_face_from_frame(frame, margin_ratio=0.1, output_size=(224, 224)):
    if frame is None:
        return None

    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi detect face: {e}")
        return None

    if not faces:
        return None

    largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
    x, y, w, h = largest_face['box']

    if w <= 0 or h <= 0:
        return None

    x_margin = int(w * margin_ratio)
    y_margin = int(h * margin_ratio)

    start_x = max(0, x - x_margin)
    start_y = max(0, y - y_margin)
    end_x = min(frame.shape[1], x + w + x_margin)
    end_y = min(frame.shape[0], y + h + y_margin)

    if start_x >= end_x or start_y >= end_y:
        return None

    cropped_face = frame[start_y:end_y, start_x:end_x]
    if cropped_face.size == 0:
        return None

    return cv2.resize(cropped_face, output_size)


def process_video(video_path, dataset_root, frames_root, cropfaces_root,
                  save_per_second=2, face_size=(224, 224), margin_ratio=0.1):
    rel_path = os.path.relpath(video_path, dataset_root)
    rel_no_ext = os.path.splitext(rel_path)[0]

    frame_output_dir = os.path.join(frames_root, rel_no_ext)
    crop_output_dir = os.path.join(cropfaces_root, rel_no_ext)

    create_folder(frame_output_dir)
    create_folder(crop_output_dir)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[LỖI] Không mở được video: {video_path}")
        return

    fps_original = cap.get(cv2.CAP_PROP_FPS)
    if fps_original <= 0:
        fps_original = 25

    save_interval = max(1, int(fps_original / save_per_second))

    total_frames = 0
    saved_frames = 0
    cropped_faces = 0
    no_face_frames = 0

    print(f"\nĐang xử lý: {video_path}")
    print(f"FPS = {fps_original:.2f}, save_interval = {save_interval}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if total_frames % save_interval == 0:
            frame_name = f"frame_{saved_frames:04d}.jpg"
            crop_name = f"crop_{saved_frames:04d}.jpg"

            frame_path = os.path.join(frame_output_dir, frame_name)
            crop_path = os.path.join(crop_output_dir, crop_name)

            cv2.imwrite(frame_path, frame)

            cropped_face = crop_face_from_frame(
                frame,
                margin_ratio=margin_ratio,
                output_size=face_size
            )

            if cropped_face is not None:
                cv2.imwrite(crop_path, cropped_face)
                cropped_faces += 1
            else:
                no_face_frames += 1

            saved_frames += 1

        total_frames += 1

    cap.release()
    cv2.destroyAllWindows()

    print("  - Tổng frame đọc:", total_frames)
    print("  - Frame đã lưu:", saved_frames)
    print("  - Crop mặt thành công:", cropped_faces)
    print("  - Không tìm thấy mặt:", no_face_frames)


def main():
    if not os.path.exists(DATASET_ROOT):
        print(f"[LỖI] Không tìm thấy thư mục dataset: {DATASET_ROOT}")
        return

    create_folder(FRAMES_ROOT)
    create_folder(CROPFACES_ROOT)

    video_paths = get_all_video_paths(DATASET_ROOT)

    if not video_paths:
        print("[LỖI] Không tìm thấy file video nào trong dataset.")
        return

    video_paths = video_paths[:5]  # test trước 5 video
    print(f"Tìm thấy {len(video_paths)} video để test.\n")

    for idx, video_path in enumerate(video_paths, 1):
        print(f"===== Video {idx}/{len(video_paths)} =====")
        process_video(
            video_path=video_path,
            dataset_root=DATASET_ROOT,
            frames_root=FRAMES_ROOT,
            cropfaces_root=CROPFACES_ROOT,
            save_per_second=SAVE_PER_SECOND,
            face_size=FACE_SIZE,
            margin_ratio=MARGIN_RATIO
        )

    print("\nHoàn tất xử lý.")


if __name__ == "__main__":
    main()