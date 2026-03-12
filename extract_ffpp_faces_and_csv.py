import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import csv
import cv2
import random
from mtcnn import MTCNN


# =========================================================
# CẤU HÌNH
# =========================================================
DATASET_ROOT = "./datasets/FFPP"          # sửa nếu cần
OUTPUT_ROOT = "./processed_ffpp"          # thư mục kết quả chung
CROPFACES_ROOT = os.path.join(OUTPUT_ROOT, "cropfaces")
CSV_PATH = os.path.join(OUTPUT_ROOT, "ffpp_faces.csv")

SAVE_PER_SECOND = 2                       # mỗi giây lấy bao nhiêu frame
FACE_SIZE = (224, 224)                   # kích thước ảnh mặt đầu ra
MARGIN_RATIO = 0.15                      # nới biên quanh mặt
MAX_VIDEOS_PER_VIDEOS_FOLDER = None        # mỗi thư mục videos lấy tối đa bao nhiêu video
RANDOM_SEED = 42

# Nếu chỉ muốn test ít video, đổi True
DEBUG_MODE = False
DEBUG_MAX_TOTAL_VIDEOS = 5


# =========================================================
# KHỞI TẠO
# =========================================================
random.seed(RANDOM_SEED)
detector = MTCNN()


# =========================================================
# HÀM PHỤ
# =========================================================
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def is_video_file(filename):
    return filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))


def get_label_and_method_from_relpath(rel_path):
    """
    rel_path ví dụ:
    manipulated_sequences/Deepfakes/c23/videos/033_097.mp4
    original_sequences/youtube/c23/videos/001.mp4
    """
    norm = rel_path.replace("\\", "/")
    parts = norm.split("/")

    label = None
    method = "unknown"

    if "original_sequences" in parts:
        label = 0
        # method lấy thư mục ngay sau original_sequences, ví dụ youtube / actors
        idx = parts.index("original_sequences")
        if idx + 1 < len(parts):
            method = parts[idx + 1]

    elif "manipulated_sequences" in parts:
        label = 1
        # method lấy thư mục ngay sau manipulated_sequences, ví dụ Deepfakes / Face2Face / FaceShifter
        idx = parts.index("manipulated_sequences")
        if idx + 1 < len(parts):
            method = parts[idx + 1]

    return label, method


def get_all_video_paths_limited(dataset_root, max_per_videos_folder=50):
    """
    Chỉ lấy file video trong các thư mục tên 'videos'.
    Với mỗi thư mục 'videos', lấy tối đa max_per_videos_folder video.
    """
    all_video_paths = []

    for root, dirs, files in os.walk(dataset_root):
        if os.path.basename(root).lower() != "videos":
            continue

        video_files = [f for f in files if is_video_file(f)]
        video_files.sort()

        if max_per_videos_folder is not None:
            video_files = video_files[:max_per_videos_folder]

        for file in video_files:
            full_path = os.path.join(root, file)
            all_video_paths.append(full_path)

    all_video_paths.sort()
    return all_video_paths


def crop_face_from_frame(frame, margin_ratio=0.15, output_size=(224, 224)):
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

    # chọn mặt lớn nhất
    largest_face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
    x, y, w, h = largest_face["box"]

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

    try:
        resized_face = cv2.resize(cropped_face, output_size)
    except Exception:
        return None

    return resized_face


def process_video(video_path, dataset_root, cropfaces_root,
                  save_per_second=2, face_size=(224, 224), margin_ratio=0.15):
    """
    Trả về danh sách record để ghi CSV.
    """
    rel_path = os.path.relpath(video_path, dataset_root)
    rel_no_ext = os.path.splitext(rel_path)[0]
    rel_no_ext_unix = rel_no_ext.replace("\\", "/")

    label, method = get_label_and_method_from_relpath(rel_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if label is None:
        print(f"[BỎ QUA] Không xác định được label cho video: {video_path}")
        return []

    crop_output_dir = os.path.join(cropfaces_root, rel_no_ext)
    create_folder(crop_output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được video: {video_path}")
        return []

    fps_original = cap.get(cv2.CAP_PROP_FPS)
    if fps_original <= 0:
        fps_original = 25

    save_interval = max(1, int(fps_original / save_per_second))

    total_frames = 0
    saved_frames = 0
    cropped_faces = 0
    no_face_frames = 0

    csv_records = []

    print(f"\nĐang xử lý: {video_path}")
    print(f"FPS = {fps_original:.2f}, save_interval = {save_interval}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if total_frames % save_interval == 0:
            crop_name = f"crop_{saved_frames:04d}.jpg"
            crop_path = os.path.join(crop_output_dir, crop_name)

            cropped_face = crop_face_from_frame(
                frame=frame,
                margin_ratio=margin_ratio,
                output_size=face_size
            )

            if cropped_face is not None:
                cv2.imwrite(crop_path, cropped_face)
                cropped_faces += 1

                csv_records.append({
                    "image_path": crop_path.replace("\\", "/"),
                    "label": label,
                    "method": method,
                    "video_name": video_name,
                    "source_video_path": video_path.replace("\\", "/"),
                    "relative_video_path": rel_path.replace("\\", "/"),
                    "split_group": rel_no_ext_unix
                })
            else:
                no_face_frames += 1

            saved_frames += 1

        total_frames += 1

    cap.release()
    cv2.destroyAllWindows()

    print("  - Tổng frame đọc:", total_frames)
    print("  - Frame đã duyệt để lưu:", saved_frames)
    print("  - Crop mặt thành công:", cropped_faces)
    print("  - Không tìm thấy mặt:", no_face_frames)

    return csv_records


def write_csv(csv_path, rows):
    create_folder(os.path.dirname(csv_path))

    fieldnames = [
        "image_path",
        "label",
        "method",
        "video_name",
        "source_video_path",
        "relative_video_path",
        "split_group",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows):
    total = len(rows)
    real_count = sum(1 for r in rows if int(r["label"]) == 0)
    fake_count = sum(1 for r in rows if int(r["label"]) == 1)

    method_stats = {}
    video_stats = {}

    for r in rows:
        method = r["method"]
        split_group = r["split_group"]

        method_stats[method] = method_stats.get(method, 0) + 1
        video_stats[split_group] = video_stats.get(split_group, 0) + 1

    print("\n================ TỔNG KẾT ================")
    print("Tổng số ảnh crop:", total)
    print("Số ảnh REAL:", real_count)
    print("Số ảnh FAKE:", fake_count)

    print("\nSố ảnh theo method:")
    for method, count in sorted(method_stats.items()):
        print(f"  - {method}: {count}")

    print("\nSố video đã sinh ảnh:", len(video_stats))
    print(f"CSV đã lưu tại: {CSV_PATH}")
    print("==========================================\n")


def main():
    if not os.path.exists(DATASET_ROOT):
        print(f"[LỖI] Không tìm thấy thư mục dataset: {DATASET_ROOT}")
        return

    create_folder(OUTPUT_ROOT)
    create_folder(CROPFACES_ROOT)

    video_paths = get_all_video_paths_limited(
        dataset_root=DATASET_ROOT,
        max_per_videos_folder=MAX_VIDEOS_PER_VIDEOS_FOLDER
    )

    if not video_paths:
        print("[LỖI] Không tìm thấy video nào.")
        return

    if DEBUG_MODE:
        video_paths = video_paths[:DEBUG_MAX_TOTAL_VIDEOS]

    print(f"Tìm thấy {len(video_paths)} video để xử lý.")

    all_rows = []

    for idx, video_path in enumerate(video_paths, 1):
        print(f"===== Video {idx}/{len(video_paths)} =====")
        rows = process_video(
            video_path=video_path,
            dataset_root=DATASET_ROOT,
            cropfaces_root=CROPFACES_ROOT,
            save_per_second=SAVE_PER_SECOND,
            face_size=FACE_SIZE,
            margin_ratio=MARGIN_RATIO
        )
        all_rows.extend(rows)

    write_csv(CSV_PATH, all_rows)
    summarize_rows(all_rows)
    print("Hoàn tất xử lý toàn bộ.")


if __name__ == "__main__":
    main()