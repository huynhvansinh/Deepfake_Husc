import os
import json
import csv
import time
from typing import List, Dict, Any

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# =========================================================
# CẤU HÌNH - ANH CHỈ CẦN SỬA Ở ĐÂY
# =========================================================

# Chọn 1 trong 2 chế độ:
# MODE = "video"      -> đánh giá 1 video
# MODE = "folder"     -> đánh giá toàn bộ video trong folder
MODE = "folder"

# Nếu MODE = "video", điền đường dẫn 1 video ở đây
VIDEO_PATH = "../data_test/test_video.mp4"

# Nếu MODE = "folder", điền đường dẫn folder chứa video ở đây
VIDEO_DIR = "../data_test"

# Đường dẫn best_model.pth
MODEL_PATH = "./training_outputs/final_eff_b4_ffpp02_celebdf02_final_both_sam_adamw_dropout_15epoch/best_model.pth"

# Chọn kiến trúc model đã train:
# "densenet121"
# "efficientnet_b0"
# "efficientnet_b4"
# "resnet18"
MODEL_NAME = "efficientnet_b4"

# Kích thước ảnh đầu vào
# DenseNet121 / EfficientNet-B0 / ResNet18 thường là 224
# EfficientNet-B4 nếu train 380 thì để 380
IMG_SIZE = 224

# Lấy bao nhiêu frame mỗi giây
SAMPLE_FPS = 2.0

# Số frame tối đa dùng để đánh giá mỗi video
MAX_FRAMES = 80

# Ngưỡng kết luận FAKE
# Nếu avg_prob_fake >= THRESHOLD thì dự đoán FAKE
THRESHOLD = 0.5

# Có crop mặt không
# Nếu model của anh train trên ảnh mặt đã crop thì nên để True
USE_FACE_CROP = True

# Nếu folder có thư mục con thì để True
RECURSIVE = False

# File lưu kết quả
SAVE_CSV = "../test/video_predictions.csv"
SAVE_JSON = "../data_test/video_predictions.json"


# =========================================================
# CÁC ĐUÔI VIDEO HỖ TRỢ
# =========================================================
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"
}


# =========================================================
# TẠO MODEL
# =========================================================
def create_model(model_name: str, num_classes: int = 2):
    model_name = model_name.lower()

    if model_name == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Không hỗ trợ MODEL_NAME='{model_name}'. "
            f"Hỗ trợ: densenet121, efficientnet_b0, efficientnet_b4, resnet18"
        )

    return model


# =========================================================
# LOAD CHECKPOINT
# =========================================================
def load_model(model_path: str, model_name: str, device: torch.device):
    model = create_model(model_name=model_name, num_classes=2)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint

    elif isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        new_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k.replace("module.", "", 1)
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=True)

    else:
        raise ValueError("Checkpoint không đúng định dạng.")

    model.to(device)
    model.eval()

    return model


# =========================================================
# FACE CROP BẰNG OPENCV
# =========================================================
class FaceCropper:
    def __init__(self, margin: float = 0.25):
        self.margin = margin

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError("Không load được Haar Cascade của OpenCV.")

    def crop_largest_face(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        h, w = frame_bgr.shape[:2]

        if len(faces) == 0:
            return self.center_crop(frame_bgr)

        x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])

        margin_x = int(fw * self.margin)
        margin_y = int(fh * self.margin)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)

        face = frame_bgr[y1:y2, x1:x2]

        if face.size == 0:
            return self.center_crop(frame_bgr)

        return face

    @staticmethod
    def center_crop(frame_bgr):
        h, w = frame_bgr.shape[:2]
        size = min(h, w)

        x1 = (w - size) // 2
        y1 = (h - size) // 2
        x2 = x1 + size
        y2 = y1 + size

        return frame_bgr[y1:y2, x1:x2]


# =========================================================
# TRANSFORM GIỐNG LÚC TRAIN
# =========================================================
def get_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =========================================================
# DỰ ĐOÁN 1 FRAME
# =========================================================
@torch.no_grad()
def predict_frame(model, frame_bgr, transform, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)

    if logits.shape[1] == 2:
        prob = F.softmax(logits, dim=1)

        # Quy ước:
        # label 0 = REAL
        # label 1 = FAKE
        prob_real = float(prob[0, 0].item())
        prob_fake = float(prob[0, 1].item())
    else:
        prob_fake = float(torch.sigmoid(logits)[0].item())
        prob_real = 1.0 - prob_fake

    return prob_real, prob_fake


# =========================================================
# DỰ ĐOÁN 1 VIDEO
# =========================================================
def predict_video(
    video_path: str,
    model,
    device,
    img_size: int,
    sample_fps: float,
    max_frames: int,
    threshold: float,
    use_face_crop: bool
) -> Dict[str, Any]:

    video_start_time = time.perf_counter()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps is None or video_fps <= 0:
        video_fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(1, int(round(video_fps / sample_fps)))

    transform = get_transform(img_size)

    cropper = FaceCropper() if use_face_crop else None

    frame_index = 0
    used_frames = 0

    prob_real_list: List[float] = []
    prob_fake_list: List[float] = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % step == 0:
            if use_face_crop:
                frame_for_model = cropper.crop_largest_face(frame)
            else:
                frame_for_model = frame

            prob_real, prob_fake = predict_frame(
                model=model,
                frame_bgr=frame_for_model,
                transform=transform,
                device=device
            )

            prob_real_list.append(prob_real)
            prob_fake_list.append(prob_fake)

            used_frames += 1

            if used_frames >= max_frames:
                break

        frame_index += 1

    cap.release()

    if used_frames == 0:
        raise RuntimeError("Không lấy được frame nào từ video.")

    avg_prob_real = sum(prob_real_list) / len(prob_real_list)
    avg_prob_fake = sum(prob_fake_list) / len(prob_fake_list)

    prediction = "FAKE" if avg_prob_fake >= threshold else "REAL"

    inference_time_seconds = time.perf_counter() - video_start_time
    inference_time_minutes = inference_time_seconds / 60.0
    avg_time_per_frame_seconds = inference_time_seconds / max(used_frames, 1)

    result = {
        "video_name": os.path.basename(video_path),
        "video_path": video_path,
        "prediction": prediction,
        "avg_prob_real": round(avg_prob_real, 6),
        "avg_prob_fake": round(avg_prob_fake, 6),
        "threshold": threshold,
        "used_frames": used_frames,
        "total_video_frames": total_frames,
        "video_fps": round(video_fps, 3),
        "sample_fps": sample_fps,
        "img_size": img_size,
        "use_face_crop": use_face_crop,

        # Thời gian nhận diện
        "inference_time_seconds": round(inference_time_seconds, 6),
        "inference_time_minutes": round(inference_time_minutes, 6),
        "avg_time_per_frame_seconds": round(avg_time_per_frame_seconds, 6),

        "status": "OK",
        "error": ""
    }

    return result


# =========================================================
# TÌM VIDEO TRONG FOLDER
# =========================================================
def find_videos_in_folder(video_dir: str, recursive: bool = False) -> List[str]:
    video_paths = []

    if recursive:
        for root, _, files in os.walk(video_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in VIDEO_EXTENSIONS:
                    video_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(video_dir):
            full_path = os.path.join(video_dir, file)

            if os.path.isfile(full_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in VIDEO_EXTENSIONS:
                    video_paths.append(full_path)

    video_paths.sort()
    return video_paths


# =========================================================
# LƯU CSV
# =========================================================
def save_results_to_csv(results: List[Dict[str, Any]], save_csv: str):
    os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)

    fieldnames = [
        "video_name",
        "video_path",
        "prediction",
        "avg_prob_real",
        "avg_prob_fake",
        "threshold",
        "used_frames",
        "total_video_frames",
        "video_fps",
        "sample_fps",
        "img_size",
        "use_face_crop",

        # Thời gian nhận diện
        "inference_time_seconds",
        "inference_time_minutes",
        "avg_time_per_frame_seconds",

        "status",
        "error"
    ]

    with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)


# =========================================================
# MAIN
# =========================================================
def main():
    total_start_time = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("======================================")
    print("DEEPFAKE VIDEO PREDICTION")
    print("======================================")
    print(f"MODE: {MODE}")
    print(f"Thiết bị: {device}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"IMG_SIZE: {IMG_SIZE}")
    print(f"SAMPLE_FPS: {SAMPLE_FPS}")
    print(f"MAX_FRAMES: {MAX_FRAMES}")
    print(f"THRESHOLD: {THRESHOLD}")
    print(f"USE_FACE_CROP: {USE_FACE_CROP}")
    print("======================================")

    print("\nĐang load model...")

    load_start_time = time.perf_counter()

    model = load_model(
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        device=device
    )

    load_time_seconds = time.perf_counter() - load_start_time

    print(f"Load model xong. Thời gian load model: {load_time_seconds:.4f} giây")

    if MODE == "video":
        video_paths = [VIDEO_PATH]

    elif MODE == "folder":
        if not os.path.isdir(VIDEO_DIR):
            raise NotADirectoryError(f"Không tìm thấy folder: {VIDEO_DIR}")

        video_paths = find_videos_in_folder(
            video_dir=VIDEO_DIR,
            recursive=RECURSIVE
        )

    else:
        raise ValueError("MODE chỉ được là 'video' hoặc 'folder'.")

    if len(video_paths) == 0:
        print("Không tìm thấy video nào.")
        return

    print(f"\nTìm thấy {len(video_paths)} video cần đánh giá.\n")

    results: List[Dict[str, Any]] = []

    for idx, video_path in enumerate(video_paths, start=1):
        print(f"[{idx}/{len(video_paths)}] Đang xử lý: {video_path}")

        try:
            result = predict_video(
                video_path=video_path,
                model=model,
                device=device,
                img_size=IMG_SIZE,
                sample_fps=SAMPLE_FPS,
                max_frames=MAX_FRAMES,
                threshold=THRESHOLD,
                use_face_crop=USE_FACE_CROP
            )

            print(
                f"    => {result['prediction']} | "
                f"REAL={result['avg_prob_real']} | "
                f"FAKE={result['avg_prob_fake']} | "
                f"frames={result['used_frames']} | "
                f"time={result['inference_time_seconds']}s"
            )

        except Exception as e:
            result = {
                "video_name": os.path.basename(video_path),
                "video_path": video_path,
                "prediction": "ERROR",
                "avg_prob_real": "",
                "avg_prob_fake": "",
                "threshold": THRESHOLD,
                "used_frames": 0,
                "total_video_frames": "",
                "video_fps": "",
                "sample_fps": SAMPLE_FPS,
                "img_size": IMG_SIZE,
                "use_face_crop": USE_FACE_CROP,

                # Thời gian nhận diện khi lỗi
                "inference_time_seconds": "",
                "inference_time_minutes": "",
                "avg_time_per_frame_seconds": "",

                "status": "ERROR",
                "error": str(e)
            }

            print(f"    => ERROR: {e}")

        results.append(result)

    save_results_to_csv(results, SAVE_CSV)
    print(f"\nĐã lưu kết quả CSV vào: {SAVE_CSV}")

    if SAVE_JSON:
        os.makedirs(os.path.dirname(SAVE_JSON) or ".", exist_ok=True)

        with open(SAVE_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Đã lưu kết quả JSON vào: {SAVE_JSON}")

    ok_results = [r for r in results if r["status"] == "OK"]
    error_results = [r for r in results if r["status"] == "ERROR"]

    real_count = sum(1 for r in ok_results if r["prediction"] == "REAL")
    fake_count = sum(1 for r in ok_results if r["prediction"] == "FAKE")

    total_inference_time = sum(
        float(r["inference_time_seconds"])
        for r in ok_results
        if r["inference_time_seconds"] != ""
    )

    avg_inference_time = total_inference_time / len(ok_results) if len(ok_results) > 0 else 0.0

    total_program_time = time.perf_counter() - total_start_time

    print("\n===== TỔNG KẾT =====")
    print(f"Tổng video: {len(results)}")
    print(f"Xử lý thành công: {len(ok_results)}")
    print(f"Lỗi: {len(error_results)}")
    print(f"Dự đoán REAL: {real_count}")
    print(f"Dự đoán FAKE: {fake_count}")
    print(f"Tổng thời gian nhận diện video OK: {total_inference_time:.4f} giây")
    print(f"Thời gian nhận diện trung bình / video OK: {avg_inference_time:.4f} giây")
    print(f"Thời gian load model: {load_time_seconds:.4f} giây")
    print(f"Tổng thời gian chạy chương trình: {total_program_time:.4f} giây")
    print("====================")


if __name__ == "__main__":
    main()