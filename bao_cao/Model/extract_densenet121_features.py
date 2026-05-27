import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN THEO CẤU TRÚC THƯ MỤC CỦA ANH
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# File này nằm trong: D:/Nghiencuu/Model/extract_densenet121_features.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # D:/Nghiencuu/Model
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                  # D:/Nghiencuu

# Model nằm trong: D:/Nghiencuu/Model/training_outputs/densenet121_ffpp/best_model.pth
MODEL_PATH = os.path.join(
    CURRENT_DIR,
    "training_outputs",
    "densenet121_ffpp",
    "best_model.pth"
)

# CSV nằm trong: D:/Nghiencuu/processed_ffpp/splits/test.csv
CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "processed_ffpp",
    "splits",
    "test.csv"
)

# Lưu feature vào: D:/Nghiencuu/Model/features_densenet121
OUTPUT_DIR = os.path.join(
    CURRENT_DIR,
    "features_densenet121"
)

OUTPUT_NPY = "test_features.npy"
OUTPUT_CSV = "test_features.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CLASSES = 2

# ============================================================
# 2. DATASET ĐỌC ẢNH TỪ CSV
# ============================================================

class FaceDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        print("Các cột trong CSV:", list(self.df.columns))

        possible_path_cols = [
            "image_path",
            "path",
            "filepath",
            "file_path",
            "crop_path",
            "face_path"
        ]

        self.path_col = None

        for col in possible_path_cols:
            if col in self.df.columns:
                self.path_col = col
                break

        if self.path_col is None:
            raise ValueError(
                "Không tìm thấy cột đường dẫn ảnh. "
                f"CSV hiện có các cột: {list(self.df.columns)}"
            )

        possible_label_cols = [
            "label",
            "target",
            "class",
            "y"
        ]

        self.label_col = None

        for col in possible_label_cols:
            if col in self.df.columns:
                self.label_col = col
                break

        print(f"Cột đường dẫn ảnh được dùng: {self.path_col}")

        if self.label_col is not None:
            print(f"Cột label được dùng: {self.label_col}")
        else:
            print("Không tìm thấy cột label, label sẽ được gán là -1")

    def __len__(self):
        return len(self.df)

    def resolve_image_path(self, raw_path):
        image_path = str(raw_path).strip().replace("\\", "/")

        if image_path.startswith("./"):
            image_path = image_path[2:]

        candidate_paths = []

        if os.path.isabs(image_path):
            candidate_paths.append(image_path)
        else:
            # Trường hợp CSV chứa: processed_ffpp/...
            candidate_paths.append(os.path.join(CURRENT_DIR, image_path))

            # Trường hợp CSV chứa: ./processed_ffpp/...
            candidate_paths.append(os.path.join(CURRENT_DIR, image_path))

            # Trường hợp CSV chứa path tính từ D:/Nghiencuu
            candidate_paths.append(os.path.join(PROJECT_ROOT, image_path))

            # Trường hợp CSV chỉ chứa path con bên trong processed_ffpp
            candidate_paths.append(os.path.join(CURRENT_DIR, "processed_ffpp", image_path))

        for path in candidate_paths:
            path = os.path.abspath(path)
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "Không tìm thấy ảnh.\n"
            f"Path trong CSV: {raw_path}\n"
            "Đã thử các đường dẫn:\n" +
            "\n".join(os.path.abspath(p) for p in candidate_paths)
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.resolve_image_path(row[self.path_col])

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.label_col is not None:
            label = int(row[self.label_col])
        else:
            label = -1

        return image, label, image_path


# ============================================================
# 3. TẠO MODEL DENSENET121 GIỐNG LÚC TRAIN
# ============================================================

def build_model():
    model = models.densenet121(weights=None)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)

    return model


# ============================================================
# 4. LOAD MODEL ĐÃ TRAIN
# ============================================================

def load_trained_model(model_path):
    model = build_model()

    checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint:
        model.load_state_dict(checkpoint)

    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        raise ValueError("Không nhận dạng được định dạng checkpoint.")

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# 5. HÀM TRÍCH XUẤT FEATURE 1024 CHIỀU
# ============================================================

def extract_features(model, images):
    with torch.no_grad():
        feature_maps = model.features(images)
        feature_maps = F.relu(feature_maps, inplace=True)

        pooled = F.adaptive_avg_pool2d(feature_maps, (1, 1))
        features = torch.flatten(pooled, 1)

        logits = model.classifier(features)
        probs = F.softmax(logits, dim=1)

    return features, logits, probs


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("===== TRÍCH XUẤT ĐẶC TRƯNG DENSENET121 =====")
    print(f"Thiết bị đang dùng: {DEVICE}")
    print(f"CURRENT_DIR:  {CURRENT_DIR}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Model path:   {MODEL_PATH}")
    print(f"CSV path:     {CSV_PATH}")
    print(f"Output dir:   {OUTPUT_DIR}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Không tìm thấy CSV: {CSV_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = FaceDataset(CSV_PATH, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"Số lượng ảnh cần xử lý: {len(dataset)}")

    model = load_trained_model(MODEL_PATH)

    all_features = []
    all_logits = []
    all_probs = []
    all_labels = []
    all_paths = []

    for images, labels, paths in tqdm(dataloader, desc="Đang trích xuất feature"):
        images = images.to(DEVICE)

        features, logits, probs = extract_features(model, images)

        all_features.append(features.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        all_labels.extend(labels.numpy().tolist())
        all_paths.extend(list(paths))

    all_features = np.concatenate(all_features, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    print("\n===== KẾT QUẢ =====")
    print("Feature shape:", all_features.shape)
    print("Logits shape: ", all_logits.shape)
    print("Probs shape:  ", all_probs.shape)

    npy_path = os.path.abspath(os.path.join(OUTPUT_DIR, OUTPUT_NPY))
    np.save(npy_path, all_features)

    feature_columns = [f"feature_{i}" for i in range(all_features.shape[1])]

    df_out = pd.DataFrame(all_features, columns=feature_columns)

    df_out.insert(0, "image_path", all_paths)
    df_out.insert(1, "label", all_labels)

    df_out["logit_real"] = all_logits[:, 0]
    df_out["logit_fake"] = all_logits[:, 1]
    df_out["prob_real"] = all_probs[:, 0]
    df_out["prob_fake"] = all_probs[:, 1]

    csv_path = os.path.abspath(os.path.join(OUTPUT_DIR, OUTPUT_CSV))
    df_out.to_csv(csv_path, index=False)

    print("\n===== ĐÃ LƯU FILE =====")
    print(f"File NPY: {npy_path}")
    print(f"File CSV: {csv_path}")
    print(f"Thư mục lưu: {os.path.abspath(OUTPUT_DIR)}")

    print("\nVí dụ 1 vector feature đầu tiên:")
    print(all_features[0])


if __name__ == "__main__":
    main()