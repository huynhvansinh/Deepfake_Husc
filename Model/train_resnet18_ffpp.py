import os
import copy
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================================================
# CẤU HÌNH
# =========================================================
TRAIN_CSV = "../processed_ffpp/splits/train.csv"
VAL_CSV = "../processed_ffpp/splits/val.csv"
TEST_CSV = "../processed_ffpp/splits/test.csv"

OUTPUT_DIR = "./training_outputs/resnet18_ffpp"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
RANDOM_SEED = 42

# Freeze backbone trước hay fine-tune toàn bộ
FREEZE_BACKBONE = False

# Nếu dữ liệu lệch lớp, bật class weights
USE_CLASS_WEIGHTS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# HỖ TRỢ
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


# =========================================================
# DATASET
# =========================================================
class FFPPDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.csv_path = csv_path

        required_cols = ["image_path", "label"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV thiếu cột bắt buộc: {col}")

        # Thư mục gốc của project: lùi 1 cấp từ thư mục chứa splits
        # Ví dụ:
        # csv_path = ../processed_ffpp/splits/train.csv
        # => project_root = ../
        csv_abs_path = os.path.abspath(csv_path)
        splits_dir = os.path.dirname(csv_abs_path)                 # .../processed_ffpp/splits
        processed_ffpp_dir = os.path.dirname(splits_dir)          # .../processed_ffpp
        project_root = os.path.dirname(processed_ffpp_dir)        # .../Nghiencuu

        def resolve_image_path(p):
            p = str(p).replace("\\", os.sep).replace("/", os.sep)

            # nếu đã là đường dẫn tuyệt đối thì giữ nguyên
            if os.path.isabs(p):
                return p

            # bỏ tiền tố ./ nếu có
            if p.startswith("." + os.sep):
                p = p[2:]

            # ghép với project_root
            return os.path.normpath(os.path.join(project_root, p))

        self.df["image_path"] = self.df["image_path"].apply(resolve_image_path)

        # bỏ các dòng file không tồn tại để tránh crash
        exists_mask = self.df["image_path"].apply(os.path.exists)
        self.df = self.df[exists_mask].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"Không còn ảnh hợp lệ trong {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# =========================================================
# TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# =========================================================
# METRICS
# =========================================================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


# =========================================================
# TRAIN / EVAL
# =========================================================
def run_one_epoch(model, dataloader, criterion, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return epoch_loss, metrics


# =========================================================
# MODEL
# =========================================================
def build_model(num_classes=2, freeze_backbone=False):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    create_folder(OUTPUT_DIR)

    print("===== THÔNG TIN THỰC NGHIỆM =====")
    print(f"Thiết bị: {DEVICE}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Val CSV:   {VAL_CSV}")
    print(f"Test CSV:  {TEST_CSV}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"FREEZE_BACKBONE: {FREEZE_BACKBONE}")
    print(f"USE_CLASS_WEIGHTS: {USE_CLASS_WEIGHTS}")
    print("=================================\n")

    train_dataset = FFPPDataset(TRAIN_CSV, transform=train_transform)
    val_dataset = FFPPDataset(VAL_CSV, transform=eval_transform)
    test_dataset = FFPPDataset(TEST_CSV, transform=eval_transform)

    print(f"Số ảnh train: {len(train_dataset)}")
    print(f"Số ảnh val:   {len(val_dataset)}")
    print(f"Số ảnh test:  {len(test_dataset)}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = build_model(num_classes=2, freeze_backbone=FREEZE_BACKBONE)
    model = model.to(DEVICE)

    if USE_CLASS_WEIGHTS:
        train_df = pd.read_csv(TRAIN_CSV)
        class_counts = train_df["label"].value_counts().sort_index()

        count_real = int(class_counts.get(0, 0))
        count_fake = int(class_counts.get(1, 0))

        # trọng số ngược với tần suất lớp
        total = count_real + count_fake
        weight_real = total / (2.0 * max(count_real, 1))
        weight_fake = total / (2.0 * max(count_fake, 1))

        class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32).to(DEVICE)

        print("Class weights:")
        print(f"  Real (0): {weight_real:.4f}")
        print(f"  Fake (1): {weight_fake:.4f}\n")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    history = []

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"========== EPOCH {epoch + 1}/{NUM_EPOCHS} ==========")

        train_loss, train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer
        )

        val_loss, val_metrics = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None
        )

        print(f"Train Loss: {train_loss:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"Prec: {train_metrics['precision']:.4f} | "
              f"Rec: {train_metrics['recall']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")

        print(f"Val   Loss: {val_loss:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"Prec: {val_metrics['precision']:.4f} | "
              f"Rec: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, BEST_MODEL_PATH)
            print(f"Đã lưu best model tại: {BEST_MODEL_PATH}")

        print()

    total_time = time.time() - start_time
    print(f"Thời gian train: {total_time / 60:.2f} phút\n")

    # Load best model
    model.load_state_dict(best_model_wts)

    print("===== ĐÁNH GIÁ TRÊN TEST SET =====")
    test_loss, test_metrics = run_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=None
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-score:  {test_metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    # Lưu history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    history_df.to_csv(history_path, index=False, encoding="utf-8")
    print(f"\nĐã lưu lịch sử train tại: {history_path}")


if __name__ == "__main__":
    main()