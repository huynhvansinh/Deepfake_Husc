import os
import copy
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CẤU HÌNH
# =========================================================
TRAIN_CSV = "../processed_ffpp/splits/train.csv"
VAL_CSV = "../processed_ffpp/splits/val.csv"
TEST_CSV = "../processed_ffpp/splits/test.csv"

OUTPUT_DIR = "./training_outputs/densenet121_ffpp_optimized_v3"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "last_model.pth")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.csv")

LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "loss_curve.png")
ACC_PLOT_PATH = os.path.join(OUTPUT_DIR, "accuracy_curve.png")
F1_PLOT_PATH = os.path.join(OUTPUT_DIR, "f1_curve.png")

IMAGE_SIZE = 224
BATCH_SIZE = 64

NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
RANDOM_SEED = 42

FREEZE_BACKBONE = False
USE_CLASS_WEIGHTS = True
USE_AMP = True
USE_CHANNELS_LAST = True
SAVE_BEST_BY = "f1"   # "f1", "loss", "acc"

DROPOUT_RATE = 0.3

# ---- EARLY STOPPING / OVERFITTING ----
EARLY_STOPPING_PATIENCE = 5        # val acc không cải thiện sau N epoch thì dừng
EARLY_STOPPING_MIN_DELTA = 0.001   # mức cải thiện tối thiểu để tính là "có tiến bộ"

OVERFIT_GAP_THRESHOLD = 0.04       # nếu train_acc - val_acc > 4% thì coi là có dấu hiệu overfit
OVERFIT_PATIENCE = 3               # nếu kéo dài N epoch liên tiếp thì dừng

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# HỖ TRỢ
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_gpu_info():
    print("===== THÔNG TIN THIẾT BỊ =====")
    print(f"Thiết bị đang dùng: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM tổng: {props.total_memory / 1024**3:.2f} GB")
        print(f"CUDA capability: {props.major}.{props.minor}")
    print("================================\n")


def save_training_plots(history, output_dir):
    if len(history) == 0:
        return

    df = pd.DataFrame(history)
    epochs = df["epoch"]

    # ---- Loss ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_loss"], marker='o', label="Train Loss")
    plt.plot(epochs, df["val_loss"], marker='o', label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # ---- Accuracy ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_acc"], marker='o', label="Train Accuracy")
    plt.plot(epochs, df["val_acc"], marker='o', label="Val Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    # ---- F1 ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_f1"], marker='o', label="Train F1")
    plt.plot(epochs, df["val_f1"], marker='o', label="Val F1")
    plt.title("Training vs Validation F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_curve.png"), dpi=300)
    plt.close()


def should_stop_early(
    val_acc,
    best_val_acc,
    no_improve_count,
    train_acc,
    overfit_count
):
    stop_by_val = no_improve_count >= EARLY_STOPPING_PATIENCE
    stop_by_overfit = overfit_count >= OVERFIT_PATIENCE

    if stop_by_val:
        print(f"[EARLY STOP] Val Accuracy không cải thiện trong {EARLY_STOPPING_PATIENCE} epoch liên tiếp.")
        return True

    if stop_by_overfit:
        print(
            f"[EARLY STOP] Train Acc cao hơn Val Acc quá nhiều "
            f"(>{OVERFIT_GAP_THRESHOLD:.2%}) trong {OVERFIT_PATIENCE} epoch liên tiếp."
        )
        return True

    return False


# =========================================================
# DATASET
# =========================================================
class FFPPDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path).copy()
        self.transform = transform
        self.csv_path = csv_path

        required_cols = ["image_path", "label"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV thiếu cột bắt buộc: {col}")

        csv_abs_path = os.path.abspath(csv_path)
        splits_dir = os.path.dirname(csv_abs_path)
        processed_ffpp_dir = os.path.dirname(splits_dir)
        project_root = os.path.dirname(processed_ffpp_dir)

        def resolve_image_path(p):
            p = str(p).replace("\\", os.sep).replace("/", os.sep)

            if os.path.isabs(p):
                return p

            if p.startswith("." + os.sep):
                p = p[2:]

            return os.path.normpath(os.path.join(project_root, p))

        self.df["image_path"] = self.df["image_path"].apply(resolve_image_path)

        before_count = len(self.df)
        exists_mask = self.df["image_path"].apply(os.path.exists)
        self.df = self.df[exists_mask].reset_index(drop=True)
        after_count = len(self.df)

        removed_count = before_count - after_count
        if removed_count > 0:
            print(f"[{os.path.basename(csv_path)}] Đã loại {removed_count} ảnh không tồn tại.")

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
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.12,
        contrast=0.12,
        saturation=0.08,
        hue=0.02
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
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
# MODEL
# =========================================================
def build_model(num_classes=2, freeze_backbone=False):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(in_features, num_classes)
    )

    return model


# =========================================================
# TRAIN / EVAL
# =========================================================
def run_one_epoch(model, dataloader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in dataloader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if USE_CHANNELS_LAST and DEVICE.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if DEVICE.type == "cuda" and USE_AMP:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_train:
                if scaler is not None and DEVICE.type == "cuda" and USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
# DATA LOADER
# =========================================================
def create_dataloader(dataset, batch_size, shuffle):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    create_folder(OUTPUT_DIR)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print_gpu_info()

    print("===== THÔNG TIN THỰC NGHIỆM =====")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Val CSV:   {VAL_CSV}")
    print(f"Test CSV:  {TEST_CSV}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"WEIGHT_DECAY: {WEIGHT_DECAY}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"FREEZE_BACKBONE: {FREEZE_BACKBONE}")
    print(f"USE_CLASS_WEIGHTS: {USE_CLASS_WEIGHTS}")
    print(f"USE_AMP: {USE_AMP}")
    print(f"USE_CHANNELS_LAST: {USE_CHANNELS_LAST}")
    print(f"DROPOUT_RATE: {DROPOUT_RATE}")
    print(f"SAVE_BEST_BY: {SAVE_BEST_BY}")
    print(f"EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}")
    print(f"OVERFIT_GAP_THRESHOLD: {OVERFIT_GAP_THRESHOLD}")
    print(f"OVERFIT_PATIENCE: {OVERFIT_PATIENCE}")
    print("=================================\n")

    train_dataset = FFPPDataset(TRAIN_CSV, transform=train_transform)
    val_dataset = FFPPDataset(VAL_CSV, transform=eval_transform)
    test_dataset = FFPPDataset(TEST_CSV, transform=eval_transform)

    print(f"Số ảnh train: {len(train_dataset)}")
    print(f"Số ảnh val:   {len(val_dataset)}")
    print(f"Số ảnh test:  {len(test_dataset)}\n")

    train_loader = create_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_dataset, BATCH_SIZE, shuffle=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    model = build_model(num_classes=2, freeze_backbone=FREEZE_BACKBONE)

    if USE_CHANNELS_LAST and DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    model = model.to(DEVICE)

    if USE_CLASS_WEIGHTS:
        class_counts = train_dataset.df["label"].value_counts().sort_index()

        count_real = int(class_counts.get(0, 0))
        count_fake = int(class_counts.get(1, 0))

        total = count_real + count_fake
        weight_real = total / (2.0 * max(count_real, 1))
        weight_fake = total / (2.0 * max(count_fake, 1))

        class_weights = torch.tensor(
            [weight_real, weight_fake],
            dtype=torch.float32,
            device=DEVICE
        )

        print("Class weights:")
        print(f"  Real (0): {weight_real:.4f}")
        print(f"  Fake (1): {weight_fake:.4f}\n")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if SAVE_BEST_BY in ["f1", "acc"] else "min",
        factor=0.5,
        patience=2
    )

    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" and USE_AMP else None

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_val_acc = -1.0

    no_improve_count = 0
    overfit_count = 0

    history = []
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"========== EPOCH {epoch + 1}/{NUM_EPOCHS} ==========")
        print(f"Learning Rate: {current_lr:.8f}")

        train_loss, train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler
        )

        val_loss, val_metrics = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None
        )

        train_acc = train_metrics["accuracy"]
        val_acc = val_metrics["accuracy"]
        acc_gap = train_acc - val_acc

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Acc: {train_acc:.4f} | "
            f"Prec: {train_metrics['precision']:.4f} | "
            f"Rec: {train_metrics['recall']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )

        print(
            f"Val   Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )

        print(f"Chênh lệch Train Acc - Val Acc: {acc_gap:.4f}")

        if DEVICE.type == "cuda":
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory allocated: {mem_alloc:.2f} GB")
            print(f"GPU memory reserved : {mem_reserved:.2f} GB")

        epoch_minutes = (time.time() - epoch_start) / 60.0
        print(f"Thời gian epoch: {epoch_minutes:.2f} phút")

        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "acc_gap": acc_gap,
            "gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1024**3 if DEVICE.type == "cuda" else 0.0,
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1024**3 if DEVICE.type == "cuda" else 0.0,
            "epoch_minutes": epoch_minutes
        })

        # cập nhật biểu đồ sau mỗi epoch
        save_training_plots(history, OUTPUT_DIR)

        # ---- scheduler ----
        if SAVE_BEST_BY == "f1":
            scheduler.step(val_metrics["f1"])
        elif SAVE_BEST_BY == "acc":
            scheduler.step(val_acc)
        else:
            scheduler.step(val_loss)

        # ---- lưu best model ----
        improved = False

        if SAVE_BEST_BY == "f1":
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                improved = True

        elif SAVE_BEST_BY == "acc":
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                improved = True

        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True

        if improved:
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, BEST_MODEL_PATH)
            print(f"Đã lưu best model tại: {BEST_MODEL_PATH}")

        # ---- early stop theo val accuracy ----
        if val_acc > best_val_acc + EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"Không cải thiện val_acc: {no_improve_count}/{EARLY_STOPPING_PATIENCE}")

        # ---- early stop theo overfitting gap ----
        if acc_gap > OVERFIT_GAP_THRESHOLD:
            overfit_count += 1
            print(f"Cảnh báo overfitting gap: {overfit_count}/{OVERFIT_PATIENCE}")
        else:
            overfit_count = 0

        torch.save(model.state_dict(), LAST_MODEL_PATH)
        print()

        if should_stop_early(
            val_acc=val_acc,
            best_val_acc=best_val_acc,
            no_improve_count=no_improve_count,
            train_acc=train_acc,
            overfit_count=overfit_count
        ):
            break

    total_time = time.time() - start_time
    print(f"Thời gian train toàn bộ: {total_time / 60:.2f} phút\n")

    model.load_state_dict(best_model_wts)

    print("===== ĐÁNH GIÁ TRÊN TEST SET =====")
    test_loss, test_metrics = run_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=None,
        scaler=None
    )

    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-score:  {test_metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")

    print(f"\nĐã lưu lịch sử train tại: {HISTORY_PATH}")
    print(f"Đã lưu best model tại: {BEST_MODEL_PATH}")
    print(f"Đã lưu last model tại: {LAST_MODEL_PATH}")
    print(f"Đã lưu biểu đồ loss tại: {LOSS_PLOT_PATH}")
    print(f"Đã lưu biểu đồ accuracy tại: {ACC_PLOT_PATH}")
    print(f"Đã lưu biểu đồ F1 tại: {F1_PLOT_PATH}")


if __name__ == "__main__":
    main()