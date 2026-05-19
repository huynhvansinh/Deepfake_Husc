import os
import copy
import time
import random
import warnings
import numpy as np
import pandas as pd
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

# ========== train bằng FFPP cũ ==========
# TRAIN_CSV = "../processed_ffpp/splits/train.csv"
# VAL_CSV   = "../processed_ffpp/splits/val.csv"
# TEST_CSV  = "../processed_ffpp/splits/test.csv"
# OUTPUT_DIR = "./training_outputs/efficientnet_b4_ffpp"

# ========== train bằng FFPP_02 ==========
TRAIN_CSV = "../processed_ffpp/splits/train.csv"
VAL_CSV   = "../processed_ffpp/splits/val.csv"
TEST_CSV  = "../processed_ffpp/splits/test.csv"

OUTPUT_DIR = "./training_outputs/efficientnet_b4_ffpp_sam_adamw_dropout_50epoch" 

# ==========================================

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "last_model.pth")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.csv")

CONFUSION_MATRIX_CSV_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_test.csv")
CONFUSION_MATRIX_NPY_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_test.npy")

IMAGE_SIZE = 224

# EfficientNet-B4 khá nặng.
# Nếu CUDA out of memory thì giảm BATCH_SIZE xuống 4.
BATCH_SIZE = 4

NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
RANDOM_SEED = 42

# True = khóa backbone, chỉ train classifier
# False = fine-tune toàn bộ model
FREEZE_BACKBONE = False

USE_CLASS_WEIGHTS = True

# Dropout tùy chỉnh ở classifier
DROPOUT_P = 0.4

# Label smoothing giúp model bớt quá tự tin
LABEL_SMOOTHING = 0.05

# =========================================================
# SAM CONFIG
# =========================================================
USE_SAM = True

# rho càng lớn regularize càng mạnh, nhưng dễ học chậm hơn
SAM_RHO = 0.05

# False = SAM thường
# True = ASAM-like adaptive perturbation, thường mạnh hơn nhưng có thể khó ổn định hơn
SAM_ADAPTIVE = False

# AMP chỉ dùng khi TRAIN.
# Lưu ý: khi USE_SAM=True, code sẽ tự tắt AMP để tránh lỗi GradScaler với 2 backward/step.
USE_AMP = True

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


def print_device_info():
    print("===== THÔNG TIN THIẾT BỊ =====")
    print(f"Thiết bị: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM tổng: {props.total_memory / (1024 ** 3):.2f} GB")

    print("================================\n")


# =========================================================
# SAM OPTIMIZER
# =========================================================
class SAM(torch.optim.Optimizer):
    """
    SAM - Sharpness-Aware Minimization.

    Ý tưởng:
    - Bước 1: đi tới điểm nhiễu theo hướng làm loss tăng mạnh nhất.
    - Bước 2: tối ưu model sao cho loss tại điểm nhiễu đó cũng thấp.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            rho = group["rho"]
            adaptive = group["adaptive"]

            scale = rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()

                if adaptive:
                    e_w = torch.pow(p, 2) * p.grad * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if "old_p" not in self.state[p]:
                    continue

                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("SAM cần gọi first_step() và second_step() thủ công.")

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device

        norms = []

        for group in self.param_groups:
            adaptive = group["adaptive"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if adaptive:
                    grad = torch.abs(p) * p.grad
                else:
                    grad = p.grad

                norms.append(torch.norm(grad, p=2).to(shared_device))

        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)

        norm = torch.norm(torch.stack(norms), p=2)
        return norm


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

        # Sửa đường dẫn ảnh để chạy được từ thư mục Model
        csv_abs_path = os.path.abspath(csv_path)
        splits_dir = os.path.dirname(csv_abs_path)
        processed_dir = os.path.dirname(splits_dir)
        project_root = os.path.dirname(processed_dir)

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
        removed_count = before_count - len(self.df)

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
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.05,
        hue=0.02
    ),

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
# TRAIN / EVAL
# =========================================================
def run_one_epoch(model, dataloader, criterion, optimizer=None, scaler=None, use_sam=False):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    effective_amp = (
        is_train
        and DEVICE.type == "cuda"
        and USE_AMP
        and not use_sam
    )

    for images, labels in dataloader:
        images = images.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
        labels = labels.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # =================================================
        # TRAIN VỚI SAM
        # =================================================
        if is_train and use_sam:
            # ---------- first forward-backward ----------
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            preds = torch.argmax(outputs, dim=1)

            optimizer.first_step(zero_grad=True)

            # ---------- second forward-backward ----------
            outputs_second = model(images)
            loss_second = criterion(outputs_second, labels)
            loss_second.backward()

            optimizer.second_step(zero_grad=True)

        # =================================================
        # TRAIN BÌNH THƯỜNG CÓ AMP
        # =================================================
        elif is_train and effective_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)

        # =================================================
        # TRAIN BÌNH THƯỜNG KHÔNG AMP HOẶC EVAL
        # =================================================
        else:
            with torch.set_grad_enabled(is_train):
                outputs = model(images)
                loss = criterion(outputs, labels)

                if is_train:
                    loss.backward()
                    optimizer.step()

            preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item() * images.size(0)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return epoch_loss, metrics


# =========================================================
# MODEL EFFICIENTNET-B4
# =========================================================
def build_model(num_classes=2, freeze_backbone=False, dropout_p=0.4):
    weights = models.EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features

    # Dropout tùy chỉnh.
    # EfficientNet-B4 mặc định cũng có dropout, nhưng ở đây ta set rõ để dễ kiểm soát.
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    return model


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    create_folder(OUTPUT_DIR)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print_device_info()

    effective_amp = USE_AMP and not USE_SAM

    print("===== THÔNG TIN THỰC NGHIỆM =====")
    print("Model: EfficientNet-B4")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Val CSV:   {VAL_CSV}")
    print(f"Test CSV:  {TEST_CSV}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"WEIGHT_DECAY: {WEIGHT_DECAY}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"FREEZE_BACKBONE: {FREEZE_BACKBONE}")
    print(f"USE_CLASS_WEIGHTS: {USE_CLASS_WEIGHTS}")
    print(f"DROPOUT_P: {DROPOUT_P}")
    print(f"LABEL_SMOOTHING: {LABEL_SMOOTHING}")
    print(f"USE_SAM: {USE_SAM}")
    print(f"SAM_RHO: {SAM_RHO}")
    print(f"SAM_ADAPTIVE: {SAM_ADAPTIVE}")
    print(f"USE_AMP khai báo: {USE_AMP}")
    print(f"USE_AMP thực tế: {effective_amp}")
    if USE_SAM and USE_AMP:
        print("Lưu ý: Đang dùng SAM nên AMP được tắt tự động để tránh lỗi 2 backward/step.")
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
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = build_model(
        num_classes=2,
        freeze_backbone=FREEZE_BACKBONE,
        dropout_p=DROPOUT_P
    )
    model = model.to(DEVICE)

    print("===== CLASSIFIER HIỆN TẠI =====")
    print(model.classifier)
    print("================================\n")

    # =====================================================
    # LOSS
    # =====================================================
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

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING
        )

    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=LABEL_SMOOTHING
        )

    # =====================================================
    # OPTIMIZER: ADAMW HOẶC SAM + ADAMW
    # =====================================================
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if USE_SAM:
        optimizer = SAM(
            trainable_params,
            base_optimizer=torch.optim.AdamW,
            rho=SAM_RHO,
            adaptive=SAM_ADAPTIVE,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        print("Optimizer: SAM + AdamW\n")
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        print("Optimizer: AdamW\n")

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda" and effective_amp)
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    history = []

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        print(f"========== EPOCH {epoch + 1}/{NUM_EPOCHS} ==========")

        train_loss, train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            use_sam=USE_SAM
        )

        val_loss, val_metrics = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            use_sam=False
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"Prec: {train_metrics['precision']:.4f} | "
            f"Rec: {train_metrics['recall']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )

        print(
            f"Val   Loss: {val_loss:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )

        if DEVICE.type == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

            print(f"GPU memory allocated     : {allocated:.2f} GB")
            print(f"GPU memory reserved      : {reserved:.2f} GB")
            print(f"GPU peak allocated       : {peak_allocated:.2f} GB")
            print(f"GPU peak reserved        : {peak_reserved:.2f} GB")

        print(f"Thời gian epoch: {epoch_time / 60:.2f} phút")

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

            "epoch_time_minutes": epoch_time / 60
        })

        # Lưu history sau mỗi epoch
        history_df = pd.DataFrame(history)
        history_df.to_csv(HISTORY_PATH, index=False, encoding="utf-8")

        # Lưu last model sau mỗi epoch
        torch.save(model.state_dict(), LAST_MODEL_PATH)

        # Lưu best model theo val_f1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, BEST_MODEL_PATH)
            print(f"Đã lưu best model tại: {BEST_MODEL_PATH}")

        print(f"Đã lưu last model tại: {LAST_MODEL_PATH}")
        print(f"Đã lưu history tại: {HISTORY_PATH}")
        print()

    total_time = time.time() - start_time

    print(f"Thời gian train toàn bộ: {total_time / 60:.2f} phút\n")

    # Load best model để đánh giá test
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    else:
        model.load_state_dict(best_model_wts)

    print("===== ĐÁNH GIÁ TRÊN TEST SET - BEST MODEL =====")

    test_loss, test_metrics = run_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=None,
        scaler=None,
        use_sam=False
    )

    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-score:  {test_metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    # =====================================================
    # LƯU MA TRẬN NHIỄU
    # =====================================================
    cm_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=["Actual_REAL_0", "Actual_FAKE_1"],
        columns=["Pred_REAL_0", "Pred_FAKE_1"]
    )

    cm_df.to_csv(CONFUSION_MATRIX_CSV_PATH, encoding="utf-8")
    np.save(CONFUSION_MATRIX_NPY_PATH, test_metrics["confusion_matrix"])

    print(f"\nĐã lưu lịch sử train tại: {HISTORY_PATH}")
    print(f"Đã lưu best model tại: {BEST_MODEL_PATH}")
    print(f"Đã lưu last model tại: {LAST_MODEL_PATH}")
    print(f"Đã lưu confusion matrix CSV tại: {CONFUSION_MATRIX_CSV_PATH}")
    print(f"Đã lưu confusion matrix NPY tại: {CONFUSION_MATRIX_NPY_PATH}")


if __name__ == "__main__":
    main()