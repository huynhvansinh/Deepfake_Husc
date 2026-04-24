import os
import copy
import time
import math
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CẤU HÌNH
# =========================================================
TRAIN_CSV = "../processed_ffpp/splits/train.csv"
VAL_CSV = "../processed_ffpp/splits/val.csv"
TEST_CSV = "../processed_ffpp/splits/test.csv"

OUTPUT_DIR = "./training_outputs/densenet121_ffpp_v3"

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "last_model.pth")

BEST_CKPT_PATH = os.path.join(OUTPUT_DIR, "best_checkpoint.pth")
LAST_CKPT_PATH = os.path.join(OUTPUT_DIR, "last_checkpoint.pth")

HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.csv")

LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "loss_curve.png")
ACC_PLOT_PATH = os.path.join(OUTPUT_DIR, "accuracy_curve.png")
F1_PLOT_PATH = os.path.join(OUTPUT_DIR, "f1_curve.png")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_WORKERS = 4
RANDOM_SEED = 42

# ---- 2 phase ----
PHASE1_EPOCHS = 12
PHASE2_EPOCHS = NUM_EPOCHS - PHASE1_EPOCHS

# ---- LR theo từng phần model ----
LR_CLASSIFIER_PHASE1 = 7e-4
LR_BLOCK4_PHASE1 = 2e-4

LR_CLASSIFIER_PHASE2 = 5e-4
LR_BLOCK4_PHASE2 = 1.5e-4
LR_TRANS3_PHASE2 = 8e-5

WEIGHT_DECAY = 3e-4

USE_CLASS_WEIGHTS = False      # dùng sampler là chính, không cộng thêm class_weights để tránh over-correct
USE_WEIGHTED_SAMPLER = True
USE_AMP = True
USE_CHANNELS_LAST = True
USE_EMA = True
EMA_DECAY = 0.999

SAVE_BEST_BY = "f1"            # "f1", "acc", "loss"
RESUME_TRAINING = False        # True nếu muốn resume từ last_checkpoint.pth

LABEL_SMOOTHING = 0.03
DROPOUT_RATE_1 = 0.4
DROPOUT_RATE_2 = 0.2

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

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, df["val_loss"], marker="o", label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, df["val_acc"], marker="o", label="Val Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_f1"], marker="o", label="Train F1")
    plt.plot(epochs, df["val_f1"], marker="o", label="Val F1")
    plt.title("Training vs Validation F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_curve.png"), dpi=300)
    plt.close()


def save_history_csv(history, path):
    pd.DataFrame(history).to_csv(path, index=False, encoding="utf-8-sig")


def current_phase(epoch_idx_zero_based):
    # epoch 0..11 -> phase1, epoch 12..49 -> phase2
    return 1 if epoch_idx_zero_based < PHASE1_EPOCHS else 2


# =========================================================
# EMA
# =========================================================
class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.device = device
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        if self.device is not None:
            self.ema.to(self.device)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if k in msd:
                    model_v = msd[k].detach()
                    if self.device is not None:
                        model_v = model_v.to(self.device)
                    if v.dtype.is_floating_point:
                        v.copy_(v * self.decay + (1.0 - self.decay) * model_v)
                    else:
                        v.copy_(model_v)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)


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
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.88, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.06,
        hue=0.015
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
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
def build_model(num_classes=2):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(DROPOUT_RATE_1),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.SiLU(inplace=True),
        nn.Dropout(DROPOUT_RATE_2),
        nn.Linear(256, num_classes)
    )
    return model


def set_trainable_phase(model, phase):
    # đóng hết trước
    for p in model.features.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    # phase 1: classifier + denseblock4 + norm5
    for p in model.features.denseblock4.parameters():
        p.requires_grad = True
    for p in model.features.norm5.parameters():
        p.requires_grad = True

    # phase 2: mở thêm transition3
    if phase >= 2:
        for p in model.features.transition3.parameters():
            p.requires_grad = True


def build_optimizer(model, phase):
    param_groups = []

    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    block4_params = [p for p in model.features.denseblock4.parameters() if p.requires_grad]
    norm5_params = [p for p in model.features.norm5.parameters() if p.requires_grad]
    trans3_params = [p for p in model.features.transition3.parameters() if p.requires_grad]

    if phase == 1:
        if classifier_params:
            param_groups.append({"params": classifier_params, "lr": LR_CLASSIFIER_PHASE1})
        if block4_params or norm5_params:
            param_groups.append({"params": block4_params + norm5_params, "lr": LR_BLOCK4_PHASE1})
    else:
        if classifier_params:
            param_groups.append({"params": classifier_params, "lr": LR_CLASSIFIER_PHASE2})
        if block4_params or norm5_params:
            param_groups.append({"params": block4_params + norm5_params, "lr": LR_BLOCK4_PHASE2})
        if trans3_params:
            param_groups.append({"params": trans3_params, "lr": LR_TRANS3_PHASE2})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    return optimizer


def build_scheduler(optimizer, phase):
    tmax = PHASE1_EPOCHS if phase == 1 else max(PHASE2_EPOCHS, 1)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=tmax,
        eta_min=1e-6
    )


# =========================================================
# TRAIN / EVAL
# =========================================================
def run_one_epoch(model, dataloader, criterion, optimizer=None, scaler=None, ema=None):
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

                if ema is not None:
                    ema.update(model)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics


def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if USE_CHANNELS_LAST and DEVICE.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            if DEVICE.type == "cuda" and USE_AMP:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics


# =========================================================
# DATALOADER
# =========================================================
def create_weighted_sampler(dataset):
    labels = dataset.df["label"].astype(int).tolist()
    class_counts = np.bincount(labels)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def create_dataloader(dataset, batch_size, shuffle=False, sampler=None):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


# =========================================================
# CHECKPOINT
# =========================================================
def get_best_score(val_loss, val_acc, val_f1):
    if SAVE_BEST_BY == "f1":
        return val_f1
    elif SAVE_BEST_BY == "acc":
        return val_acc
    else:
        return -val_loss


def save_checkpoint(path, epoch, phase, model, optimizer, scheduler, scaler, ema, history, best_score):
    ckpt = {
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "history": history,
        "best_score": best_score,
        "save_best_by": SAVE_BEST_BY,
        "config": {
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "phase1_epochs": PHASE1_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING
        }
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, ema=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    if ema is not None and ckpt.get("ema_state_dict") is not None:
        ema.load_state_dict(ckpt["ema_state_dict"])

    return ckpt


def print_test_result(title, test_loss, test_metrics):
    print(title)
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-score:  {test_metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])


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
    print(f"IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"PHASE1_EPOCHS: {PHASE1_EPOCHS}")
    print(f"PHASE2_EPOCHS: {PHASE2_EPOCHS}")
    print(f"WEIGHT_DECAY: {WEIGHT_DECAY}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"USE_WEIGHTED_SAMPLER: {USE_WEIGHTED_SAMPLER}")
    print(f"USE_CLASS_WEIGHTS: {USE_CLASS_WEIGHTS}")
    print(f"USE_AMP: {USE_AMP}")
    print(f"USE_CHANNELS_LAST: {USE_CHANNELS_LAST}")
    print(f"USE_EMA: {USE_EMA}")
    print(f"EMA_DECAY: {EMA_DECAY}")
    print(f"LABEL_SMOOTHING: {LABEL_SMOOTHING}")
    print(f"SAVE_BEST_BY: {SAVE_BEST_BY}")
    print(f"RESUME_TRAINING: {RESUME_TRAINING}")
    print("=================================\n")

    train_dataset = FFPPDataset(TRAIN_CSV, transform=train_transform)
    val_dataset = FFPPDataset(VAL_CSV, transform=eval_transform)
    test_dataset = FFPPDataset(TEST_CSV, transform=eval_transform)

    print(f"Số ảnh train: {len(train_dataset)}")
    print(f"Số ảnh val:   {len(val_dataset)}")
    print(f"Số ảnh test:  {len(test_dataset)}\n")

    train_sampler = create_weighted_sampler(train_dataset) if USE_WEIGHTED_SAMPLER else None

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=not USE_WEIGHTED_SAMPLER,
        sampler=train_sampler
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    model = build_model(num_classes=2)

    if USE_CHANNELS_LAST and DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    model = model.to(DEVICE)

    set_trainable_phase(model, phase=1)
    optimizer = build_optimizer(model, phase=1)
    scheduler = build_scheduler(optimizer, phase=1)

    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" and USE_AMP else None
    ema = ModelEMA(model, decay=EMA_DECAY, device=DEVICE) if USE_EMA else None

    if USE_CLASS_WEIGHTS:
        class_counts = train_dataset.df["label"].value_counts().sort_index()
        count_real = int(class_counts.get(0, 0))
        count_fake = int(class_counts.get(1, 0))
        total = count_real + count_fake
        weight_real = total / (2.0 * max(count_real, 1))
        weight_fake = total / (2.0 * max(count_fake, 1))
        class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    history = []
    best_score = -float("inf")
    start_epoch = 0

    if RESUME_TRAINING and os.path.exists(LAST_CKPT_PATH):
        print(f"Đang resume từ: {LAST_CKPT_PATH}")
        ckpt = load_checkpoint(LAST_CKPT_PATH, model, optimizer, scheduler, scaler, ema)
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        best_score = ckpt.get("best_score", -float("inf"))

        phase_now = current_phase(start_epoch)
        set_trainable_phase(model, phase_now)
        optimizer = build_optimizer(model, phase_now)
        scheduler = build_scheduler(optimizer, phase_now)

        ckpt = load_checkpoint(LAST_CKPT_PATH, model, optimizer, scheduler, scaler, ema)
        print(f"Resume từ epoch {start_epoch + 1}/{NUM_EPOCHS}\n")

    start_time = time.time()

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            phase = current_phase(epoch)

            # chuyển phase đúng thời điểm
            if epoch == PHASE1_EPOCHS:
                print("\n========== CHUYỂN SANG PHASE 2 ==========")
                set_trainable_phase(model, phase=2)
                optimizer = build_optimizer(model, phase=2)
                scheduler = build_scheduler(optimizer, phase=2)

                # scaler giữ nguyên được
                # ema tiếp tục giữ
                print("Mở thêm: transition3 + denseblock4 + norm5 + classifier")
                print("=========================================\n")

            epoch_start = time.time()

            lr_list = [group["lr"] for group in optimizer.param_groups]
            print(f"========== EPOCH {epoch + 1}/{NUM_EPOCHS} | PHASE {phase} ==========")
            print("Learning Rates:", ", ".join([f"{lr:.8f}" for lr in lr_list]))

            train_loss, train_metrics = run_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema
            )

            eval_model = ema.ema if ema is not None else model
            val_loss, val_metrics = evaluate_model(
                model=eval_model,
                dataloader=val_loader,
                criterion=criterion
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
            else:
                mem_alloc = 0.0
                mem_reserved = 0.0

            epoch_minutes = (time.time() - epoch_start) / 60.0
            print(f"Thời gian epoch: {epoch_minutes:.2f} phút")

            history.append({
                "epoch": epoch + 1,
                "phase": phase,
                "lr_group_1": lr_list[0] if len(lr_list) > 0 else None,
                "lr_group_2": lr_list[1] if len(lr_list) > 1 else None,
                "lr_group_3": lr_list[2] if len(lr_list) > 2 else None,
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
                "gpu_mem_alloc_gb": mem_alloc,
                "gpu_mem_reserved_gb": mem_reserved,
                "epoch_minutes": epoch_minutes
            })

            # ===== LƯU NGAY SAU MỖI EPOCH =====
            save_history_csv(history, HISTORY_PATH)
            save_training_plots(history, OUTPUT_DIR)

            current_score = get_best_score(val_loss, val_acc, val_metrics["f1"])

            save_checkpoint(
                path=LAST_CKPT_PATH,
                epoch=epoch,
                phase=phase,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                history=history,
                best_score=best_score
            )
            torch.save(model.state_dict(), LAST_MODEL_PATH)

            if current_score > best_score:
                best_score = current_score

                best_model_source = ema.ema if ema is not None else model
                torch.save(best_model_source.state_dict(), BEST_MODEL_PATH)

                save_checkpoint(
                    path=BEST_CKPT_PATH,
                    epoch=epoch,
                    phase=phase,
                    model=best_model_source,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    history=history,
                    best_score=best_score
                )
                print(f"Đã lưu best model tại: {BEST_MODEL_PATH}")
                print(f"Đã lưu best checkpoint tại: {BEST_CKPT_PATH}")

            scheduler.step()
            print()

    except KeyboardInterrupt:
        print("\n[INFO] Đã nhận Ctrl+C. Đang lưu trạng thái hiện tại...")
        save_history_csv(history, HISTORY_PATH)
        save_training_plots(history, OUTPUT_DIR)

        phase = current_phase(max(len(history) - 1, 0))
        save_checkpoint(
            path=LAST_CKPT_PATH,
            epoch=max(len(history) - 1, 0),
            phase=phase,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            history=history,
            best_score=best_score
        )
        torch.save(model.state_dict(), LAST_MODEL_PATH)

        print("[INFO] Đã lưu history và last checkpoint.")
        print("[INFO] Sẽ đánh giá trên test bằng best checkpoint hiện có nếu tồn tại.\n")

    total_time = time.time() - start_time
    print(f"Thời gian train toàn bộ: {total_time / 60:.2f} phút\n")

    # =====================================================
    # ĐÁNH GIÁ TEST BẰNG BEST CHECKPOINT
    # =====================================================
    best_eval_model = build_model(num_classes=2)
    if USE_CHANNELS_LAST and DEVICE.type == "cuda":
        best_eval_model = best_eval_model.to(memory_format=torch.channels_last)
    best_eval_model = best_eval_model.to(DEVICE)

    if os.path.exists(BEST_MODEL_PATH):
        best_eval_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False))
        test_loss, test_metrics = evaluate_model(best_eval_model, test_loader, criterion)
        print_test_result("===== ĐÁNH GIÁ TRÊN TEST SET (BEST MODEL) =====", test_loss, test_metrics)
    else:
        print("[CẢNH BÁO] Chưa có BEST_MODEL_PATH nên không thể đánh giá best model trên test.")

    print(f"\nĐã lưu lịch sử train tại: {HISTORY_PATH}")
    print(f"Đã lưu best model tại: {BEST_MODEL_PATH}")
    print(f"Đã lưu last model tại: {LAST_MODEL_PATH}")
    print(f"Đã lưu best checkpoint tại: {BEST_CKPT_PATH}")
    print(f"Đã lưu last checkpoint tại: {LAST_CKPT_PATH}")
    print(f"Đã lưu biểu đồ loss tại: {LOSS_PLOT_PATH}")
    print(f"Đã lưu biểu đồ accuracy tại: {ACC_PLOT_PATH}")
    print(f"Đã lưu biểu đồ F1 tại: {F1_PLOT_PATH}")


if __name__ == "__main__":
    main()