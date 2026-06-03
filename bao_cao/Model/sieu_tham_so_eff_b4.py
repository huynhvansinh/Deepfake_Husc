import os
import copy
import time
import json
import random
import warnings
import gc

import optuna
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
# CẤU HÌNH CHÍNH
# =========================================================

TRAIN_CSV_LIST = [
    "../processed_ffpp_02/splits/train.csv",
    "../processed_celebdf_02/splits/train.csv",
]

VAL_CSV_LIST = [
    "../processed_ffpp_02/splits/val.csv",
    "../processed_celebdf_02/splits/val.csv",
]

# TEST TRÊN CẢ 2 DATASET
TEST_CSV_LIST = [
    "../processed_ffpp_02/splits/test.csv",
    "../processed_celebdf_02/splits/test.csv",
]

OUTPUT_DIR = "./training_outputs/optuna_effb4_sam_search"

STUDY_NAME = "effb4_sam_optuna_search"
STORAGE_PATH = os.path.join(OUTPUT_DIR, "optuna_study.db")
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"

BEST_PARAMS_PATH = os.path.join(OUTPUT_DIR, "best_params.json")
TRIALS_CSV_PATH = os.path.join(OUTPUT_DIR, "optuna_trials.csv")
BEST_TRIAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_trial_model.pth")

BEST_OVERALL_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_overall.pth")
BEST_OVERALL_RESULT_TXT = os.path.join(OUTPUT_DIR, "best_overall_result.txt")
BEST_OVERALL_PARAMS_JSON = os.path.join(OUTPUT_DIR, "best_overall_params.json")

BEST_OVERALL_TEST_RESULT_TXT = os.path.join(OUTPUT_DIR, "best_overall_test_result.txt")
CONFUSION_MATRIX_TEST_CSV_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_test.csv")
CONFUSION_MATRIX_TEST_NPY_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_test.npy")

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 4
RANDOM_SEED = 42


# =========================================================
# OPTUNA CONFIG
# =========================================================

N_TRIALS = 10
TRIAL_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 3

FREEZE_BACKBONE = False
USE_CLASS_WEIGHTS = True

USE_SAM = True
SAM_ADAPTIVE = False
USE_AMP = True

# Nếu True thì mỗi trial sẽ lưu thêm best_model.pth riêng.
# Tốn dung lượng, nên mặc định False.
SAVE_EACH_TRIAL_MODEL = False

SAVE_EACH_TRIAL_HISTORY = True

# Test sau khi Optuna chọn xong best trial.
RUN_TEST_AFTER_SEARCH = True

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


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_trial_dir(trial_number):
    trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial_number}")
    create_folder(trial_dir)
    return trial_dir


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_trial_result_txt(
    path,
    trial_number,
    params,
    best_val_f1,
    best_epoch,
    trial_time_minutes,
    train_csv_list,
    val_csv_list
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("===== TRIAL RESULT =====\n")
        f.write("========================\n")
        f.write(f"Trial number: {trial_number}\n")
        f.write(f"Best Val F1 : {best_val_f1:.6f}\n")
        f.write(f"Best epoch  : {best_epoch}\n")
        f.write(f"Trial time  : {trial_time_minutes:.2f} minutes\n\n")

        f.write("Best params:\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTrain CSV list:\n")
        for p in train_csv_list:
            f.write(f"- {p}\n")

        f.write("\nVal CSV list:\n")
        for p in val_csv_list:
            f.write(f"- {p}\n")


def save_best_overall_result_txt(path, study):
    with open(path, "w", encoding="utf-8") as f:
        f.write("===== BEST OVERALL TRIAL RESULT =====\n")
        f.write("=====================================\n")
        f.write(f"Trial number: {study.best_trial.number}\n")
        f.write(f"Best Val F1 : {study.best_value:.6f}\n\n")

        f.write("Best params:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")

        f.write("\nExtra info:\n")
        f.write(f"best_epoch: {study.best_trial.user_attrs.get('best_epoch', None)}\n")
        f.write(f"trial_time_minutes: {study.best_trial.user_attrs.get('trial_time_minutes', None)}\n")
        f.write(f"n_trials: {N_TRIALS}\n")
        f.write(f"trial_epochs: {TRIAL_EPOCHS}\n")
        f.write(f"early_stopping_patience: {EARLY_STOPPING_PATIENCE}\n")


def save_test_result_txt(path, test_loss, test_metrics, best_params, best_trial_number, method_ratio=None, source_ratio=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("===== BEST OVERALL TEST RESULT =====\n")
        f.write("====================================\n")
        f.write("Test set: FFPP + CelebDF\n")
        f.write(f"Best trial number: {best_trial_number}\n\n")

        f.write("Best params:\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTest metrics:\n")
        f.write(f"Test Loss:      {test_loss:.5f}\n")
        f.write(f"Test Accuracy:  {test_metrics['accuracy']:.5f}\n")
        f.write(f"Test Precision: {test_metrics['precision']:.5f}\n")
        f.write(f"Test Recall:    {test_metrics['recall']:.5f}\n")
        f.write(f"Test F1-score:  {test_metrics['f1']:.5f}\n")

        f.write("\nConfusion Matrix:\n")
        f.write(str(test_metrics["confusion_matrix"]))
        f.write("\n")

        if source_ratio is not None:
            f.write("\n===== SOURCE RATIO =====\n")
            f.write(str(source_ratio))
            f.write("\n")

        if method_ratio is not None:
            f.write("\n===== METHOD RATIO =====\n")
            f.write(str(method_ratio))
            f.write("\n")


# =========================================================
# SAM OPTIMIZER
# =========================================================

class SAM(torch.optim.Optimizer):
    """
    SAM - Sharpness-Aware Minimization.
    Dùng với base optimizer AdamW.
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

        return torch.norm(torch.stack(norms), p=2)


# =========================================================
# DATASET
# =========================================================

class FaceDataset(Dataset):
    """
    Dataset đọc 1 CSV.

    CSV bắt buộc có:
    - image_path
    - label: 0 = REAL, 1 = FAKE
    """

    def __init__(self, csv_path, transform=None, source_name=None):
        self.csv_path = csv_path
        self.transform = transform
        self.source_name = source_name if source_name is not None else os.path.basename(csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy CSV: {csv_path}")

        self.df = pd.read_csv(csv_path, low_memory=False).copy()

        required_cols = ["image_path", "label"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV thiếu cột bắt buộc '{col}': {csv_path}")

        self.df["source_csv"] = csv_path
        self.df["source_name"] = self.source_name

        csv_abs_path = os.path.abspath(csv_path)
        splits_dir = os.path.dirname(csv_abs_path)
        processed_dir = os.path.dirname(splits_dir)
        project_root = os.path.dirname(processed_dir)

        def resolve_image_path(p):
            p = str(p).strip()
            p = p.replace("\\", os.sep).replace("/", os.sep)

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
            print(f"[{self.source_name}] Đã loại {removed_count} ảnh không tồn tại.")

        if len(self.df) == 0:
            raise ValueError(f"Không còn ảnh hợp lệ trong {csv_path}")

        self.df["label"] = self.df["label"].astype(int)

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


class MultiCSVDataset(Dataset):
    """
    Gộp nhiều CSV thành 1 dataset.
    Dùng cho train/val/test FFPP + CelebDF.
    """

    def __init__(self, csv_paths, transform=None, split_name="train"):
        if not isinstance(csv_paths, (list, tuple)):
            csv_paths = [csv_paths]

        if len(csv_paths) == 0:
            raise ValueError(f"{split_name}: csv_paths đang rỗng.")

        self.datasets = []
        dfs = []

        for csv_path in csv_paths:
            normalized = csv_path.replace("\\", "/").lower()

            if "ffpp" in normalized or "faceforensics" in normalized:
                source_name = "FFPP"
            elif "celeb" in normalized:
                source_name = "CelebDF"
            else:
                source_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))

            ds = FaceDataset(
                csv_path=csv_path,
                transform=transform,
                source_name=f"{split_name}_{source_name}"
            )

            self.datasets.append(ds)
            dfs.append(ds.df)

        self.df = pd.concat(dfs, ignore_index=True)
        self.cumulative_sizes = np.cumsum([len(ds) for ds in self.datasets]).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx

        dataset_idx = int(np.searchsorted(self.cumulative_sizes, idx, side="right"))

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]


def print_dataset_stats(dataset, name):
    print(f"===== THỐNG KÊ {name} =====")
    print(f"Tổng ảnh: {len(dataset)}")

    if hasattr(dataset, "df"):
        source_counts = dataset.df["source_name"].value_counts()
        print("Theo nguồn:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")

        label_counts = dataset.df["label"].value_counts().sort_index()
        print("Theo label:")
        print(f"  REAL 0: {int(label_counts.get(0, 0))}")
        print(f"  FAKE 1: {int(label_counts.get(1, 0))}")

        cross = pd.crosstab(dataset.df["source_name"], dataset.df["label"])
        print("Theo nguồn và label:")
        for source in cross.index:
            real_count = int(cross.loc[source].get(0, 0))
            fake_count = int(cross.loc[source].get(1, 0))
            print(f"  {source}: REAL={real_count}, FAKE={fake_count}")

        if "method" in dataset.df.columns:
            print("Theo method:")
            method_counts = dataset.df["method"].value_counts()
            for method, count in method_counts.items():
                print(f"  {method}: {count}")

    print("================================\n")


# =========================================================
# TRANSFORMS
# =========================================================

def build_train_transform():
    return transforms.Compose([
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


def build_eval_transform():
    return transforms.Compose([
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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

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

        if is_train and use_sam:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            preds = torch.argmax(outputs, dim=1)

            optimizer.first_step(zero_grad=True)

            outputs_second = model(images)
            loss_second = criterion(outputs_second, labels)
            loss_second.backward()

            optimizer.second_step(zero_grad=True)

        elif is_train and effective_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)

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

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=True),
        nn.Linear(in_features, num_classes)
    )

    return model


# =========================================================
# LOSS
# =========================================================

def build_criterion(train_dataset, label_smoothing):
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
        print(f"  Fake (1): {weight_fake:.4f}")

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )

    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    return criterion


# =========================================================
# OPTUNA OBJECTIVE
# =========================================================

def objective(trial):
    cleanup_memory()
    set_seed(RANDOM_SEED + trial.number)

    learning_rate = trial.suggest_float(
        "LEARNING_RATE",
        1e-6,
        5e-4,
        log=True
    )

    weight_decay = trial.suggest_float(
        "WEIGHT_DECAY",
        1e-6,
        1e-2,
        log=True
    )

    dropout_p = trial.suggest_float(
        "DROPOUT_P",
        0.2,
        0.6
    )

    label_smoothing = trial.suggest_float(
        "LABEL_SMOOTHING",
        0.0,
        0.12
    )

    sam_rho = trial.suggest_float(
        "SAM_RHO",
        0.02,
        0.10
    )

    print("\n==================================================")
    print(f"TRIAL {trial.number}")
    print("==================================================")
    print(f"LEARNING_RATE:   {learning_rate}")
    print(f"WEIGHT_DECAY:    {weight_decay}")
    print(f"DROPOUT_P:       {dropout_p}")
    print(f"LABEL_SMOOTHING: {label_smoothing}")
    print(f"SAM_RHO:         {sam_rho}")
    print("==================================================\n")

    trial_dir = get_trial_dir(trial.number)

    trial_params = {
        "trial_number": trial.number,
        "LEARNING_RATE": learning_rate,
        "WEIGHT_DECAY": weight_decay,
        "DROPOUT_P": dropout_p,
        "LABEL_SMOOTHING": label_smoothing,
        "SAM_RHO": sam_rho,
        "BATCH_SIZE": BATCH_SIZE,
        "IMAGE_SIZE": IMAGE_SIZE,
        "TRIAL_EPOCHS": TRIAL_EPOCHS,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "USE_SAM": USE_SAM,
        "SAM_ADAPTIVE": SAM_ADAPTIVE,
        "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
        "FREEZE_BACKBONE": FREEZE_BACKBONE,
        "TRAIN_CSV_LIST": TRAIN_CSV_LIST,
        "VAL_CSV_LIST": VAL_CSV_LIST,
        "TEST_CSV_LIST": TEST_CSV_LIST,
    }

    save_json(os.path.join(trial_dir, "trial_params.json"), trial_params)

    train_transform = build_train_transform()
    eval_transform = build_eval_transform()

    train_dataset = MultiCSVDataset(
        TRAIN_CSV_LIST,
        transform=train_transform,
        split_name="train"
    )

    val_dataset = MultiCSVDataset(
        VAL_CSV_LIST,
        transform=eval_transform,
        split_name="val"
    )

    if trial.number == 0:
        print_dataset_stats(train_dataset, "TRAIN")
        print_dataset_stats(val_dataset, "VAL")

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

    model = build_model(
        num_classes=2,
        freeze_backbone=FREEZE_BACKBONE,
        dropout_p=dropout_p
    )
    model = model.to(DEVICE)

    criterion = build_criterion(
        train_dataset=train_dataset,
        label_smoothing=label_smoothing
    )

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = SAM(
        trainable_params,
        base_optimizer=torch.optim.AdamW,
        rho=sam_rho,
        adaptive=SAM_ADAPTIVE,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    effective_amp = USE_AMP and not USE_SAM

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda" and effective_amp)
    )

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    trial_history = []

    trial_start = time.time()

    for epoch in range(TRIAL_EPOCHS):
        epoch_start = time.time()

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        print(f"\n========== TRIAL {trial.number} | EPOCH {epoch + 1}/{TRIAL_EPOCHS} ==========")

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

        val_f1 = val_metrics["f1"]
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

            print(f"GPU memory allocated: {allocated:.2f} GB")
            print(f"GPU memory reserved : {reserved:.2f} GB")
            print(f"GPU peak allocated  : {peak_allocated:.2f} GB")
            print(f"GPU peak reserved   : {peak_reserved:.2f} GB")

        print(f"Thời gian epoch: {epoch_time / 60:.2f} phút")

        trial_history.append({
            "trial": trial.number,
            "epoch": epoch + 1,

            "LEARNING_RATE": learning_rate,
            "WEIGHT_DECAY": weight_decay,
            "DROPOUT_P": dropout_p,
            "LABEL_SMOOTHING": label_smoothing,
            "SAM_RHO": sam_rho,

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

            "best_val_f1_so_far": max(best_val_f1, val_f1),
            "epoch_time_minutes": epoch_time / 60
        })

        if SAVE_EACH_TRIAL_HISTORY:
            history_df = pd.DataFrame(trial_history)
            history_df.to_csv(
                os.path.join(trial_dir, "trial_history.csv"),
                index=False,
                encoding="utf-8-sig"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())

            print(f"Trial {trial.number}: best_val_f1 mới = {best_val_f1:.4f} tại epoch {best_epoch}")
        else:
            patience_counter += 1
            print(f"Trial {trial.number}: val_f1 không tăng. Patience {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        trial.report(best_val_f1, step=epoch)

        if trial.should_prune():
            print(f"Trial {trial.number} bị Optuna prune tại epoch {epoch + 1}.")
            raise optuna.exceptions.TrialPruned()

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Trial {trial.number} dừng sớm do EarlyStopping tại epoch {epoch + 1}.")
            break

    trial_time = time.time() - trial_start

    print("\n==================================================")
    print(f"KẾT THÚC TRIAL {trial.number}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Best epoch:  {best_epoch}")
    print(f"Thời gian trial: {trial_time / 60:.2f} phút")
    print("==================================================\n")

    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("trial_time_minutes", trial_time / 60)

    trial_result = {
        "trial_number": trial.number,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "trial_time_minutes": trial_time / 60,
        "params": trial_params
    }

    save_json(
        os.path.join(trial_dir, "trial_result.json"),
        trial_result
    )

    save_trial_result_txt(
        path=os.path.join(trial_dir, "trial_result.txt"),
        trial_number=trial.number,
        params=trial_params,
        best_val_f1=best_val_f1,
        best_epoch=best_epoch,
        trial_time_minutes=trial_time / 60,
        train_csv_list=TRAIN_CSV_LIST,
        val_csv_list=VAL_CSV_LIST
    )

    if SAVE_EACH_TRIAL_MODEL:
        torch.save(
            best_model_wts,
            os.path.join(trial_dir, "best_model.pth")
        )

    current_best = None
    try:
        current_best = trial.study.best_value
    except ValueError:
        current_best = None

    if current_best is None or best_val_f1 >= current_best:
        torch.save(best_model_wts, BEST_TRIAL_MODEL_PATH)
        torch.save(best_model_wts, BEST_OVERALL_MODEL_PATH)

        print(f"Đã lưu best trial model tạm thời tại: {BEST_TRIAL_MODEL_PATH}")
        print(f"Đã lưu best overall model tại: {BEST_OVERALL_MODEL_PATH}")

    del model
    del optimizer
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    cleanup_memory()

    return best_val_f1


# =========================================================
# TEST BEST MODEL SAU KHI SEARCH
# =========================================================

def evaluate_best_model_on_test(study):
    if not RUN_TEST_AFTER_SEARCH:
        return

    for test_csv in TEST_CSV_LIST:
        if not os.path.exists(test_csv):
            print(f"[CẢNH BÁO] Không tìm thấy TEST_CSV: {test_csv}")
            return

    if not os.path.exists(BEST_OVERALL_MODEL_PATH):
        print(f"[CẢNH BÁO] Không tìm thấy best model: {BEST_OVERALL_MODEL_PATH}")
        return

    print("\n===== ĐÁNH GIÁ BEST OVERALL MODEL TRÊN TEST SET =====")
    print("Lưu ý: Test chỉ chạy sau khi Optuna đã chọn xong best trial.")
    print("Test set: FFPP + CelebDF\n")

    best_params = study.best_params

    dropout_p = float(best_params["DROPOUT_P"])
    label_smoothing = float(best_params["LABEL_SMOOTHING"])

    eval_transform = build_eval_transform()

    test_dataset = MultiCSVDataset(
        TEST_CSV_LIST,
        transform=eval_transform,
        split_name="test"
    )

    print_dataset_stats(test_dataset, "TEST - FFPP + CELEBDF")

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
        dropout_p=dropout_p
    )

    model.load_state_dict(torch.load(BEST_OVERALL_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )

    test_loss, test_metrics = run_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=None,
        scaler=None,
        use_sam=False
    )

    print(f"Test Loss:      {test_loss:.5f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.5f}")
    print(f"Test Precision: {test_metrics['precision']:.5f}")
    print(f"Test Recall:    {test_metrics['recall']:.5f}")
    print(f"Test F1-score:  {test_metrics['f1']:.5f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    cm_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=["Actual_REAL_0", "Actual_FAKE_1"],
        columns=["Pred_REAL_0", "Pred_FAKE_1"]
    )

    cm_df.to_csv(CONFUSION_MATRIX_TEST_CSV_PATH, encoding="utf-8-sig")
    np.save(CONFUSION_MATRIX_TEST_NPY_PATH, test_metrics["confusion_matrix"])

    source_ratio = None
    if "source_name" in test_dataset.df.columns:
        source_ratio = test_dataset.df["source_name"].value_counts(normalize=True) * 100

    method_ratio = None
    if "method" in test_dataset.df.columns:
        method_ratio = test_dataset.df["method"].value_counts(normalize=True) * 100

    save_test_result_txt(
        path=BEST_OVERALL_TEST_RESULT_TXT,
        test_loss=test_loss,
        test_metrics=test_metrics,
        best_params=best_params,
        best_trial_number=study.best_trial.number,
        method_ratio=method_ratio,
        source_ratio=source_ratio
    )

    print(f"\nĐã lưu test result tại: {BEST_OVERALL_TEST_RESULT_TXT}")
    print(f"Đã lưu confusion matrix CSV tại: {CONFUSION_MATRIX_TEST_CSV_PATH}")
    print(f"Đã lưu confusion matrix NPY tại: {CONFUSION_MATRIX_TEST_NPY_PATH}")

    del model
    del test_loader
    del test_dataset
    cleanup_memory()


# =========================================================
# MAIN
# =========================================================

def main():
    create_folder(OUTPUT_DIR)
    set_seed(RANDOM_SEED)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print_device_info()

    print("===== CẤU HÌNH OPTUNA SEARCH =====")
    print(f"Model: EfficientNet-B4")
    print(f"N_TRIALS: {N_TRIALS}")
    print(f"TRIAL_EPOCHS: {TRIAL_EPOCHS}")
    print(f"EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}")
    print(f"IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"USE_SAM: {USE_SAM}")
    print(f"SAM_ADAPTIVE: {SAM_ADAPTIVE}")
    print(f"USE_CLASS_WEIGHTS: {USE_CLASS_WEIGHTS}")
    print(f"FREEZE_BACKBONE: {FREEZE_BACKBONE}")
    print(f"SAVE_EACH_TRIAL_MODEL: {SAVE_EACH_TRIAL_MODEL}")
    print(f"SAVE_EACH_TRIAL_HISTORY: {SAVE_EACH_TRIAL_HISTORY}")
    print(f"RUN_TEST_AFTER_SEARCH: {RUN_TEST_AFTER_SEARCH}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    print("\nTrain CSV list:")
    for p in TRAIN_CSV_LIST:
        print(f"  - {p}")

    print("\nVal CSV list:")
    for p in VAL_CSV_LIST:
        print(f"  - {p}")

    print("\nTest CSV list:")
    for p in TEST_CSV_LIST:
        print(f"  - {p}")

    print("=================================\n")

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=2,
        max_resource=TRIAL_EPOCHS,
        reduction_factor=3
    )

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_SEED
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=STORAGE_URL,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        gc_after_trial=True
    )

    print("\n================ KẾT QUẢ OPTUNA ================")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best val F1: {study.best_value:.6f}")

    print("\nBest params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_result = {
        "best_trial_number": study.best_trial.number,
        "best_val_f1": study.best_value,
        "best_params": study.best_params,
        "best_epoch": study.best_trial.user_attrs.get("best_epoch", None),
        "trial_time_minutes": study.best_trial.user_attrs.get("trial_time_minutes", None),
        "n_trials": N_TRIALS,
        "trial_epochs": TRIAL_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "train_csv_list": TRAIN_CSV_LIST,
        "val_csv_list": VAL_CSV_LIST,
        "test_csv_list": TEST_CSV_LIST
    }

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_result, f, ensure_ascii=False, indent=4)

    save_json(BEST_OVERALL_PARAMS_JSON, best_result)
    save_best_overall_result_txt(BEST_OVERALL_RESULT_TXT, study)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(TRIALS_CSV_PATH, index=False, encoding="utf-8-sig")

    evaluate_best_model_on_test(study)

    print("\nĐã lưu:")
    print(f"  Best params:              {BEST_PARAMS_PATH}")
    print(f"  Best overall params:      {BEST_OVERALL_PARAMS_JSON}")
    print(f"  Best overall result txt:  {BEST_OVERALL_RESULT_TXT}")
    print(f"  Trials CSV:               {TRIALS_CSV_PATH}")
    print(f"  Study DB:                 {STORAGE_PATH}")
    print(f"  Best trial model:         {BEST_TRIAL_MODEL_PATH}")
    print(f"  Best overall model:       {BEST_OVERALL_MODEL_PATH}")

    if RUN_TEST_AFTER_SEARCH:
        print(f"  Test result txt:          {BEST_OVERALL_TEST_RESULT_TXT}")
        print(f"  Confusion matrix CSV:     {CONFUSION_MATRIX_TEST_CSV_PATH}")
        print(f"  Confusion matrix NPY:     {CONFUSION_MATRIX_TEST_NPY_PATH}")

    print("=================================================")


if __name__ == "__main__":
    main()