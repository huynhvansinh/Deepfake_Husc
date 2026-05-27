import os
import time
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# =========================================================
# CẤU HÌNH MẶC ĐỊNH
# =========================================================

# File này đặt trong:
# D:/Nghiencuu/Model/test_model.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root là:
# D:/Nghiencuu
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_NAME = "efficientnet_b4"
NUM_CLASSES = 2

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 4

# CSV tổng sau khi lọc ngưỡng blur
FILTERED_CSV = os.path.join(
    PROJECT_ROOT,
    "processed_celebdf_02",
    "splits_blur20",
    "test.csv"
)

# Model đã train
CHECKPOINT_PATH = os.path.join(
    SCRIPT_DIR,
    "training_outputs",
    "efficientnet_b4_ffpp_celebdf_train_test_ffpp_sam_adamw_dropout_50epoch",
    "best_model.pth"
)

# Thư mục lưu kết quả test
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "testing_outputs",
    "test_on_celebdf_filtered_model_efficientnet_b4_ffpp_celebdf_train_test_ffpp_sam_adamw_dropout_50epoch"
)

# Nếu model lúc train có dropout custom thì chỉnh đúng với lúc train
DROPOUT_RATE = 0.4

USE_AMP = True


# =========================================================
# HÀM PHỤ
# =========================================================
def create_folder(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_divide(a, b):
    if b == 0:
        return 0.0
    return a / b


def resolve_image_path(image_path):
    """
    Fix lỗi đường dẫn ảnh trong CSV.

    CSV đang có dạng:
    ./processed_celebdf_02/cropfaces_retina_filtered/...

    Nhưng khi chạy test_model.py từ D:/Nghiencuu/Model,
    nếu dùng trực tiếp os.path.exists(image_path), Python sẽ hiểu nhầm thành:
    D:/Nghiencuu/Model/processed_celebdf_02/...

    Trong khi ảnh thật nằm ở:
    D:/Nghiencuu/processed_celebdf_02/...
    """
    if pd.isna(image_path):
        return None

    image_path = str(image_path).strip().replace("\\", "/")

    if image_path == "":
        return None

    candidates = []

    # 1. Dùng nguyên path trong CSV
    candidates.append(image_path)

    # 2. Bỏ ./ nếu có
    if image_path.startswith("./"):
        path_no_dot = image_path[2:]
    else:
        path_no_dot = image_path

    # 3. Ghép với PROJECT_ROOT = D:/Nghiencuu
    candidates.append(os.path.join(PROJECT_ROOT, path_no_dot))

    # 4. Ghép với SCRIPT_DIR = D:/Nghiencuu/Model
    candidates.append(os.path.join(SCRIPT_DIR, path_no_dot))

    # 5. Nếu path đã là absolute thì thử luôn absolute path
    if os.path.isabs(image_path):
        candidates.append(image_path)

    normalized = path_no_dot.replace("\\", "/")

    # 6. Nếu path bắt đầu bằng processed_celebdf_02 thì ghép trực tiếp với PROJECT_ROOT
    if normalized.startswith("processed_celebdf_02/"):
        candidates.append(os.path.join(PROJECT_ROOT, normalized))

    # 7. Nếu path chứa processed_celebdf_02 ở giữa, cắt từ đoạn đó
    marker = "processed_celebdf_02/"
    if marker in normalized:
        idx = normalized.find(marker)
        sub_path = normalized[idx:]
        candidates.append(os.path.join(PROJECT_ROOT, sub_path))

    checked = set()

    for candidate in candidates:
        candidate = os.path.normpath(candidate)

        if candidate in checked:
            continue

        checked.add(candidate)

        if os.path.exists(candidate):
            return candidate

    return None


# =========================================================
# DATASET
# =========================================================
class FaceDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        required_cols = ["image_path", "label"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV thiếu cột bắt buộc: {col}")

        self.df["image_path"] = self.df["image_path"].astype(str)
        self.df["label"] = self.df["label"].astype(int)

        print("\n================ KIỂM TRA ĐƯỜNG DẪN ẢNH ================")
        print("SCRIPT_DIR:", SCRIPT_DIR)
        print("PROJECT_ROOT:", PROJECT_ROOT)
        print("CSV:", csv_path)

        if len(self.df) > 0:
            print("\nVí dụ image_path trong CSV:")
            print(self.df.iloc[0]["image_path"])

        resolved_paths = []
        missing_examples = []

        for path in self.df["image_path"].tolist():
            resolved = resolve_image_path(path)

            if resolved is None:
                resolved_paths.append(None)

                if len(missing_examples) < 10:
                    missing_examples.append(path)
            else:
                resolved_paths.append(resolved)

        self.df["resolved_image_path"] = resolved_paths

        before = len(self.df)
        self.df = self.df[self.df["resolved_image_path"].notna()].reset_index(drop=True)
        after = len(self.df)

        missing = before - after

        print("\nTổng ảnh trong CSV:", before)
        print("Ảnh tìm thấy:", after)
        print("Ảnh không tồn tại:", missing)

        if after > 0:
            print("\nVí dụ path đã resolve đúng:")
            print(self.df.iloc[0]["resolved_image_path"])

        if missing > 0:
            print("\nVí dụ 10 path không tìm thấy:")
            for p in missing_examples:
                print("  -", p)

        print("=========================================================\n")

        if len(self.df) == 0:
            raise ValueError(
                "Không còn ảnh hợp lệ để test. "
                "Khả năng cao là PROJECT_ROOT đang sai hoặc thư mục processed_celebdf_02 không nằm ở D:/Nghiencuu."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["resolved_image_path"]
        label = int(row["label"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            img = cv2.imread(image_path)

            if img is None:
                raise RuntimeError(f"Không đọc được ảnh: {image_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img)

        if self.transform:
            image = self.transform(image)

        return image, label, image_path


# =========================================================
# MODEL
# =========================================================
def build_model(model_name, num_classes=2, dropout_rate=0.4):
    model_name = model_name.lower()

    if model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"MODEL_NAME không hỗ trợ: {model_name}")

    return model


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    print("Đang load checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("Load từ key: model_state_dict")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("Load từ key: state_dict")
        else:
            state_dict = checkpoint
            print("Load trực tiếp checkpoint dạng state_dict")
    else:
        state_dict = checkpoint

    new_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k.replace("module.", "", 1)] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)

    print("Load model thành công.")
    return model


# =========================================================
# EVALUATE
# =========================================================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=True):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_preds = []
    all_probs_fake = []
    all_image_paths = []

    start_time = time.time()

    for batch_idx, (images, labels, image_paths) in enumerate(dataloader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(use_amp and device.type == "cuda")
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs_fake.extend(probs[:, 1].cpu().numpy().tolist())
        all_image_paths.extend(list(image_paths))

        if batch_idx % 50 == 0:
            print(f"[TIẾN ĐỘ] Batch {batch_idx}/{len(dataloader)}")

    avg_loss = total_loss / total_samples

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    elapsed = time.time() - start_time

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "labels": all_labels,
        "preds": all_preds,
        "probs_fake": all_probs_fake,
        "image_paths": all_image_paths,
        "elapsed_seconds": elapsed
    }


# =========================================================
# SAVE RESULTS
# =========================================================
def save_predictions_csv(results, output_path):
    rows = []

    for image_path, label, pred, prob_fake in zip(
        results["image_paths"],
        results["labels"],
        results["preds"],
        results["probs_fake"]
    ):
        rows.append({
            "image_path": image_path,
            "true_label": label,
            "pred_label": pred,
            "prob_real": round(1.0 - prob_fake, 6),
            "prob_fake": round(prob_fake, 6),
            "correct": int(label == pred)
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("Đã lưu predictions:")
    print(output_path)


def save_metrics_txt(results, output_path):
    cm = results["confusion_matrix"]

    report = classification_report(
        results["labels"],
        results["preds"],
        labels=[0, 1],
        target_names=["REAL_0", "FAKE_1"],
        zero_division=0
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("===== TEST MODEL ON FILTERED CSV =====\n\n")

        f.write(f"Loss:      {results['loss']:.6f}\n")
        f.write(f"Accuracy:  {results['accuracy']:.6f}\n")
        f.write(f"Precision: {results['precision']:.6f}\n")
        f.write(f"Recall:    {results['recall']:.6f}\n")
        f.write(f"F1-score:  {results['f1']:.6f}\n")
        f.write(f"Time:      {results['elapsed_seconds']:.2f} seconds\n")

        f.write("\n===== CONFUSION MATRIX =====\n")
        f.write("Rows = Actual, Columns = Predicted\n")
        f.write("           Pred_REAL_0  Pred_FAKE_1\n")
        f.write(f"Actual_REAL_0    {cm[0][0]}          {cm[0][1]}\n")
        f.write(f"Actual_FAKE_1    {cm[1][0]}          {cm[1][1]}\n")

        f.write("\n===== CLASSIFICATION REPORT =====\n")
        f.write(report)

    print("Đã lưu metrics:")
    print(output_path)


def save_confusion_matrix_csv(cm, output_path):
    df = pd.DataFrame(
        cm,
        index=["Actual_REAL_0", "Actual_FAKE_1"],
        columns=["Pred_REAL_0", "Pred_FAKE_1"]
    )

    df.to_csv(output_path, encoding="utf-8")

    print("Đã lưu confusion matrix CSV:")
    print(output_path)


# =========================================================
# VẼ CONFUSION MATRIX ĐẸP + 5 ĐỘ ĐO
# =========================================================
def plot_confusion_matrix_with_metrics(results, output_path):
    """
    Vẽ confusion matrix dạng đẹp hơn.

    Format confusion matrix:
        [[TN, FP],
         [FN, TP]]

    Vì label:
        REAL = 0
        FAKE = 1

    Nên:
        TN = Actual REAL, Pred REAL
        FP = Actual REAL, Pred FAKE
        FN = Actual FAKE, Pred REAL
        TP = Actual FAKE, Pred FAKE
    """

    cm = np.array(results["confusion_matrix"], dtype=int)

    if cm.shape != (2, 2):
        print("[LỖI] Confusion matrix phải có dạng 2x2.")
        print("Shape hiện tại:", cm.shape)
        return

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = tn + fp + fn + tp

    loss = results["loss"]
    accuracy = results["accuracy"]
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(cm, cmap="Blues")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["REAL", "FAKE"], fontsize=12)
    ax.set_yticklabels(["REAL", "FAKE"], fontsize=12)

    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("Actual label", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=16, pad=14)

    cell_labels = [
        ["TN", "FP"],
        ["FN", "TP"]
    ]

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0

    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            label = cell_labels[i][j]
            percent = safe_divide(value, total) * 100

            ax.text(
                j,
                i,
                f"{label}\n{value:,}\n{percent:.2f}%",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=13,
                fontweight="bold"
            )

    # Kẻ ô trắng ở giữa cho rõ 4 vùng
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    metrics_text = (
        "Evaluation Metrics\n"
        "------------------\n"
        f"Loss      : {loss:.4f}\n"
        f"Accuracy  : {accuracy:.4f}\n"
        f"Precision : {precision:.4f}\n"
        f"Recall    : {recall:.4f}\n"
        f"F1-score  : {f1:.4f}\n\n"
        "Counts\n"
        "------------------\n"
        f"TN: {tn:,}\n"
        f"FP: {fp:,}\n"
        f"FN: {fn:,}\n"
        f"TP: {tp:,}\n"
        f"Total: {total:,}"
    )

    # Hộp metrics bên phải
    fig.text(
        0.73,
        0.50,
        metrics_text,
        fontsize=12,
        va="center",
        ha="left",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="gray",
            alpha=0.95
        )
    )

    plt.tight_layout(rect=[0, 0, 0.70, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Đã lưu confusion matrix ảnh đẹp:")
    print(output_path)


def plot_confusion_matrix_basic(cm, output_path):
    """
    Bản cũ đơn giản.
    Hiện tại không dùng nữa, giữ lại nếu sau này anh muốn so sánh.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred REAL", "Pred FAKE"])
    plt.yticks(tick_marks, ["Actual REAL", "Actual FAKE"])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Đã lưu confusion matrix ảnh basic:")
    print(output_path)


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, default=FILTERED_CSV)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE)

    args = parser.parse_args()

    create_folder(args.output_dir)

    print("\n================ THÔNG TIN TEST ================")
    print("SCRIPT_DIR:", SCRIPT_DIR)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Model:", args.model)
    print("Checkpoint:", args.checkpoint)
    print("CSV filtered:", args.csv)
    print("Output dir:", args.output_dir)
    print("Image size:", args.image_size)
    print("Batch size:", args.batch_size)
    print("Num workers:", args.num_workers)
    print("Dropout:", args.dropout)
    print("================================================\n")

    if not os.path.exists(args.csv):
        print("[LỖI] Không tìm thấy CSV filtered:")
        print(args.csv)
        return

    if not os.path.exists(args.checkpoint):
        print("[LỖI] Không tìm thấy checkpoint:")
        print(args.checkpoint)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===== THÔNG TIN THIẾT BỊ =====")
    print("Thiết bị:", device)

    if device.type == "cuda":
        print("CUDA available:", torch.cuda.is_available())
        print("Tên GPU:", torch.cuda.get_device_name(0))
        print(f"VRAM tổng: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    print("================================\n")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = FaceDataset(
        csv_path=args.csv,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    print("Số ảnh hợp lệ để test:", len(dataset))

    label_counts = dataset.df["label"].value_counts().sort_index()
    print("REAL (0):", int(label_counts.get(0, 0)))
    print("FAKE (1):", int(label_counts.get(1, 0)))

    if "method" in dataset.df.columns:
        print("\nSố ảnh theo method:")
        for method, count in dataset.df["method"].value_counts().items():
            print(f"  - {method}: {count}")

    if "blur_score" in dataset.df.columns:
        blur_values = pd.to_numeric(dataset.df["blur_score"], errors="coerce").dropna()
        if len(blur_values) > 0:
            print("\nThống kê blur_score của data test:")
            print(f"  - Min:    {blur_values.min():.4f}")
            print(f"  - Mean:   {blur_values.mean():.4f}")
            print(f"  - Median: {blur_values.median():.4f}")
            print(f"  - Max:    {blur_values.max():.4f}")

    print()

    model = build_model(
        model_name=args.model,
        num_classes=NUM_CLASSES,
        dropout_rate=args.dropout
    )

    model = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    results = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        use_amp=USE_AMP
    )

    cm = results["confusion_matrix"]

    print("\n================ KẾT QUẢ TEST TRÊN DATA FILTERED ================")
    print(f"Loss:      {results['loss']:.6f}")
    print(f"Accuracy:  {results['accuracy']:.6f}")
    print(f"Precision: {results['precision']:.6f}")
    print(f"Recall:    {results['recall']:.6f}")
    print(f"F1-score:  {results['f1']:.6f}")

    print("\nConfusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    print("           Pred_REAL_0  Pred_FAKE_1")
    print(f"Actual_REAL_0    {cm[0][0]}          {cm[0][1]}")
    print(f"Actual_FAKE_1    {cm[1][0]}          {cm[1][1]}")

    print(f"\nThời gian test: {results['elapsed_seconds']:.2f} giây")
    print("=================================================================\n")

    predictions_csv = os.path.join(args.output_dir, "predictions_filtered.csv")
    metrics_txt = os.path.join(args.output_dir, "metrics_filtered.txt")
    cm_csv = os.path.join(args.output_dir, "confusion_matrix_filtered.csv")

    # Ảnh mới, đẹp hơn
    cm_png = os.path.join(args.output_dir, "confusion_matrix_filtered_with_metrics.png")

    save_predictions_csv(results, predictions_csv)
    save_metrics_txt(results, metrics_txt)
    save_confusion_matrix_csv(cm, cm_csv)

    # Dùng bản mới thay cho bản vẽ cũ
    plot_confusion_matrix_with_metrics(results, cm_png)

    print("\nHOÀN TẤT.")
    print("Các file kết quả nằm trong:")
    print(args.output_dir)
    print("\nẢnh ma trận nhiễu mới:")
    print(cm_png)


if __name__ == "__main__":
    main()