import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "./training_outputs/efficientnet_b4_ffpp_celebdf_train_test_ffpp_sam_adamw_dropout_50epoch/training_history.csv"
OUTPUT_DIR = "./training_outputs/efficientnet_b4_ffpp_celebdf_train_test_ffpp_sam_adamw_dropout_50epoch/plots"

CONFUSION_MATRIX_CSV = "./training_outputs/efficientnet_b4_ffpp_celebdf_train_test_ffpp_sam_adamw_dropout_50epoch/confusion_matrix_test.csv"

# Nếu muốn nhập tay thì sửa ở đây:
# MANUAL_CONFUSION_MATRIX = [
#     [TN, FP],
#     [FN, TP]
# ]
MANUAL_CONFUSION_MATRIX = []


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_metric(df, x_col, y_cols, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))

    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], marker="o", linewidth=2, label=col)
        else:
            print(f"[CẢNH BÁO] Không tìm thấy cột: {col}")

    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, title, save_path):
    cm = np.array(cm, dtype=int)

    if cm.shape != (2, 2):
        print("[LỖI] Confusion matrix phải có dạng 2x2.")
        print("Shape hiện tại:", cm.shape)
        return

    total = cm.sum()
    if total == 0:
        print("[LỖI] Tổng confusion matrix bằng 0, không thể vẽ.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Màu giống kiểu mẫu
    im = ax.imshow(cm, cmap="Blues")

    # Thanh màu bên phải
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    # Nhãn trục
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["REAL", "FAKE"], fontsize=11)
    ax.set_yticklabels(["REAL", "FAKE"], fontsize=11)

    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("Actual label", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)

    # 4 loại: TN, FP, FN, TP
    cell_labels = [
        ["TN", "FP"],
        ["FN", "TP"]
    ]

    threshold = cm.max() / 2.0

    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            label = cell_labels[i][j]

            ax.text(
                j, i,
                f"{label}\n{value}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=12,
                fontweight="bold"
            )

    # Kẻ ô cho đẹp hơn
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Đã lưu confusion matrix tại: {save_path}")


def load_confusion_matrix():
    # Ưu tiên đọc từ file
    if os.path.exists(CONFUSION_MATRIX_CSV):
        print(f"Đang đọc confusion matrix từ file: {CONFUSION_MATRIX_CSV}")

        try:
            # File của anh có dạng:
            # ,Pred_REAL_0,Pred_FAKE_1
            # Actual_REAL_0,3352,147
            # Actual_FAKE_1,213,6834
            cm_df = pd.read_csv(CONFUSION_MATRIX_CSV, index_col=0)

            print("Nội dung confusion matrix đọc được:")
            print(cm_df)

            cm_df = cm_df.apply(pd.to_numeric, errors="coerce")

            if cm_df.isnull().values.any():
                print("[LỖI] File confusion matrix có giá trị không phải số.")
                print(cm_df)
                return None

            cm = cm_df.values.astype(int)

            if cm.shape != (2, 2):
                print("[LỖI] Confusion matrix đọc từ file không phải dạng 2x2.")
                print("Shape hiện tại:", cm.shape)
                print(cm_df)
                return None

            return cm

        except Exception as e:
            print("[LỖI] Không thể đọc file confusion matrix.")
            print("Chi tiết lỗi:", e)
            return None

    print("[THÔNG BÁO] Không tìm thấy file confusion matrix.")
    print(f"Đường dẫn đang kiểm tra: {CONFUSION_MATRIX_CSV}")

    if MANUAL_CONFUSION_MATRIX == []:
        print("[THÔNG BÁO] Không vẽ confusion matrix vì MANUAL_CONFUSION_MATRIX đang rỗng.")
        print("Nếu muốn vẽ tay, hãy sửa:")
        print("MANUAL_CONFUSION_MATRIX = [[TN, FP], [FN, TP]]")
        return None

    print("[THÔNG BÁO] Đang dùng confusion matrix nhập tay.")
    return np.array(MANUAL_CONFUSION_MATRIX, dtype=int)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[LỖI] Không tìm thấy file training history: {CSV_PATH}")
        return

    create_folder(OUTPUT_DIR)

    df = pd.read_csv(CSV_PATH)

    plot_metric(
        df=df,
        x_col="epoch",
        y_cols=["train_loss", "val_loss"],
        title="Loss theo Epoch - EfficientNet B4 FF++",
        ylabel="Loss",
        save_path=os.path.join(OUTPUT_DIR, "loss_plot.png")
    )

    plot_metric(
        df=df,
        x_col="epoch",
        y_cols=["train_acc", "val_acc"],
        title="Accuracy theo Epoch - EfficientNet B4 FF++",
        ylabel="Accuracy",
        save_path=os.path.join(OUTPUT_DIR, "accuracy_plot.png")
    )

    plot_metric(
        df=df,
        x_col="epoch",
        y_cols=["train_f1", "val_f1"],
        title="F1-score theo Epoch - EfficientNet B4 FF++",
        ylabel="F1-score",
        save_path=os.path.join(OUTPUT_DIR, "f1_plot.png")
    )

    plot_metric(
        df=df,
        x_col="epoch",
        y_cols=["train_precision", "val_precision"],
        title="Precision theo Epoch - EfficientNet B4 FF++",
        ylabel="Precision",
        save_path=os.path.join(OUTPUT_DIR, "precision_plot.png")
    )

    plot_metric(
        df=df,
        x_col="epoch",
        y_cols=["train_recall", "val_recall"],
        title="Recall theo Epoch - EfficientNet B4 FF++",
        ylabel="Recall",
        save_path=os.path.join(OUTPUT_DIR, "recall_plot.png")
    )

    cm = load_confusion_matrix()

    if cm is not None:
        plot_confusion_matrix(
            cm=cm,
            title="Confusion Matrix - EfficientNet B4 FF++",
            save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        )

    print("Đã vẽ xong các biểu đồ training history.")
    print("Thư mục lưu ảnh:", OUTPUT_DIR)


if __name__ == "__main__":
    main()