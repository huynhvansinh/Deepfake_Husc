import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "./training_outputs/densenet121_ffpp_optimized_v2/training_history.csv"
OUTPUT_DIR = "./training_outputs/densenet121_ffpp_optimized_v2/plots"


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot_metric(df, x_col, y_cols, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))

    for col in y_cols:
        if col not in df.columns:
            print(f"[CẢNH BÁO] Không có cột: {col}")
            continue
        plt.plot(df[x_col], df[col], marker="o", label=col)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[LỖI] Không tìm thấy file: {CSV_PATH}")
        return

    create_folder(OUTPUT_DIR)

    df = pd.read_csv(CSV_PATH)

    if "epoch" not in df.columns:
        print("[LỖI] File CSV không có cột 'epoch'")
        return

    # Loss
    plot_metric(
        df=df,
        x_col="epoch",
        y_cols=["train_loss", "val_loss"],
        title="Loss theo Epoch - DenseNet121 Optimized v2",
        ylabel="Loss",
        save_path=os.path.join(OUTPUT_DIR, "loss_plot.png")
    )

    # Accuracy
    plot_metric(
        df=df,
        x_col="epoch",
        y_cols=["train_acc", "val_acc"],
        title="Accuracy theo Epoch - DenseNet121 Optimized v2",
        ylabel="Accuracy",
        save_path=os.path.join(OUTPUT_DIR, "accuracy_plot.png")
    )

    # F1-score
    plot_metric(
        df=df,
        x_col="epoch",
        y_cols=["train_f1", "val_f1"],
        title="F1-score theo Epoch - DenseNet121 Optimized v2",
        ylabel="F1-score",
        save_path=os.path.join(OUTPUT_DIR, "f1_plot.png")
    )

    # Precision
    plot_metric(
        df=df,
        x_col="epoch",
        y_cols=["train_precision", "val_precision"],
        title="Precision theo Epoch - DenseNet121 Optimized v2",
        ylabel="Precision",
        save_path=os.path.join(OUTPUT_DIR, "precision_plot.png")
    )

    # Recall
    plot_metric(
        df=df,
        x_col="epoch",
        y_cols=["train_recall", "val_recall"],
        title="Recall theo Epoch - DenseNet121 Optimized v2",
        ylabel="Recall",
        save_path=os.path.join(OUTPUT_DIR, "recall_plot.png")
    )

    # Learning rate
    if "lr" in df.columns:
        plot_metric(
            df=df,
            x_col="epoch",
            y_cols=["lr"],
            title="Learning Rate theo Epoch - DenseNet121 Optimized v2",
            ylabel="Learning Rate",
            save_path=os.path.join(OUTPUT_DIR, "lr_plot.png")
        )

    # Time per epoch
    if "epoch_minutes" in df.columns:
        plot_metric(
            df=df,
            x_col="epoch",
            y_cols=["epoch_minutes"],
            title="Thời gian mỗi Epoch - DenseNet121 Optimized v2",
            ylabel="Minutes",
            save_path=os.path.join(OUTPUT_DIR, "epoch_time_plot.png")
        )

    print("Đã vẽ xong biểu đồ.")
    print("Thư mục lưu ảnh:", OUTPUT_DIR)


if __name__ == "__main__":
    main()