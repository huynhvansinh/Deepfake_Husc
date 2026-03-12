import os
import random
import pandas as pd


# =========================================================
# CẤU HÌNH
# =========================================================
INPUT_CSV = "./processed_ffpp/ffpp_faces_filtered.csv"
OUTPUT_DIR = "./processed_ffpp/splits"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, \
    "Tổng TRAIN_RATIO + VAL_RATIO + TEST_RATIO phải bằng 1.0"


# =========================================================
# HÀM PHỤ
# =========================================================
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_split_stats(name, df):
    print(f"\n===== {name.upper()} =====")
    print(f"Số ảnh: {len(df)}")

    if "label" in df.columns:
        label_counts = df["label"].value_counts(dropna=False).sort_index()
        real_count = int(label_counts.get(0, 0))
        fake_count = int(label_counts.get(1, 0))
        print(f"REAL (0): {real_count}")
        print(f"FAKE (1): {fake_count}")

    if "split_group" in df.columns:
        print(f"Số video (split_group): {df['split_group'].nunique()}")

    if "method" in df.columns:
        print("Số ảnh theo method:")
        method_counts = df["method"].value_counts()
        for method, count in method_counts.items():
            print(f"  - {method}: {count}")


# =========================================================
# CHIA THEO VIDEO
# =========================================================
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"[LỖI] Không tìm thấy file: {INPUT_CSV}")
        return

    create_folder(OUTPUT_DIR)

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["image_path", "label", "method", "split_group"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[LỖI] CSV thiếu cột bắt buộc: {col}")
            return

    print("Đã đọc CSV thành công.")
    print(f"Tổng số ảnh: {len(df)}")
    print(f"Tổng số video (split_group): {df['split_group'].nunique()}")

    # -----------------------------------------------------
    # Kiểm tra mỗi split_group chỉ có 1 label duy nhất
    # -----------------------------------------------------
    group_label_check = df.groupby("split_group")["label"].nunique()
    bad_groups = group_label_check[group_label_check > 1]

    if len(bad_groups) > 0:
        print("[LỖI] Có split_group chứa nhiều label khác nhau.")
        print(bad_groups.head(10))
        return

    # -----------------------------------------------------
    # Tạo bảng video-level
    # Mỗi split_group đại diện cho 1 video
    # -----------------------------------------------------
    video_df = (
        df.groupby("split_group", as_index=False)
        .agg({
            "label": "first",
            "method": "first"
        })
    )

    print(f"\nSố video hợp lệ để chia: {len(video_df)}")

    # -----------------------------------------------------
    # Chia riêng theo label để giảm lệch real/fake
    # -----------------------------------------------------
    random.seed(RANDOM_SEED)

    real_groups = video_df[video_df["label"] == 0]["split_group"].tolist()
    fake_groups = video_df[video_df["label"] == 1]["split_group"].tolist()

    random.shuffle(real_groups)
    random.shuffle(fake_groups)

    def split_groups(groups, train_ratio, val_ratio):
        n = len(groups)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_g = groups[:train_end]
        val_g = groups[train_end:val_end]
        test_g = groups[val_end:]

        return train_g, val_g, test_g

    train_real, val_real, test_real = split_groups(real_groups, TRAIN_RATIO, VAL_RATIO)
    train_fake, val_fake, test_fake = split_groups(fake_groups, TRAIN_RATIO, VAL_RATIO)

    train_groups = set(train_real + train_fake)
    val_groups = set(val_real + val_fake)
    test_groups = set(test_real + test_fake)

    # -----------------------------------------------------
    # Lọc ảnh theo split_group
    # -----------------------------------------------------
    train_df = df[df["split_group"].isin(train_groups)].copy()
    val_df = df[df["split_group"].isin(val_groups)].copy()
    test_df = df[df["split_group"].isin(test_groups)].copy()

    # -----------------------------------------------------
    # Lưu file CSV
    # -----------------------------------------------------
    train_csv = os.path.join(OUTPUT_DIR, "train.csv")
    val_csv = os.path.join(OUTPUT_DIR, "val.csv")
    test_csv = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")
    test_df.to_csv(test_csv, index=False, encoding="utf-8")

    # -----------------------------------------------------
    # In thống kê
    # -----------------------------------------------------
    print("\n================ KẾT QUẢ CHIA TẬP ================")
    print(f"Tổng số video REAL: {len(real_groups)}")
    print(f"Tổng số video FAKE: {len(fake_groups)}")

    print(f"\nTrain groups: {len(train_groups)}")
    print(f"Val groups:   {len(val_groups)}")
    print(f"Test groups:  {len(test_groups)}")

    print_split_stats("train", train_df)
    print_split_stats("val", val_df)
    print_split_stats("test", test_df)

    print("\nĐã lưu file:")
    print(train_csv)
    print(val_csv)
    print(test_csv)
    print("==================================================")


if __name__ == "__main__":
    main()