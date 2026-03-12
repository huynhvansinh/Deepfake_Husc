import os
import csv
import cv2
import shutil


# =========================================================
# CẤU HÌNH
# =========================================================
INPUT_CSV = "./processed_ffpp/ffpp_faces.csv"
OUTPUT_ROOT = "./processed_ffpp"

# Thư mục chứa ảnh crop đã có
OLD_CROPFACES_ROOT = os.path.join(OUTPUT_ROOT, "cropfaces")

# Thư mục mới chỉ chứa ảnh không mờ
NEW_CROPFACES_ROOT = os.path.join(OUTPUT_ROOT, "cropfaces_filtered")

# CSV mới sau khi lọc
OUTPUT_CSV = os.path.join(OUTPUT_ROOT, "ffpp_faces_filtered.csv")

# CSV báo cáo blur score của tất cả ảnh
BLUR_REPORT_CSV = os.path.join(OUTPUT_ROOT, "blur_report.csv")

# Ngưỡng blur
BLUR_THRESHOLD = 50.0


# =========================================================
# HÀM PHỤ
# =========================================================
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(score)


def is_blurry(image, threshold=80.0):
    score = get_blur_score(image)
    return score < threshold, score


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"[LỖI] Không tìm thấy file CSV: {INPUT_CSV}")
        return

    create_folder(NEW_CROPFACES_ROOT)

    kept_rows = []
    blur_rows = []

    total_images = 0
    kept_images = 0
    blurry_images = 0
    missing_images = 0
    read_error_images = 0

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Tìm thấy {len(rows)} dòng trong CSV.")
    print(f"Bắt đầu lọc ảnh mờ với ngưỡng BLUR_THRESHOLD = {BLUR_THRESHOLD}")

    for idx, row in enumerate(rows, 1):
        image_path = row["image_path"]

        total_images += 1

        if not os.path.exists(image_path):
            missing_images += 1
            blur_rows.append({
                "image_path": image_path.replace("\\", "/"),
                "blur_score": "",
                "status": "missing_file"
            })
            continue

        img = cv2.imread(image_path)
        if img is None:
            read_error_images += 1
            blur_rows.append({
                "image_path": image_path.replace("\\", "/"),
                "blur_score": "",
                "status": "read_error"
            })
            continue

        blurry, blur_score = is_blurry(img, threshold=BLUR_THRESHOLD)

        blur_rows.append({
            "image_path": image_path.replace("\\", "/"),
            "blur_score": round(blur_score, 4),
            "status": "blurry" if blurry else "kept"
        })

        if blurry:
            blurry_images += 1
            continue

        # Tạo đường dẫn mới trong cropfaces_filtered
        rel_path = os.path.relpath(image_path, OLD_CROPFACES_ROOT)
        new_image_path = os.path.join(NEW_CROPFACES_ROOT, rel_path)

        create_folder(os.path.dirname(new_image_path))
        shutil.copy2(image_path, new_image_path)

        new_row = dict(row)
        new_row["image_path"] = new_image_path.replace("\\", "/")
        new_row["blur_score"] = round(blur_score, 4)

        kept_rows.append(new_row)
        kept_images += 1

        if idx % 1000 == 0:
            print(f"Đã xử lý {idx}/{len(rows)} ảnh...")

    # Ghi CSV mới chỉ gồm ảnh giữ lại
    if kept_rows:
        fieldnames = list(kept_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
    else:
        print("[CẢNH BÁO] Không có ảnh nào được giữ lại.")
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_path",
                "label",
                "method",
                "video_name",
                "source_video_path",
                "relative_video_path",
                "split_group",
                "blur_score"
            ])

    # Ghi CSV báo cáo blur cho toàn bộ ảnh
    with open(BLUR_REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "blur_score", "status"])
        writer.writeheader()
        writer.writerows(blur_rows)

    print("\n================ KẾT QUẢ LỌC ẢNH MỜ ================")
    print("Tổng số ảnh trong CSV:", total_images)
    print("Số ảnh giữ lại:", kept_images)
    print("Số ảnh bị loại do mờ:", blurry_images)
    print("Số ảnh thiếu file:", missing_images)
    print("Số ảnh lỗi đọc:", read_error_images)
    print(f"Ngưỡng blur: {BLUR_THRESHOLD}")
    print(f"Ảnh sau lọc lưu tại: {NEW_CROPFACES_ROOT}")
    print(f"CSV sạch lưu tại: {OUTPUT_CSV}")
    print(f"Báo cáo blur lưu tại: {BLUR_REPORT_CSV}")
    print("====================================================")


if __name__ == "__main__":
    main()