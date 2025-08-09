import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import List

def yolo8_to_gt_csv(labels_dir: str, images_dir: str, output_csv: str = 'public_gt_solution_24-10-24.csv'):
    gt_data = []

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue

        base_name = os.path.splitext(label_file)[0]

        # Найти изображение с любым распространённым расширением (учитываем регистр)
        possible_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_path = None

        for ext in possible_exts:
            candidate = os.path.join(images_dir, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"[!] Пропущено: {base_name} — нет изображения.")
            continue

        with Image.open(image_path) as img:
            w_img, h_img = img.size

        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                label, xc, yc, w, h = map(float, parts)

                gt_data.append({
                    'image_id': os.path.basename(image_path),
                    'label': int(label),
                    'xc': xc,
                    'yc': yc,
                    'w': w,
                    'h': h,
                    'w_img': w_img,
                    'h_img': h_img,
                    'score': 1.0,
                    'time_spent': 0.0
                })

    df = pd.DataFrame(gt_data)
    df.to_csv(output_csv, index=False)
    print(f"[✓] GT-файл сохранён: {output_csv}")

def load_images_from_folder(folder_path: str, extensions={'.jpg', '.jpeg', '.png'}):
    image_paths = []
    images = []

    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in extensions:
            full_path = os.path.join(folder_path, fname)
            try:
                img = Image.open(full_path).convert('RGB')
                images.append(np.array(img))
                image_paths.append(full_path)
            except Exception as e:
                print(f"Ошибка при загрузке {fname}: {e}")

    return images, image_paths

def save_predictions_to_csv(predictions: List[List[dict]], image_paths: List[str], processing_times: List[float], output_csv_path: str):
    import csv
    from PIL import Image

    csv_rows = []
    csv_header = ['image_id', 'xc', 'yc', 'w', 'h', 'label', 'score', 'time_spent', 'w_img', 'h_img']

    for preds, img_path, time_spent in zip(predictions, image_paths, processing_times):
        image = Image.open(img_path)
        w_img, h_img = image.size

        image_id = os.path.basename(img_path)  # Пример: "1.jpg"

        for pred in preds:
            csv_rows.append([
                image_id,
                pred['xc'],
                pred['yc'],
                pred['w'],
                pred['h'],
                pred['label'],
                pred['score'],
                time_spent,
                w_img,
                h_img
            ])

    with open(output_csv_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)

    print(f"Результаты сохранены в: {output_csv_path}")


