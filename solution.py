import torch
from typing import List, Union
import numpy as np

segmentation_model = 'vlad/seg-drone.pt'
yolo_model = "andry/best2.pt"
classification="egor/best2.pth"

pad_to_multiple=128
padding_value=0.5

# трешхолд для сегментации
seg_threshold = 0.5

# трешхолд для объектов после сегментации
prob_threshold=0.6

# трешхолд для первичной классификации боксов моделью егора
classification_threshold = 0.8

# размер тайла, который вырезается для области предсказанной моделькой сегментации
tile_size=512

# нужно ли логить промежуточные результаты
log = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# макс расстояние между пикселями в одном кластере
dbscan_eps = 4

# мин кол-во пикселей в кластере
dbscan_min_samples = 2

# трешхолд для копий детекций
copies_threshold=1e-3

from inference_code import DroneObjectDetector
detector = DroneObjectDetector(
    segmentation_model,
    yolo_model,
    classification,
    pad_to_multiple,
    padding_value,
    seg_threshold,
    prob_threshold,
    classification_threshold,
    tile_size,
    log,
    device,
    dbscan_eps,
    dbscan_min_samples,
    copies_threshold
)

import PIL.Image as Image

def model_predict_one_image(image: np.ndarray) -> list:
    """
    Выполняет предсказание модели для одного изображения.

    Args:
        image (np.ndarray): Изображение в формате RGB (H, W, C).

    Returns:
        list: Список словарей с результатами предсказания.
    """
    # Преобразование изображения из numpy в PIL и применение трансформаций
    im = Image.fromarray(image)

    # Инференс модели
    detections = detector.detect(im)

    # Обработка результатов
    image_results = []
    for detection in detections:
        class_id, prob, xc, yc, w, h = detection
        image_results.append({
            'xc': xc,
            'yc': yc,
            'w': w,
            'h': h,
            'label': class_id,
            'score': round(prob.item(), 4),
        })

    return image_results

def predict(images: Union[List[np.ndarray], np.ndarray]) -> list:
    """
    Выполняет инференс модели на одном или нескольких изображениях.

    Args:
        images (Union[List[np.ndarray], np.ndarray]): Изображение или список изображений
            в формате RGB (H, W, C).

    Returns:
        list: Список списков словарей с результатами предсказания для каждого изображения.
        [
            [
                {
                    'xc': float,  # Центр по x (нормализованный)
                    'yc': float,  # Центр по y (нормализованный)
                    'w': float,   # Ширина (нормализованная)
                    'h': float,   # Высота (нормализованная)
                    'label': int, # Метка класса (0 для человека)
                    'score': float, # Вероятность
                },
                ...
            ],
            ...
        ]
    """
    results = []
    if isinstance(images, np.ndarray):
        images = [images]

    for image in images:
        image_results = model_predict_one_image(image)
        results.append(image_results)

    return results
