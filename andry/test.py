from ultralytics import YOLO

# Загружаем веса
model = YOLO("/home/bowie/repos/ml/persons_detection/runs/detect/train12/weights/best.pt")  # путь к твоему .pt файлу

# Запуск на изображении
results = model("/mnt/datadisk/Dataset/dataset/yolo_sliced/images/val/1_000165_0_7.JPG", save=True)  # save=True сохранит картинку с боксами в runs/detect/predict

# Если хочешь вывести детекции в консоль
for r in results:
    boxes = r.boxes.xyxy  # координаты боксов [x1, y1, x2, y2]
    confs = r.boxes.conf  # уверенности
    clss  = r.boxes.cls   # ID классов
    print(boxes, confs, clss)