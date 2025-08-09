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

device = 'cuda'

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
import sys
from drone_common import plot_image_with_boxes_and_probs

if __name__=="__main__":
    # read first argument which is folder
    if len(sys.argv) > 1:
        impath = sys.argv[1]  # sys.argv[0] is the script name
        im = Image.open(impath)
        detection = detector.detect(im)
        plot_image_with_boxes_and_probs(im,detection)
    else:
        print("Provide path to file")
