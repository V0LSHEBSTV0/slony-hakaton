import torch
from ultralytics import YOLO
from egor.model import *
import torchvision.transforms as T
from drone_common import *
from kemsekov_torch.utils import PadToMultiple
import numpy as np
from sklearn.cluster import DBSCAN


class DroneObjectDetector:
    def __init__(
        self,
        segmentation_model='vlad/seg-drone.pt',
        yolo_model="andry/best.pt",
        classification="egor/best_false_positive_no_resize.pth",
        pad_to_multiple=128,
        padding_value=0.5,
        seg_threshold=0.5,
        prob_threshold=0.6,
        classification_threshold=0.8,
        tile_size=512,
        log=True,
        device='cuda',
        dbscan_eps=4,
        dbscan_min_samples=2,
        copies_threshold=1e-3
    ):
        self.segmentation_model_path = segmentation_model
        self.yolo_model_path = yolo_model
        self.classification_model_path = classification

        self.pad_to_multiple = pad_to_multiple
        self.padding_value = padding_value
        self.seg_threshold = seg_threshold
        self.prob_threshold = prob_threshold
        self.classification_threshold = classification_threshold
        self.tile_size = tile_size
        self.log = log
        self.device = device
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.copies_threshold = copies_threshold

        # Load models
        self.model = torch.jit.load(self.segmentation_model_path).eval().to(self.device)
        self.classifier = CustomResNetClassifier(self.classification_model_path, pretrained=False).eval().to(self.device)
        self.yolo = YOLO(self.yolo_model_path).to(self.device)

        # Define transforms
        self.im_transform = T.Compose([
            T.Lambda(self.to_tensor_if_pil),
            PadToMultiple(self.pad_to_multiple, self.padding_value)
        ])

    @staticmethod
    def to_tensor_if_pil(x):
        if isinstance(x, torch.Tensor):
            return x
        return T.ToTensor()(x)

    def get_objects(self, out):
        outw, outh = out.shape[-2:]
        _, x, y = torch.where(out > self.seg_threshold)
        xy = torch.stack([x, y], -1)
        if len(xy) == 0:
            return []

        labels = DBSCAN(self.dbscan_eps, min_samples=self.dbscan_min_samples).fit_predict(xy)
        centroinds = []
        for obj in np.unique(labels):
            xy_obj = xy[labels == obj]
            prob = out[:, xy_obj[:, 0], xy_obj[:, 1]].mean().item()
            if prob < self.prob_threshold:
                continue
            xy_obj = 1.0 * xy_obj
            xy_obj[:, 0] /= outw
            xy_obj[:, 1] /= outh
            xy_obj[:, [0, 1]] = xy_obj[:, [1, 0]]
            centroid_min = xy_obj.min(0)[0]
            centroid_max = xy_obj.max(0)[0]
            size = (centroid_max - centroid_min)
            size[0] += 1 / outh
            size[1] += 1 / outw
            centroinds.append((0, prob, *xy_obj.mean(0).tolist(), *size.tolist()))
        return centroinds

    def filter_detections(self, im, detection):
        max_w = im.shape[-2]
        max_h = im.shape[-1]
        images = []
        images_pos = []

        for d in detection:
            x, y = d[2:4]
            w = int(y * max_w)
            h = int(x * max_h)

            w_left = max(w - self.tile_size // 2, 0)
            w_right = min(w_left + self.tile_size, max_w)
            w_left = max(w_right - self.tile_size, 0)

            h_left = max(h - self.tile_size // 2, 0)
            h_right = min(h_left + self.tile_size, max_h)
            h_left = max(h_right - self.tile_size, 0)

            cut = im[:, w_left:w_right, h_left:h_right]
            images.append(cut)
            images_pos.append((w_left, h_left))

        images = torch.stack(images).to(self.device)

        with torch.no_grad():
            cls = self.classifier(images).softmax(-1)[:, 1].cpu()
            if self.log:
                print(cls)

            detection = [d for c, d in zip(cls, detection) if c < self.classification_threshold]
            images_pos = [i for c, i in zip(cls, images_pos) if c < self.classification_threshold]
            images = [i for c,i in zip(cls,images) if c<self.classification_threshold]
            if len(images)>0:
                images=torch.stack(images)
                if self.log:
                    print(f"Running yolo at {len(images)} detections")
                boxes = self.yolo(images, verbose=False)
                boxes_count = [len(b) for b in boxes]
                if self.log:
                    print(boxes_count)

                detection = [d for d, b in zip(detection, boxes_count) if (b > 0)]
                images_pos = [i for i, b in zip(images_pos, boxes_count) if (b > 0)]
                boxes = [b for b in boxes if len(b) > 0]

        images_pos = [(w / max_w, h / max_h) for w, h in images_pos]
        new_detection = []
        if len(images)>0:
            for pos_xy, r in zip(images_pos, boxes):
                for xyxy, conf in zip(r.boxes.xyxy, r.boxes.conf):
                    x1, y1, x2, y2 = xyxy.cpu()
                    y1 = y1 / max_w + pos_xy[0]
                    y2 = y2 / max_w + pos_xy[0]
                    x1 = x1 / max_h + pos_xy[1]
                    x2 = x2 / max_h + pos_xy[1]
                    new_detection.append([0, conf.cpu(), (x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1])

        return new_detection

    @staticmethod
    def l2_distance_matrix(A, B):
        A_sq = np.sum(A ** 2, axis=1).reshape(-1, 1)
        B_sq = np.sum(B ** 2, axis=1).reshape(1, -1)
        dist_sq = A_sq + B_sq - 2 * A @ B.T
        dist_sq = np.clip(dist_sq, 0, None)
        return np.sqrt(dist_sq)

    def remove_copies(self, detection):
        if len(detection)==0: return detection
        cords = np.array(detection)[:, [2, 3]]
        dist = self.l2_distance_matrix(cords, cords)
        dist[dist <= self.copies_threshold] = 0
        np.fill_diagonal(dist, 1)
        added_rows = []
        skip_rows = []
        for i, row in enumerate(dist):
            if i in skip_rows:
                continue
            added_rows.append(i)
            skip_rows.extend(np.where(row == 0)[0])
        return [detection[i] for i in added_rows]

    def detect(self, image, true_detection=None):
        im = self.im_transform(image)
        with torch.no_grad():
            out = self.model(im[None, :].to(self.device))[0]
            out = out.sigmoid().cpu()

        detection = self.get_objects(out)
        if len(detection) == 0:
            return []

        if true_detection is not None and len(true_detection) > 0:
            a = np.array(true_detection)[:, [1, 2]]
            b = np.array(detection)[:, [2, 3]]
            dist = self.l2_distance_matrix(a, b).min(0)
            for i in range(len(dist)):
                det, d_dist = detection[i], dist[i]
                if d_dist < 0.05:
                    det = list(det)
                    det[0] = 1
                detection[i] = det

        if self.log:
            print("detections after segmentation", len(detection))

        detection = self.filter_detections(im, detection)

        if self.log:
            print("detections after filtering", len(detection))

        detection = [d for d in detection if d[1] > self.prob_threshold]
        if self.log:
            print("detections after thresholding boxes", len(detection))

        detection = self.remove_copies(detection)
        if self.log:
            print("detections after removing copies", len(detection))

        # Fix coordinate scaling
        if isinstance(image, torch.Tensor):
            old_shape = image.shape[-2:]
        else:
            old_shape = image.size
        new_shape = im.shape[-2:]
        scale_x = old_shape[1] / new_shape[0]
        scale_y = old_shape[0] / new_shape[1]

        for i, d in enumerate(detection):
            d = list(d)
            d[2] /= scale_y
            d[3] /= scale_x
            d[4] /= scale_y
            d[5] /= scale_x
            detection[i] = d

        return detection
