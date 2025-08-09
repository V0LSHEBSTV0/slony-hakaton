import os
import PIL.Image as Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def identity(x): return x

class DroneDataset(Dataset):
    def __init__(self,images,labels,image_transform = identity):
        super().__init__()
        
        self.ims = {v.split('.')[0]:os.path.join(images,v) for v in os.listdir(images)}
        self.labels = {v.split('.')[0]:self.extract_label(labels,v) for v in os.listdir(labels)}
        self.keys = list(set(self.ims.keys()).intersection(set(self.labels.keys())))
        self.image_transform=image_transform

    def extract_label(self, labels,v):
        lines =  open(os.path.join(labels,v)).readlines()
        lines = [l.replace('\n','').split(' ') for l in lines]
        def line_to_arr(x):
            x=[v for v in x if v!='']
            return [float(v) for v in x]
        lines = [line_to_arr(line) for line in lines]
        return lines
    
    def __getitem__(self, index):
        key = self.keys[index]
        im = Image.open(self.ims[key])
        return self.image_transform(im),self.labels[key]
    
    def __len__(self):
        return len(self.keys)
def detection_to_segmentation_mask(image_size, labels, num_classes=None):
    """
    Convert detection bounding boxes to segmentation mask.

    Parameters:
    - image_size: Tuple (height, width) of output mask
    - labels: Tensor or list of shape (N, 5), where each row is (class_id, x, y, width, height)
              in normalized coordinates [0, 1]
    - num_classes: Optional number of classes. If None, max class_id in labels + 1 will be used.

    Returns:
    - mask: torch.Tensor of shape (num_classes, height, width)
    """
    height, width = image_size
    labels = torch.tensor(labels)
    if labels.numel()==0:
        return torch.zeros((1, height, width), dtype=torch.uint8)
    
    class_ids = labels[:, 0].long()
    x = labels[:, 1] * width
    y = labels[:, 2] * height
    w = labels[:, 3] * width
    h = labels[:, 4] * height

    x1 = (x - w / 2).clamp(0, width - 1).long()
    y1 = (y - h / 2).clamp(0, height - 1).long()
    x2 = (x + w / 2).clamp(0, width - 1).long()
    y2 = (y + h / 2).clamp(0, height - 1).long()

    if num_classes is None:
        num_classes = int(class_ids.max().item()) + 1

    mask = torch.zeros((num_classes, height, width), dtype=torch.uint8)

    for i in range(labels.size(0)):
        cls = class_ids[i]
        mask[cls, y1[i]:y2[i]+1, x1[i]:x2[i]+1] = 1

    return mask

class SegmentationDrone(Dataset):
    def __init__(self,dataset,transform=identity):
        super().__init__()
        self.dataset=dataset
        self.transform=transform
        
    def __getitem__(self, index):
        v = self.dataset[index]
        im,label = self.dataset[index]
        mask = detection_to_segmentation_mask(im.shape[1:],label,1)
        return self.transform_im_mask(im,mask)
    
    def transform_im_mask(self,im,mask):
        res = self.transform(torch.concat([im,mask],0))
        im = res[:len(im)]
        mask = res[len(im):]
        return im,mask
        
    def __len__(self):
        return len(self.dataset)
    

def plot_image_with_boxes(image, annotations):
    """
    Plot a PIL image with bounding boxes based on annotations.
    
    Args:
        image: PIL.Image object (RGB)
        annotations: List of [class_id, x, y, width, height] where coordinates are normalized [0, 1]
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display the image
    if isinstance(image,torch.Tensor):
        image = T.ToPILImage()(image)
    ax.imshow(image)
    
    # Get image dimensions
    img_width, img_height = image.size  # (2736, 1824)
    
    # Plot each bounding box
    for ann in annotations:
        if len(ann)==0:
            print("no annotations")
            break
        class_id, x, y, width, height = ann
        
        # Convert normalized coordinates to pixel values
        x_pixel = x * img_width
        y_pixel = y * img_height
        width_pixel = width * img_width
        height_pixel = height * img_height
        
        # Calculate top-left corner of the box
        x_top_left = x_pixel - (width_pixel / 2)
        y_top_left = y_pixel - (height_pixel / 2)
        
        # Create a rectangle patch
        rect = patches.Rectangle(
            (x_top_left, y_top_left),
            width_pixel,
            height_pixel,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        # Add class ID label near the box
        # ax.text(
        #     x_top_left,
        #     y_top_left - 5,
        #     f'Class {int(class_id)}',
        #     color='r',
        #     fontsize=10,
        #     bbox=dict(facecolor='white', alpha=0.5)
        # )
    
    # Remove axis ticks for cleaner visualization
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_image_with_boxes_and_probs(image, annotations):
    """
    Plot a PIL image with bounding boxes and probabilities based on annotations.

    Args:
        image: PIL.Image object or Torch Tensor (RGB)
        annotations: List of [class_id, prob, x, y, width, height] 
                     where coordinates are normalized [0, 1]
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 20))

    # Convert torch.Tensor to PIL.Image if necessary
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)
    
    # Show the image
    ax.imshow(image)

    # Get image dimensions
    img_width, img_height = image.size

    # Plot each bounding box with probability
    for ann in annotations:
        if len(ann) == 0:
            print("no annotations")
            break

        class_id, prob, x, y, width, height = ann

        # Convert normalized coordinates to pixel values
        x_pixel = x * img_width
        y_pixel = y * img_height
        width_pixel = width * img_width
        height_pixel = height * img_height

        # Calculate top-left corner of the box
        x_top_left = x_pixel - (width_pixel / 2)
        y_top_left = y_pixel - (height_pixel / 2)

        # Create a rectangle patch
        rect = patches.Rectangle(
            (x_top_left, y_top_left),
            width_pixel,
            height_pixel,
            linewidth=1,
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add class ID and probability label
        ax.text(
            x_top_left,
            y_top_left,
            f'Class {int(class_id)}: {prob:.2f}',
            color='black',
            fontsize=7,
            bbox=dict(facecolor='lime', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
        )

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Finalize and show
    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()