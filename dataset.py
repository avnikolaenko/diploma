import os

import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO


class DefectDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        """
          A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
        """

        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = [elem['category_id']+1 for elem in coco_annotation]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Annotation is in dictionary format
        annotation = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, annotation

    def __len__(self):
        return len(self.ids)

