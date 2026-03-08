
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BreakHisDataset(Dataset):

    def __init__(self, image_paths, labels, train=True):

        if train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Resize(224,224),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224,224),
                ToTensorV2()
            ])

        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(image=img)["image"]

        label = torch.tensor(self.labels[idx]).long()

        return img, label
