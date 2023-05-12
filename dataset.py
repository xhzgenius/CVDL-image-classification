import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from utils import label_str2int, label_int2str
print("CUDA availability:", torch.cuda.is_available())

class MyDataset(Dataset):
    def __init__(self, data_path):
        print("Loading dataset... ")
        self.images = []
        self.labels = []
        for label_name in os.listdir(data_path):
            for image_name in os.listdir(os.path.join(data_path, label_name)):
                image = cv2.imread(os.path.join(data_path, label_name, image_name))
                if image.shape != (150, 150, 3):
                    # raise ValueError("Wrong shape of image: %s, %s, %s" % image.shape)
                    continue
                self.images.append(image)
                self.labels.append(label_str2int(label_name))
            print("Label %s has %d instances. "%(label_name, len(os.listdir(os.path.join(data_path, label_name)))))
        print("Dataset loaded successfully. ")
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)
