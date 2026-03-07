import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io

class COCO_Dataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir

        with open(annotation_file, 'r') as file:
            dataset = json.load(file)

        self.idfile = {img["id"]: img["file_name"] for img in dataset["images"]}
        self.samples = []
        for ann in dataset["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            file_name = self.idfile[image_id]

            img_path = os.path.join(image_dir, file_name)
            self.samples.append((img_path, caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = io.read_image(img_path).float() / 255.0
        return image, caption
    
dataset = COCO_Dataset(
    image_dir="val2017",
    annotation_file="captions_val2017.json"
)

# print(dataset.__len__())
# first_Data = dataset.__getitem__(0)
# print(first_Data[0].shape)
# image, caption = first_Data
# print(caption)
# print(image)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)