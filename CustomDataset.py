import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for filename in os.listdir(root_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(root_dir, filename), 'r') as f:
                    label = f.readline().strip()
                    try:
                        label = float(label)
                    except ValueError:
                        print(f"Error converting label to float in file: {filename}")
                        continue
                    image_path = os.path.join(root_dir, filename.replace('.txt', '.png'))
                    if os.path.exists(image_path):
                        self.image_paths.append(image_path)
                        self.labels.append(label)
                    else:
                        print(f"Image file not found for label: {filename}")


    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        label = self.labels[idx]

        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        image_pair = torch.cat((img1, img2), dim=0)
        label = torch.tensor([label], dtype=torch.float32)

        return image_pair, label
