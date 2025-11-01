import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from typing import Dict, Any, Tuple
import polars as pl
from pathlib import Path

class BiomassDataset(Dataset):
    def __init__(self, dataframe: pl.DataFrame, img_dir: Path):
        self.dataframe = dataframe
        self.img_dir = img_dir
        # ImageNet normalization required for pretrained models
        self.transform = transforms.Compose([
            transforms.Resize((448, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        item: Dict[str, Any] = self.dataframe.row(idx, named=True)
        img_name = os.path.join(self.img_dir, item['image_path'])
        raw_image = Image.open(img_name).convert('RGB')
        image = self.transform(raw_image)
        # Ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        labels = np.array([ item['Dry_Green_g'], item['Dry_Dead_g'], item['Dry_Clover_g'], item['GDM_g'], item['Dry_Total_g']], dtype=np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels