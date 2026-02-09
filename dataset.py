import os
import re
from typing import Tuple, List
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import config

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageClassDataset(Dataset):
    def __init__(self, root: str, image_size: int = 512):
        self.root = root
        self.samples: List[Tuple[int, str]] = []
        
        # Filter valid images
        candidates = []
        if not os.path.exists(self.root):
            raise ValueError(f"Image directory not found: {self.root}")

        for fn in sorted(os.listdir(self.root)):
            if not fn.lower().endswith(".jpg"):
                continue
            # Regex to parse "ClassID_ImageID.jpg"
            m = re.match(r"^(\d+)_(\d+)\.jpg$", fn)
            if not m:
                continue
            cls = int(m.group(1))
            candidates.append((cls, os.path.join(self.root, fn)))

        if not candidates:
            raise ValueError("No valid images found matching format 'ClassID_ImageID.jpg'")

        # Basic validation (skipping pixel check for speed, trust A100 IO)
        self.samples = candidates

        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1]
        ])

    def _prompt(self, cls: int) -> str:
        return config.CLASS_PROMPTS.get(cls, f"a painting of class {cls}")
    
    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int):
        cls, path = self.samples[idx]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                img = self.tf(im)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy tensor or handle recursively (omitted for brevity)
            return self.__getitem__((idx + 1) % len(self))

        caption = self._prompt(cls)
        return {"pixel_values": img, "caption": caption, "tag": cls}

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], 0),
        "caption": [b["caption"] for b in batch],
        "tag": torch.tensor([b["tag"] for b in batch], dtype=torch.long),
    }