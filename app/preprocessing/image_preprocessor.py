"""
Image Data Preprocessor module.
Phase 4 — Handles image validation, loading, channel check, augmentations,
and tensor normalization for deep learning backbones (ResNet, EfficientNet).
"""
import os
import torch
from pathlib import Path
from typing import Tuple, Optional, Union
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T


DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImagePreprocessor:
    def __init__(
        self,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        mean: list = DEFAULT_MEAN,
        std: list = DEFAULT_STD
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

        # Training augmentation pipeline
        self.train_transforms = T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

        # Validation / Test transformation pipeline
        self.val_transforms = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def validate_image_file(self, image_path: Union[str, Path]) -> Path:
        """
        Validate image file existence, extension, and readability.
        Raises ValueError or FileNotFoundError if invalid/corrupted.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist at: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension '{path.suffix}'. Supported: {SUPPORTED_EXTENSIONS}")

        try:
            with Image.open(path) as img:
                img.verify()  # Verify image integrity
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ValueError(f"Corrupted or unreadable image file at '{path}': {e}")

        return path

    def load_and_preprocess(
        self,
        image_path: Union[str, Path],
        is_training: bool = False
    ) -> torch.Tensor:
        """
        Load an image file, convert to RGB, apply transforms, and return a 4D Tensor (1, C, H, W).
        """
        valid_path = self.validate_image_file(image_path)

        # Open image and convert to 3-channel RGB
        with Image.open(valid_path) as img:
            img_rgb = img.convert("RGB")
            
            # Verify dimensions
            if img_rgb.width < 10 or img_rgb.height < 10:
                raise ValueError(f"Invalid image dimensions ({img_rgb.width}x{img_rgb.height}) at '{valid_path}'")

            transform = self.train_transforms if is_training else self.val_transforms
            tensor = transform(img_rgb)  # Shape: (3, H, W)
            
            return tensor.unsqueeze(0)  # Shape: (1, 3, H, W)

    def preprocess_pil_image(
        self,
        img: Image.Image,
        is_training: bool = False
    ) -> torch.Tensor:
        """
        Preprocess an already opened PIL Image instance (e.g. from FastAPI file upload).
        """
        img_rgb = img.convert("RGB")
        transform = self.train_transforms if is_training else self.val_transforms
        tensor = transform(img_rgb)
        return tensor.unsqueeze(0)
