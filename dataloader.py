import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

from utils.image_utils import random_augmentation, crop_img
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

# Try to import additional libraries for enhanced functionality
try:
    import cv2
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Albumentations not found. Using basic transforms only.")


class HighResDataset(Dataset):
    """
    Enhanced dataset for high-resolution image restoration with multiple loading strategies
    """
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        img_size: Tuple[int, int] = (1280, 720),
        crop_size: Optional[Tuple[int, int]] = None,
        patch_size: Optional[int] = None,
        patches_per_image: int = 16,
        augment: bool = True,
        cache_size: int = 0,
        paired_naming: str = "same",  # "same", "suffix", "prefix", "txt_file"
        pair_list_file: Optional[str] = None,
        source_suffix: str = "",
        target_suffix: str = "",
        preload: bool = False,
        normalize: bool = True,
        task: str = "derain",
        random_offset: bool = True
    ):
        """
        Args:
            source_dir: Directory containing source domain images (e.g., rainy/noisy)
            target_dir: Directory containing target domain images (e.g., clean)
            img_size: Target image size (width, height) for resizing
            crop_size: Random crop size (height, width), if None, no cropping is applied
            patch_size: Size of patches to extract from each image, if None, whole images are used
            patches_per_image: Number of patches to extract from each image when using patching
            augment: Whether to use data augmentation
            cache_size: Number of images to cache in memory (0 = no caching)
            paired_naming: How image pairs are named ('same', 'suffix', 'prefix', 'txt_file')
            pair_list_file: Path to a text file listing image pairs (source,target per line)
            source_suffix: Suffix for source images when using suffix pairing
            target_suffix: Suffix for target images when using suffix pairing
            preload: Whether to preload all images into memory
            normalize: Whether to normalize images to [-1, 1] range
            task: Specific task type for specialized augmentations ('derain', 'denoise', 'deblur')
            random_offset: Use random offset when extracting patches for better generalization
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.img_size = (img_size[1], img_size[0])  # (H, W) for PIL
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.cache_size = cache_size
        self.paired_naming = paired_naming
        self.source_suffix = source_suffix
        self.target_suffix = target_suffix
        self.preload = preload
        self.normalize = normalize
        self.task = task
        self.random_offset = random_offset
        
        # Cache storage
        self.cache = {}
        
        # Find image pairs
        self.paired_files = self._find_paired_files(pair_list_file)
        print(f"Found {len(self.paired_files)} image pairs")
        
        # Create transforms
        self._create_transforms()
        
        # Preload images if requested
        if self.preload:
            self._preload_images()
            
    def _find_paired_files(self, pair_list_file=None):
        """Find paired source and target files based on naming convention"""
        paired_files = []
        
        if self.paired_naming == "txt_file" and pair_list_file:
            # Load pairs from text file
            with open(pair_list_file, 'r') as f:
                for line in f:
                    if line.strip():
                        source_path, target_path = line.strip().split(',')
                        paired_files.append((source_path, target_path))
            return paired_files
            
        # Find all image files in source directory
        source_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            source_files.extend(list(self.source_dir.glob(f"*{ext}")))
            source_files.extend(list(self.source_dir.glob(f"*{ext.upper()}")))
        
        if self.paired_naming == "same":
            # Assume same filename in both directories
            for source_path in source_files:
                target_path = self.target_dir / source_path.name
                if target_path.exists():
                    paired_files.append((str(source_path), str(target_path)))
                    
        elif self.paired_naming == "suffix":
            # Remove suffix from source and add target suffix
            for source_path in source_files:
                source_stem = source_path.stem
                if self.source_suffix and source_stem.endswith(self.source_suffix):
                    base_name = source_stem[:-len(self.source_suffix)]
                else:
                    base_name = source_stem
                    
                target_stem = base_name + self.target_suffix
                target_path = self.target_dir / f"{target_stem}{source_path.suffix}"
                
                if target_path.exists():
                    paired_files.append((str(source_path), str(target_path)))
                    
        elif self.paired_naming == "prefix":
            # Use filename as common identifier
            for source_path in source_files:
                source_name = source_path.name
                # Find matching target file with same name
                target_candidates = list(self.target_dir.glob(f"*{source_name}"))
                if target_candidates:
                    paired_files.append((str(source_path), str(target_candidates[0])))
        
        return paired_files
    
    def _create_transforms(self):
        """Create transforms based on task and available libraries"""
        # Basic transforms
        self.resize_transform = transforms.Resize(self.img_size, interpolation=Image.Resampling.BICUBIC)
        
        # Specialized task-based augmentations using albumentations if available
        if ALBUMENTATIONS_AVAILABLE and self.augment:
            # Common augmentations
            common_augs = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
            ]
            
            # Task-specific augmentations
            if self.task == "derain":
                task_augs = [
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
                ]
            elif self.task == "denoise":
                task_augs = [
                    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
                    A.GaussNoise(var_limit=(1, 5), p=0.1),  # Small amount to avoid overwhelming existing noise
                ]
            elif self.task == "deblur":
                task_augs = [
                    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
                    A.MotionBlur(blur_limit=3, p=0.1),  # Small amount to avoid overwhelming existing blur
                ]
            else:  # General restoration
                task_augs = [
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                ]
                
            # Create final transform
            self.aug_transform = A.Compose(
                common_augs + task_augs,
                additional_targets={'target': 'image'}
            )
        else:
            self.aug_transform = None
    
    def _preload_images(self):
        """Preload images into memory"""
        print("Preloading images into memory...")
        for i, (source_path, target_path) in enumerate(self.paired_files):
            # Only preload up to cache_size
            if self.cache_size > 0 and i >= self.cache_size:
                break
                
            # Load and store images
            source_img = Image.open(source_path).convert('RGB')
            target_img = Image.open(target_path).convert('RGB')
            
            # Resize if needed
            source_img = self.resize_transform(source_img)
            target_img = self.resize_transform(target_img)
            
            self.cache[i] = (source_img, target_img)
            
        print(f"Preloaded {len(self.cache)} image pairs")
    
    def __len__(self):
        if self.patch_size is not None:
            # When using patches, return total number of patches
            return len(self.paired_files) * self.patches_per_image
        else:
            return len(self.paired_files)
    
    def _apply_augmentation(self, source_img, target_img):
        """Apply data augmentation to both images consistently"""
        if not self.augment:
            return source_img, target_img
            
        # Apply albumentations if available
        if ALBUMENTATIONS_AVAILABLE and self.aug_transform is not None:
            # Convert to numpy arrays
            source_np = np.array(source_img)
            target_np = np.array(target_img)
            
            # Apply same augmentation to both images
            augmented = self.aug_transform(image=source_np, target=target_np)
            source_aug = Image.fromarray(augmented['image'])
            target_aug = Image.fromarray(augmented['target'])
            return source_aug, target_aug
        else:
            # Basic PyTorch transforms
            # Random horizontal flip
            if random.random() > 0.5:
                source_img = TF.hflip(source_img)
                target_img = TF.hflip(target_img)
                
            # Random vertical flip
            if random.random() > 0.5:
                source_img = TF.vflip(source_img)
                target_img = TF.vflip(target_img)
                
            # Random rotation
            if random.random() > 0.7:
                angle = random.choice([90, 180, 270])
                source_img = TF.rotate(source_img, angle)
                target_img = TF.rotate(target_img, angle)
                
            return source_img, target_img
    
    def _random_crop(self, source_img, target_img):
        """Apply random cropping to both images at the same location"""
        if self.crop_size is None:
            return source_img, target_img
            
        # Get current dimensions
        width, height = source_img.size
        crop_height, crop_width = self.crop_size
        
        # Ensure crop size is not larger than image
        crop_height = min(height, crop_height)
        crop_width = min(width, crop_width)
        
        # Get random crop position
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        
        # Apply crop
        source_crop = TF.crop(source_img, top, left, crop_height, crop_width)
        target_crop = TF.crop(target_img, top, left, crop_height, crop_width)
        
        return source_crop, target_crop
    
    def _extract_patches(self, source_img, target_img, idx):
        """Extract multiple patches from a single image pair"""
        if self.patch_size is None:
            return [source_img], [target_img]
            
        # Convert to numpy for easier patching
        source_np = np.array(source_img)
        target_np = np.array(target_img)
        
        # Get image dimensions
        height, width = source_np.shape[:2]
        
        # Calculate maximum valid offsets
        max_h = height - self.patch_size
        max_w = width - self.patch_size
        
        if max_h <= 0 or max_w <= 0:
            # Image too small, resize and try again
            source_img = source_img.resize((self.patch_size * 2, self.patch_size * 2), Image.Resampling.BICUBIC)
            target_img = target_img.resize((self.patch_size * 2, self.patch_size * 2), Image.Resampling.BICUBIC)
            return self._extract_patches(source_img, target_img, idx)
        
        source_patches = []
        target_patches = []
        
        # Generate deterministic random offsets based on idx
        np.random.seed(idx)
        
        for i in range(self.patches_per_image):
            # Use deterministic or random offset
            if self.random_offset:
                h_offset = np.random.randint(0, max_h + 1)
                w_offset = np.random.randint(0, max_w + 1)
            else:
                # Evenly distribute patches in a grid-like pattern
                grid_size = int(np.ceil(np.sqrt(self.patches_per_image)))
                h_step = max(1, max_h // grid_size)
                w_step = max(1, max_w // grid_size)
                
                h_idx = (i // grid_size) % grid_size
                w_idx = i % grid_size
                
                h_offset = min(h_idx * h_step, max_h)
                w_offset = min(w_idx * w_step, max_w)
            
            # Extract patches
            source_patch = source_np[h_offset:h_offset+self.patch_size, w_offset:w_offset+self.patch_size, :]
            target_patch = target_np[h_offset:h_offset+self.patch_size, w_offset:w_offset+self.patch_size, :]
            
            # Convert back to PIL
            source_patches.append(Image.fromarray(source_patch))
            target_patches.append(Image.fromarray(target_patch))
        
        return source_patches, target_patches
    
    def _transform_to_tensor(self, img):
        """Convert PIL image to normalized tensor"""
        # Convert to tensor (0-1 range)
        tensor = TF.to_tensor(img)
        
        # Normalize to [-1, 1] range if requested
        if self.normalize:
            tensor = (tensor * 2) - 1
            
        return tensor
    
    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2
    
    def __getitem__(self, idx):
        # Calculate original image index and patch index
        if self.patch_size is not None:
            img_idx = idx // self.patches_per_image
            patch_idx = idx % self.patches_per_image
        else:
            img_idx = idx
            patch_idx = 0
        
        # Check bounds
        if img_idx >= len(self.paired_files):
            img_idx = img_idx % len(self.paired_files)
        
        # Try to get from cache first
        if img_idx in self.cache:
            source_img, target_img = self.cache[img_idx]
        else:
            # Load images
            source_path, target_path = self.paired_files[img_idx]
            source_img = Image.open(source_path).convert('RGB')
            target_img = Image.open(target_path).convert('RGB')
            
            # Resize if needed
            if source_img.size != self.img_size:
                source_img = self.resize_transform(source_img)
            if target_img.size != self.img_size:
                target_img = self.resize_transform(target_img)
            
            # Add to cache if there's space
            if self.cache_size > 0 and len(self.cache) < self.cache_size:
                self.cache[img_idx] = (source_img.copy(), target_img.copy())
        
        # Apply augmentation
        source_img, target_img = self._apply_augmentation(source_img, target_img)
        
        # Apply random crop if specified
        if self.crop_size is not None:
            source_img, target_img = self._random_crop(source_img, target_img)
        
        # Extract patches if needed
        if self.patch_size is not None:
            source_patches, target_patches = self._extract_patches(source_img, target_img, img_idx)
            source_img = source_patches[patch_idx]
            target_img = target_patches[patch_idx]
        
        # # Convert to tensors
        # source_tensor = self._transform_to_tensor(source_img)
        # target_tensor = self._transform_to_tensor(target_img)
        
        # return source_tensor, target_tensor
        
        degrad_patch_1, clean_patch_1 = random_augmentation(*self._crop_patch(source_img, target_img))
        degrad_patch_2, clean_patch_2 = random_augmentation(*self._crop_patch(source_img, target_img))
        
        # Convert to tensors
        # source_tensor = self._transform_to_tensor(source_img)
        # target_tensor = self._transform_to_tensor(target_img)
        clean_patch_1, clean_patch_2 = self.toTensor(clean_patch_1), self.toTensor(clean_patch_2)
        degrad_patch_1, degrad_patch_2 = self.toTensor(degrad_patch_1), self.toTensor(degrad_patch_2)
        
        return ["", ""], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2



class TiledPredictionDataset(Dataset):
    """
    Dataset for inference with tiled prediction on very large images
    """
    def __init__(
        self,
        input_dir: str,
        tile_size: int = 512,
        tile_overlap: int = 64,
        max_image_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ):
        """
        Args:
            input_dir: Directory containing input images
            tile_size: Size of tiles to extract (square)
            tile_overlap: Overlap between adjacent tiles
            max_image_size: Maximum image size (width, height), larger images will be resized
            normalize: Whether to normalize images to [-1, 1] range
        """
        self.input_dir = Path(input_dir)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.max_image_size = max_image_size
        self.normalize = normalize
        
        # Find all image files
        self.image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            self.image_files.extend(list(self.input_dir.glob(f"*{ext}")))
            self.image_files.extend(list(self.input_dir.glob(f"*{ext.upper()}")))
        
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images")
        
        # Precompute tiles for each image
        self.tiles = self._precompute_tiles()
        
    def _precompute_tiles(self):
        """Precompute tiles for each image"""
        all_tiles = []
        
        for img_idx, img_path in enumerate(self.image_files):
            # Load image (but don't keep in memory)
            img = Image.open(img_path).convert('RGB')
            original_size = img.size  # (W, H)
            
            # Resize if needed
            if self.max_image_size and (original_size[0] > self.max_image_size[0] or original_size[1] > self.max_image_size[1]):
                scale = min(self.max_image_size[0] / original_size[0], self.max_image_size[1] / original_size[1])
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                resized = True
            else:
                new_size = original_size
                resized = False
            
            # Calculate effective stride
            stride = self.tile_size - self.tile_overlap
            
            # Calculate number of tiles in each dimension
            n_tiles_w = max(1, (new_size[0] - self.tile_overlap - 1) // stride + 1)
            n_tiles_h = max(1, (new_size[1] - self.tile_overlap - 1) // stride + 1)
            
            # Store tiles info
            for h_idx in range(n_tiles_h):
                for w_idx in range(n_tiles_w):
                    # Calculate tile coordinates
                    left = min(w_idx * stride, new_size[0] - self.tile_size)
                    top = min(h_idx * stride, new_size[1] - self.tile_size)
                    
                    # Store tile info
                    tile_info = {
                        'img_idx': img_idx,
                        'img_path': str(img_path),
                        'original_size': original_size,
                        'resized': resized,
                        'new_size': new_size,
                        'tile_pos': (left, top),
                        'tile_idx': (h_idx, w_idx),
                        'n_tiles': (n_tiles_h, n_tiles_w)
                    }
                    all_tiles.append(tile_info)
        
        return all_tiles
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        tile_info = self.tiles[idx]
        
        # Load image
        img = Image.open(tile_info['img_path']).convert('RGB')
        
        # Resize if needed
        if tile_info['resized']:
            img = img.resize(tile_info['new_size'], Image.Resampling.BICUBIC)
        
        # Extract tile
        left, top = tile_info['tile_pos']
        tile = img.crop((left, top, left + self.tile_size, top + self.tile_size))
        
        # Convert to tensor
        tensor = TF.to_tensor(tile)
        
        # Normalize if requested
        if self.normalize:
            tensor = (tensor * 2) - 1
        
        return tensor, tile_info


class DatasetFactory:
    """
    Factory class for creating appropriate datasets based on task and configuration
    """
    @staticmethod
    def create_dataset(dataset_config):
        """
        Create dataset based on configuration
        
        Args:
            dataset_config: Dictionary containing dataset configuration
        
        Returns:
            Dataset object
        """
        print(f"dataset_config: {dataset_config}")
        dataset_type = dataset_config.get("type", "standard")
        print(f"Creating dataset with type: {dataset_type}")
        
        if dataset_type == "high_res":
            return HighResDataset(
                source_dir=dataset_config["source_dir"],
                target_dir=dataset_config["target_dir"],
                img_size=dataset_config.get("img_size", (1280, 720)),
                crop_size=dataset_config.get("crop_size", None),
                patch_size=dataset_config.get("patch_size", None),
                patches_per_image=dataset_config.get("patches_per_image", 16),
                augment=dataset_config.get("augment", True),
                cache_size=dataset_config.get("cache_size", 0),
                paired_naming=dataset_config.get("paired_naming", "same"),
                pair_list_file=dataset_config.get("pair_list_file", None),
                source_suffix=dataset_config.get("source_suffix", ""),
                target_suffix=dataset_config.get("target_suffix", ""),
                preload=dataset_config.get("preload", False),
                normalize=dataset_config.get("normalize", True),
                task=dataset_config.get("task", "derain"),
                random_offset=dataset_config.get("random_offset", True)
            )
        elif dataset_type == "tiled_prediction":
            return TiledPredictionDataset(
                input_dir=dataset_config["input_dir"],
                tile_size=dataset_config.get("tile_size", 512),
                tile_overlap=dataset_config.get("tile_overlap", 64),
                max_image_size=dataset_config.get("max_image_size", None),
                normalize=dataset_config.get("normalize", True)
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    @staticmethod
    def create_data_loaders(train_config, val_config=None, test_config=None):
        """
        Create data loaders for training, validation, and testing
        
        Args:
            train_config: Dictionary containing training dataset configuration
            val_config: Dictionary containing validation dataset configuration
            test_config: Dictionary containing test dataset configuration
        
        Returns:
            Tuple of data loaders (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = DatasetFactory.create_dataset(train_config)
        
        val_dataset = None
        if val_config:
            val_dataset = DatasetFactory.create_dataset(val_config)
            
        test_dataset = None
        if test_config:
            test_dataset = DatasetFactory.create_dataset(test_config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.get("batch_size", 1),
            shuffle=train_config.get("shuffle", True),
            num_workers=train_config.get("num_workers", 4),
            pin_memory=train_config.get("pin_memory", True),
            drop_last=train_config.get("drop_last", True),
            persistent_workers=train_config.get("persistent_workers", False)
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_config.get("batch_size", 1),
                shuffle=False,
                num_workers=val_config.get("num_workers", 2),
                pin_memory=val_config.get("pin_memory", True)
            )
            
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=test_config.get("batch_size", 1),
                shuffle=False,
                num_workers=test_config.get("num_workers", 2),
                pin_memory=test_config.get("pin_memory", True)
            )
        
        return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataloader")
    parser.add_argument("--source-dir", type=str, required=True, help="Source directory")
    parser.add_argument("--target-dir", type=str, required=True, help="Target directory")
    parser.add_argument("--output-dir", type=str, default="dataloader_test", help="Output directory")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size")
    parser.add_argument("--img-width", type=int, default=1280, help="Image width")
    parser.add_argument("--img-height", type=int, default=720, help="Image height")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    dataset_config = {
        "type": "high_res",
        "source_dir": args.source_dir,
        "target_dir": args.target_dir,
        "img_size": (args.img_width, args.img_height),
        "patch_size": args.patch_size,
        "patches_per_image": 4,
        "augment": True,
        "cache_size": 5,
        "preload": True,
        "task": "derain"
    }
    
    dataset = DatasetFactory.create_dataset(dataset_config)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    
    # Visualize some samples
    for i, (source, target) in enumerate(dataloader):
        if i >= 3:  # Visualize 3 batches
            break
            
        # Convert back to image
        source = (source + 1) / 2
        target = (target + 1) / 2
        
        # Create grid
        batch_size = source.size(0)
        grid = []
        for b in range(batch_size):
            # Add source and target side by side
            s = source[b]
            t = target[b]
            pair = torch.cat([s, t], dim=2)
            grid.append(pair)
        
        # Stack all pairs in batch
        grid = torch.cat(grid, dim=1)
        
        # Save image
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(grid_np).save(f"{args.output_dir}/batch_{i}.png")
    
    print(f"Visualized {min(3, len(dataloader))} batches in {args.output_dir}")