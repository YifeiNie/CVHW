import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

class Dataset_(Dataset):
    def __init__(self, size, rgb_path, depth_path):
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        if isinstance(size, tuple):
            self.size, _ = size
        else:
            self.size = size

        self.transform = ImageTransform(size=self.size)

        # Get a list of files with RGB and depth images containing only .png files
        self.rgb_imgs = sorted([f for f in os.listdir(self.rgb_path) if f.endswith(".png")])
        self.depth_imgs = sorted([f for f in os.listdir(self.depth_path) if f.endswith(".png")])
        print(f"num rgb is {len(self.rgb_imgs)}, num depth is {len(self.depth_imgs)}")
        assert len(self.rgb_imgs) == len(self.depth_imgs), "Mismatch between RGB and depth image counts"

    def __len__(self):
        return len(self.rgb_imgs)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_path, self.rgb_imgs[idx]) # Returns tensor, shape (C, H, W)
        depth_path = os.path.join(self.depth_path, self.depth_imgs[idx])

        rgb_image = cv2.imread(rgb_path)
        # The image loaded by OpenCV is in BGR format
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) # Convert to RGB
        depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image was loaded successfully
        if rgb_image is None:
            raise ValueError(f"Failed to load RGB image: {rgb_path}")
        if depth_image is None:
            raise ValueError(f"Failed to load depth image: {depth_path}")

        # Convert to PIL Image
        rgb_image = Image.fromarray(rgb_image)
        depth_image = Image.fromarray(depth_image).convert("L")
        if self.transform:
            rgb_image, depth_image = self.transform(rgb_image, depth_image)

        return rgb_image, depth_image

class ImageTransform:
    def __init__(self, size):
        self.size = size
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, rgb_image, depth_image):
        # RandomResizedCrop parameters
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            rgb_image, scale=[0.08, 1.0], ratio=[0.75, 1.3333]
        )

        rgb_crop = TF.resized_crop(rgb_image, i, j, h, w, (self.size, self.size))
        depth_crop = TF.resized_crop(depth_image, i, j, h, w, (self.size, self.size))

        # RandomHorizontalFlip
        if random.random() > 0.5:
            rgb_flip = TF.hflip(rgb_crop)
            depth_flip = TF.hflip(depth_crop)

            # Convert to tensors
            rgb_tensor = self.pil_to_tensor(rgb_flip)
            depth_tensor = self.pil_to_tensor(depth_flip)
        else:
            # Convert to tensors
            rgb_tensor = self.pil_to_tensor(rgb_crop)
            depth_tensor = self.pil_to_tensor(depth_crop)

        # Generate random sigma
        sigma = random.uniform(0.1, 2.0)

        # Calculate kernel_size, ensure it"s an odd integer
        kernel_size = int(0.1 * self.size)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel_size is odd
        # Apply Gaussian Blur using torchvision.transforms.functional
        rgb_tensor = TF.gaussian_blur(rgb_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        depth_tensor = TF.gaussian_blur(depth_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        return rgb_tensor, depth_tensor