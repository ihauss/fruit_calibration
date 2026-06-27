from .base import SegmentationEngine, get_largest_contour
import torch
import cv2
import numpy as np
from albumentations import Compose, Normalize, Resize, ToTensorV2
import segmentation_models_pytorch as smp
from typing import Tuple

def remove_model_prefix(state_dict):
    return {key[6:] if key.startswith('model.') else key: value for key, value in state_dict.items()}

class UNetSegmentation(SegmentationEngine):
    def __init__(self, checkpoint_path: str,
                 device: str = None,
                 input_size: Tuple[int, int] = (256, 256),
                 smooth_kernel: int = 15):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = input_size
        self.smooth_kernel = smooth_kernel

        # Construire le modèle
        self.model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=3
        )
        # Charger les poids
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = remove_model_prefix(checkpoint['state_dict'])
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Préparer les transformations (identiques à l'entraînement)
        self.transform = Compose([
            Resize(*self.input_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _preprocess(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = rgb.shape[:2]
        transformed = self.transform(image=rgb)
        tensor = transformed["image"].unsqueeze(0)
        return tensor, original_size, rgb

    def _predict(self, tensor, original_size):
        with torch.no_grad():
            tensor = tensor.to(self.device)
            output = self.model(tensor)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
        mask = cv2.resize(pred.astype(np.uint8),
                          (original_size[1], original_size[0]),
                          interpolation=cv2.INTER_NEAREST)
        return mask

    def _smooth_mask(self, mask):
        # Morphologie + Gaussian (vous pouvez adapter)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.smooth_kernel, self.smooth_kernel))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        mask_float = opened.astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(mask_float, (self.smooth_kernel, self.smooth_kernel), 1.0)
        _, smoothed = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)
        return (smoothed * 255).astype(np.uint8)

    def get_masks(self, image):
        tensor, orig_size, _ = self._preprocess(image)
        mask_full = self._predict(tensor, orig_size)
        mango_bin = (mask_full == 1).astype(np.uint8) * 255
        card_bin = (mask_full == 2).astype(np.uint8) * 255
        mango_smooth = self._smooth_mask(mango_bin)
        card_smooth = self._smooth_mask(card_bin)
        return {'mango': mango_smooth, 'card': card_smooth}

    def get_contours(self, image):
        masks = self.get_masks(image)
        return {k: get_largest_contour(v) for k, v in masks.items()}
