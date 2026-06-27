from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, Tuple
import cv2

def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

class SegmentationEngine(ABC):
    """Interface commune pour tous les moteurs de segmentation."""

    @abstractmethod
    def get_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Retourne les masques binaires pour la mangue et la carte.
        Ex: {'mango': mask_uint8, 'card': mask_uint8}
        Les masques sont en uint8 (0 ou 255).
        """
        pass

    @abstractmethod
    def get_contours(self, image: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        Retourne les contours lissés de la mangue et de la carte.
        Ex: {'mango': contour, 'card': contour}
        """
        pass
