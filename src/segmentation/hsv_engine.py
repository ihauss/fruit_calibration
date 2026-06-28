"""
src/segmentation/hsv_engine.py

HSV-based segmentation engine for mango and reference card detection.
Uses color thresholding in HSV space to isolate yellow (mango) and red (card) regions.
"""

from .base import SegmentationEngine, get_largest_contour
import cv2
import numpy as np
from typing import Dict, Optional, Tuple


class HSVSegmentation(SegmentationEngine):
    """
    Segmentation engine based on HSV color thresholding.

    This engine uses fixed HSV ranges to detect:
        - Mango: yellow hues (default: 10-50)
        - Card: red hues (two ranges: 0-10 and 170-179, due to hue wrap-around)

    Morphological operations (opening and closing) are applied to clean
    the binary masks by removing small noise and filling small holes.

    The HSV ranges can be customized via constructor parameters, making
    this engine adaptable to different lighting conditions or fruit varieties.

    Usage:
        engine = HSVSegmentation(
            mango_lower=(10, 50, 50),
            mango_upper=(50, 255, 255),
            kernel_size=15
        )
        contours = engine.get_contours(image)
    """

    def __init__(self,
                 mango_lower: Tuple[int, int, int] = (10, 50, 50),
                 mango_upper: Tuple[int, int, int] = (50, 255, 255),
                 card_lower1: Tuple[int, int, int] = (0, 50, 50),
                 card_upper1: Tuple[int, int, int] = (10, 255, 255),
                 card_lower2: Tuple[int, int, int] = (170, 50, 50),
                 card_upper2: Tuple[int, int, int] = (179, 255, 255),
                 kernel_size: int = 25):
        """
        Initialize the HSV segmentation engine with configurable color ranges.

        Args:
            mango_lower: Lower HSV bounds for mango detection (H, S, V).
            mango_upper: Upper HSV bounds for mango detection (H, S, V).
            card_lower1: Lower HSV bounds for red card (first range).
            card_upper1: Upper HSV bounds for red card (first range).
            card_lower2: Lower HSV bounds for red card (second range).
            card_upper2: Upper HSV bounds for red card (second range).
            kernel_size: Size of the structuring element for morphological operations.
                         Larger values produce more aggressive cleaning.
        """
        # Convert tuples to numpy arrays for OpenCV operations
        self.mango_lower = np.array(mango_lower)
        self.mango_upper = np.array(mango_upper)
        self.card_lower1 = np.array(card_lower1)
        self.card_upper1 = np.array(card_upper1)
        self.card_lower2 = np.array(card_lower2)
        self.card_upper2 = np.array(card_upper2)
        # Structuring element for morphological operations
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def get_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate binary masks for mango and card using HSV thresholding.

        The process:
            1. Convert image from BGR to HSV color space.
            2. Apply inRange() for mango (yellow) and card (red).
            3. Clean masks using morphological opening (remove small noise)
               and closing (fill small holes).

        Args:
            image: Input image in BGR format (as read by OpenCV).

        Returns:
            Dictionary with keys 'mango' and 'card', each mapping to a
            binary mask (uint8) where 255 = object, 0 = background.
        """
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --- Mango detection (yellow) ---
        # Yellow typically falls in the 20-40 hue range, but we use a wider
        # range (10-50) to handle variations in lighting and fruit ripeness.
        mask_mango = cv2.inRange(hsv, self.mango_lower, self.mango_upper)
        # Opening removes small noise and smooths edges
        mask_mango = cv2.morphologyEx(mask_mango, cv2.MORPH_OPEN, self.kernel)

        # --- Card detection (red) ---
        # Red wraps around the hue circle (0 and 180 in OpenCV HSV).
        # We need two ranges to capture all red shades.
        mask1 = cv2.inRange(hsv, self.card_lower1, self.card_upper1)  # Red range 1 (0-10)
        mask2 = cv2.inRange(hsv, self.card_lower2, self.card_upper2)  # Red range 2 (170-179)
        mask_card = cv2.bitwise_or(mask1, mask2)

        # Clean the card mask: opening to remove noise, then closing to fill holes
        # Closing is particularly useful for cards with text/logos that create
        # internal gaps in the mask.
        mask_card = cv2.morphologyEx(mask_card, cv2.MORPH_OPEN, self.kernel)
        mask_card = cv2.morphologyEx(mask_card, cv2.MORPH_CLOSE, self.kernel)

        return {'mango': mask_mango, 'card': mask_card}

    def get_contours(self, image: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract the largest contour for each segmented object.

        This method generates binary masks and then extracts the largest
        external contour from each mask. The resulting contours can be
        used for dimension calculation and visualization.

        Args:
            image: Input image in BGR format.

        Returns:
            Dictionary with keys 'mango' and 'card', each mapping to a
            contour (numpy array of shape (N, 1, 2)) or None if the
            object was not detected.
        """
        # Generate binary masks using HSV thresholding
        masks = self.get_masks(image)

        # Extract the largest contour from each mask
        # The dictionary comprehension applies get_largest_contour() to each mask
        return {k: get_largest_contour(v) for k, v in masks.items()}
