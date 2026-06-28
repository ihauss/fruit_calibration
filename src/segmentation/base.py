"""
src/segmentation/base.py

Defines the abstract base class for all segmentation engines.
Provides a common interface for different segmentation implementations
(HSV, UNet, etc.) and a utility function for contour extraction.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, Tuple
import cv2


def get_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the largest contour from a binary mask.

    This function finds all external contours in the mask and returns
    the one with the largest area. Useful for filtering out small
    artifacts or noise after segmentation.

    Args:
        mask: Binary mask (uint8) with values 0 or 255.

    Returns:
        The largest contour as a numpy array of shape (N, 1, 2),
        or None if no contours are found.
    """
    # Find all external contours (RETR_EXTERNAL ignores inner holes)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Return the contour with maximum area
    return max(contours, key=cv2.contourArea)


class SegmentationEngine(ABC):
    """
    Abstract base class for all segmentation engines.

    This interface defines the contract that all concrete segmentation
    implementations must follow. It ensures that any engine (HSV, UNet,
    SAM, etc.) can be used interchangeably in the pipeline.

    Implementing classes must provide:
        - get_masks(): Returns binary masks for mango and card.
        - get_contours(): Returns the largest contours from the masks.

    Usage:
        class MySegmentation(SegmentationEngine):
            def get_masks(self, image):
                # Implementation specific to my method
                ...
            def get_contours(self, image):
                masks = self.get_masks(image)
                return {
                    'mango': get_largest_contour(masks['mango']),
                    'card': get_largest_contour(masks['card'])
                }
    """

    @abstractmethod
    def get_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate binary masks for the mango and the reference card.

        This method performs the core segmentation and returns two
        binary masks: one for the mango and one for the card.

        Args:
            image: Input image in BGR format (as read by OpenCV).

        Returns:
            Dictionary with keys 'mango' and 'card', each mapping to a
            binary mask (uint8) where pixels belonging to the object
            are set to 255 and the background is 0.
        """
        pass

    @abstractmethod
    def get_contours(self, image: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract the largest contour for each segmented object.

        This method typically calls get_masks() internally and then
        extracts the largest contour from each mask using the utility
        function get_largest_contour().

        Args:
            image: Input image in BGR format.

        Returns:
            Dictionary with keys 'mango' and 'card', each mapping to a
            contour (numpy array of shape (N, 1, 2)) or None if the
            object was not detected.
        """
        pass
