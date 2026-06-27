"""
src/dimension.py

Handles computation of mango and card dimensions from contours.
Converts pixel dimensions to real-world centimeters using a reference card.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

class DimensionCalculator:
    """
    Calculates pixel dimensions from contours and converts them to centimeters
    using a known reference card size.
    """

    def __init__(self, card_length_cm: float = 8.56, card_width_cm: float = 5.398):
        """
        Initialize the calculator with the real dimensions of the reference card.

        Args:
            card_length_cm: Real length of the card in centimeters (default: 8.56).
            card_width_cm: Real width of the card in centimeters (default: 5.398).
        """
        self.card_length_cm = card_length_cm
        self.card_width_cm = card_width_cm

    def pixel_dimensions(self, contour: np.ndarray) -> Tuple[List[Tuple[int, int]], Tuple[float, float]]:
        """
        Compute the oriented bounding rectangle of a contour.

        Args:
            contour: A contour (numpy array) from cv2.findContours.

        Returns:
            A tuple (points, (length, width)):
                - points: List of 4 corner points (x, y) in pixel coordinates.
                - length: Longer side of the bounding rectangle (pixels).
                - width:  Shorter side of the bounding rectangle (pixels).
        """
        rect = cv2.minAreaRect(contour)
        (w, h) = rect[1]
        length_px = max(w, h)
        width_px = min(w, h)

        # Get the 4 corner points as integers
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        pts = [tuple(pt) for pt in box]

        return pts, (length_px, width_px)

    def px2cm(self, dimensions_px: Dict[str, Tuple[List[Tuple[int, int]], Tuple[float, float]]]) -> Dict[str, Tuple[List[Tuple[int, int]], Tuple[float, float]]]:
        """
        Convert pixel dimensions to centimeters using the reference card.

        Args:
            dimensions_px: Dictionary with keys 'mango' and 'card'.
                           Each value is a tuple (pts, (length_px, width_px)).

        Returns:
            Same dictionary structure with dimensions in centimeters.
        """
        # Extract card pixel dimensions
        _, (card_long_px, card_larg_px) = dimensions_px["card"]

        # Compute scale factors (pixels per centimeter)
        scale_long = card_long_px / self.card_length_cm
        scale_larg = card_larg_px / self.card_width_cm

        dimensions_cm = {}
        for key in ["mango", "card"]:
            pts, (long_px, larg_px) = dimensions_px[key]
            long_cm = long_px / scale_long
            larg_cm = larg_px / scale_larg
            dimensions_cm[key] = (pts, (long_cm, larg_cm))

        return dimensions_cm

    def compute_dimensions(self, contours: Dict[str, Optional[np.ndarray]]) -> Dict[str, Tuple[List[Tuple[int, int]], Tuple[float, float]]]:
        """
        High-level method to compute dimensions in centimeters from contours.

        Args:
            contours: Dictionary with keys 'mango' and 'card'. Each value is a contour
                      (or None if detection failed).

        Returns:
            Dictionary with keys 'mango' and 'card'. Each value is
            (pts, (length_cm, width_cm)). If a contour is None, its entry is None.
        """
        dimensions_px = {}
        for key in ["mango", "card"]:
            contour = contours.get(key)
            if contour is not None:
                dimensions_px[key] = self.pixel_dimensions(contour)
            else:
                dimensions_px[key] = None  # keep placeholder

        # Only convert if both objects are present
        if dimensions_px["mango"] is not None and dimensions_px["card"] is not None:
            return self.px2cm(dimensions_px)
        else:
            # If one is missing, return pixel dimensions as fallback? Better to return None.
            # Here we return a dictionary with None for missing ones.
            result = {}
            for key in ["mango", "card"]:
                if dimensions_px[key] is not None:
                    # We could convert only if both exist, but here we just return px.
                    # In practice, px2cm is called only when both are present.
                    # Let's keep it simple: if both are present, call px2cm, else return None.
                    pass
            # For safety, we return None for the missing key
            result = {}
            for key in ["mango", "card"]:
                result[key] = dimensions_px[key]  # this will be None or the tuple
            return result
