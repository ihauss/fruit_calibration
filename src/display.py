"""
src/display.py

Handles visualization of segmentation contours and dimension annotations.
Provides a Visualizer class to overlay filled regions, outlines, oriented
bounding boxes, and dimension labels in centimeters.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

class Visualizer:
    """
    Visualizes segmentation contours and dimension annotations.

    This class provides methods to overlay filled contours, draw outlines,
    display oriented bounding rectangles, and annotate with dimensions in cm.
    All visualization parameters (colors, thickness, text size, opacity) are
    configurable via the constructor.

    Usage:
        vis = Visualizer(contour_thickness=2, text_scale=1.0)
        result_img = vis.display_results(image, contours, dimensions_cm)
    """

    def __init__(self,
                 mango_color=(255, 0, 0),      # BGR for mango (red)
                 card_color=(255, 255, 0),     # BGR for card (cyan)
                 rect_color=(0, 255, 0),       # BGR for bounding rectangle
                 contour_thickness=15,
                 text_scale=2.0,
                 text_thickness=5,
                 fill_opacity=0.4,
                 outline_weight=0.6):
        """
        Initialize the visualizer with customizable appearance parameters.

        Args:
            mango_color: BGR color tuple for filling the mango region.
            card_color: BGR color tuple for filling the card region.
            rect_color: BGR color for the oriented rectangle border.
            contour_thickness: Thickness of contour outlines (if >0).
                               When using `cv2.FILLED`, pass -1 separately.
            text_scale: Font scale factor for dimension labels.
            text_thickness: Thickness of dimension text strokes.
            fill_opacity: Opacity weight (0-1) for the filled overlay.
            outline_weight: Weight (0-1) for the outline overlay before blending.
        """
        self.mango_color = mango_color
        self.card_color = card_color
        self.rect_color = rect_color
        self.contour_thickness = contour_thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.fill_opacity = fill_opacity
        self.outline_weight = outline_weight

    def _draw_contour(self, image: np.ndarray, contour: np.ndarray,
                      color: Tuple[int, int, int], fill: bool = False) -> np.ndarray:
        """
        Draw a single contour on a copy of the image.

        Internal helper; returns a new image with the contour drawn.
        If fill is True, the contour is filled completely; otherwise only
        the outline is drawn (using self.contour_thickness).

        Args:
            image: Input image (BGR).
            contour: Contour array from cv2.findContours.
            color: BGR color.
            fill: If True, fill the interior; else draw outline.

        Returns:
            Image with the contour drawn (copy).
        """
        img_copy = image.copy()
        if fill:
            # Use cv2.FILLED to fill the entire region
            cv2.drawContours(img_copy, [contour], -1, color, thickness=cv2.FILLED)
        else:
            # Draw only the outline with the configured thickness
            cv2.drawContours(img_copy, [contour], -1, color, self.contour_thickness)
        return img_copy

    def display_contours(self, image: np.ndarray,
                         contours: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
        """
        Overlay filled contours and outlines for mango and card.

        This method creates a blended overlay: a semi‑transparent filled
        region is superimposed with a crisp outline of the same contour.
        This results in a visually pleasing and clear segmentation mask.

        Args:
            image: Original image (BGR).
            contours: Dictionary with keys 'mango' and 'card'.
                      Each value is a contour (numpy array) or None if not detected.

        Returns:
            Image with the contour overlay blended onto the original.
        """
        # Safely retrieve contours; default to None if key missing
        mango_contour = contours.get('mango')
        card_contour = contours.get('card')

        # 1. Build the filled overlay
        display = image.copy()
        if mango_contour is not None:
            display = self._draw_contour(display, mango_contour, self.mango_color, fill=True)
        if card_contour is not None:
            display = self._draw_contour(display, card_contour, self.card_color, fill=True)

        # 2. Build the outline overlay (crisp edges)
        outline_img = image.copy()
        if mango_contour is not None:
            outline_img = self._draw_contour(outline_img, mango_contour, self.mango_color, fill=False)
        if card_contour is not None:
            outline_img = self._draw_contour(outline_img, card_contour, self.card_color, fill=False)

        # 3. Blend the two layers to combine transparency and sharp edges
        blended = cv2.addWeighted(display, self.fill_opacity,
                                  outline_img, self.outline_weight, 5)
        return blended

    def draw_dimensions(self, image: np.ndarray,
                        dimensions_cm: Dict[str, Tuple[List[Tuple[int, int]], Tuple[float, float]]],
                        rect_color: Optional[Tuple[int, int, int]] = None,
                        thickness: Optional[int] = None) -> np.ndarray:
        """
        Draw oriented rectangles and dimension labels.

        For each object (mango, card), the oriented bounding box is drawn
        using the provided corner points, and a centered label shows the
        length and width in centimeters (white text on a black background).

        Args:
            image: Image to annotate (BGR).
            dimensions_cm: Dictionary with keys 'mango' and 'card'.
                           Each value is (pts, (length_cm, width_cm)).
                           pts is a list of 4 corner points (x,y) in pixels.
            rect_color: Color for the rectangle border. If None, uses self.rect_color.
            thickness: Thickness of rectangle lines. If None, uses self.contour_thickness.

        Returns:
            Annotated image with rectangles and dimension labels.
        """
        img_out = image.copy()
        rect_color = rect_color if rect_color is not None else self.rect_color
        thickness = thickness if thickness is not None else self.contour_thickness

        for obj_name, (pts, (long_cm, larg_cm)) in dimensions_cm.items():
            # Skip if no points (e.g., detection failed)
            if pts is None:
                continue

            # Convert points to int32 and draw the rotated rectangle
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img_out, [pts], isClosed=True, color=rect_color, thickness=thickness)

            # Compute the centroid of the rectangle for text placement
            center = np.mean(pts, axis=0).astype(int)
            cx, cy = center[0], center[1]

            # Build the dimension label
            text = f"{long_cm:.1f} x {larg_cm:.1f} cm"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(text, font, self.text_scale, self.text_thickness)

            # Draw a black rectangle behind the text for readability
            rect_x1 = cx - text_w // 2 - 5
            rect_y1 = cy - text_h // 2 - 5
            rect_x2 = cx + text_w // 2 + 5
            rect_y2 = cy + text_h // 2 + 5
            cv2.rectangle(img_out, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)

            # Draw the text in white
            cv2.putText(img_out, text,
                        (cx - text_w // 2, cy + text_h // 2),
                        font, self.text_scale, (255, 255, 255), self.text_thickness)

        return img_out

    def display_results(self, image: np.ndarray,
                        contours: Dict[str, Optional[np.ndarray]],
                        dimensions_cm: Dict[str, Tuple[List[Tuple[int, int]], Tuple[float, float]]]) -> np.ndarray:
        """
        Full visualization pipeline: overlay contours + dimension annotations.

        This is the primary method to call from the main application. It combines
        both contour overlay and dimension labels into a single annotated image.

        Args:
            image: Original BGR image.
            contours: Dictionary of contours (from segmentation engine).
            dimensions_cm: Dictionary of dimensions in centimeters (from DimensionCalculator).

        Returns:
            Fully annotated image ready for display or saving.
        """
        # First, overlay the filled and outlined contours
        annotated = self.display_contours(image, contours)
        # Then, add the oriented rectangles and dimension labels
        annotated = self.draw_dimensions(annotated, dimensions_cm)
        return annotated
