"""
src/display.py

Handles visualization of contours and dimensions on images.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

class Visualizer:
    """
    Visualizes segmentation contours and dimension annotations.
    """

    def __init__(self,
                 mango_color=(255, 0, 0),      # BGR for mango (red)
                 card_color=(255, 255, 0),     # BGR for card (cyan)
                 rect_color=(0, 255, 0),       # BGR for rectangle
                 contour_thickness=15,
                 text_scale=2.0,
                 text_thickness=5,
                 fill_opacity=0.4,
                 outline_weight=0.6):
        """
        Args:
            mango_color: BGR color for mango fill.
            card_color: BGR color for card fill.
            rect_color: BGR color for the oriented rectangle.
            contour_thickness: thickness of contour lines (or cv2.FILLED for fill).
            text_scale: Font scale for dimension labels.
            text_thickness: Thickness of dimension text.
            fill_opacity: Opacity weight for filled regions.
            outline_weight: Weight for outline overlay.
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
        """
        img_copy = image.copy()
        if fill:
            cv2.drawContours(img_copy, [contour], -1, color, thickness=cv2.FILLED)
        else:
            cv2.drawContours(img_copy, [contour], -1, color, self.contour_thickness)
        return img_copy

    def display_contours(self, image: np.ndarray,
                         contours: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
        """
        Overlay filled contours and outlines for mango and card.

        Args:
            image: Original image (BGR).
            contours: Dictionary with keys 'mango' and 'card' holding contours (or None).

        Returns:
            Image with contours overlaid.
        """
        # Handle missing contours
        mango_contour = contours.get('mango')
        card_contour = contours.get('card')

        # Create base display with filled regions
        display = image.copy()
        if mango_contour is not None:
            display = self._draw_contour(display, mango_contour, self.mango_color, fill=True)
        if card_contour is not None:
            display = self._draw_contour(display, card_contour, self.card_color, fill=True)

        # Create overlay with outlines only
        outline_img = image.copy()
        if mango_contour is not None:
            outline_img = self._draw_contour(outline_img, mango_contour, self.mango_color, fill=False)
        if card_contour is not None:
            outline_img = self._draw_contour(outline_img, card_contour, self.card_color, fill=False)

        # Blend filled and outline
        blended = cv2.addWeighted(display, self.fill_opacity,
                                  outline_img, self.outline_weight, 5)
        return blended

    def draw_dimensions(self, image: np.ndarray,
                        dimensions_cm: Dict[str, Tuple[List[Tuple[int, int]], Tuple[float, float]]],
                        rect_color: Optional[Tuple[int, int, int]] = None,
                        thickness: Optional[int] = None) -> np.ndarray:
        """
        Draw oriented rectangles and dimension labels (white on black background).

        Args:
            image: Image to annotate.
            dimensions_cm: Dictionary with 'mango' and 'card' keys.
                           Each value is (pts, (length_cm, width_cm)).
            rect_color: Color for rectangle (if None, uses self.rect_color).
            thickness: Thickness for rectangle (if None, uses self.contour_thickness).

        Returns:
            Annotated image.
        """
        img_out = image.copy()
        rect_color = rect_color if rect_color is not None else self.rect_color
        thickness = thickness if thickness is not None else self.contour_thickness

        for obj_name, (pts, (long_cm, larg_cm)) in dimensions_cm.items():
            if pts is None:
                continue
            pts = np.array(pts, dtype=np.int32)
            # Draw oriented rectangle
            cv2.polylines(img_out, [pts], isClosed=True, color=rect_color, thickness=thickness)

            # Center
            center = np.mean(pts, axis=0).astype(int)
            cx, cy = center[0], center[1]

            # Prepare text
            text = f"{long_cm:.1f} x {larg_cm:.1f} cm"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(text, font, self.text_scale, self.text_thickness)

            # Black background rectangle
            rect_x1 = cx - text_w // 2 - 5
            rect_y1 = cy - text_h // 2 - 5
            rect_x2 = cx + text_w // 2 + 5
            rect_y2 = cy + text_h // 2 + 5
            cv2.rectangle(img_out, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)

            # White text
            cv2.putText(img_out, text,
                        (cx - text_w // 2, cy + text_h // 2),
                        font, self.text_scale, (255, 255, 255), self.text_thickness)

        return img_out

    def display_results(self, image: np.ndarray,
                        contours: Dict[str, Optional[np.ndarray]],
                        dimensions_cm: Dict[str, Tuple[List[Tuple[int, int]], Tuple[float, float]]]) -> np.ndarray:
        """
        Full visualization: contours overlay + dimension annotations.

        Args:
            image: Original BGR image.
            contours: Contours from segmentation.
            dimensions_cm: Dimensions in cm.

        Returns:
            Fully annotated image.
        """
        # Draw contours (filled + outlines)
        annotated = self.display_contours(image, contours)
        # Add dimension labels
        annotated = self.draw_dimensions(annotated, dimensions_cm)
        return annotated
