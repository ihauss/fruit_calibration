from .base import SegmentationEngine, get_largest_contour
import cv2
import numpy as np

class HSVSegmentation(SegmentationEngine):
    def __init__(self,
                 mango_lower=(10, 50, 50), mango_upper=(50, 255, 255),
                 card_lower1=(0, 50, 50), card_upper1=(10, 255, 255),
                 card_lower2=(170, 50, 50), card_upper2=(179, 255, 255),
                 kernel_size=25):
        self.mango_lower = np.array(mango_lower)
        self.mango_upper = np.array(mango_upper)
        self.card_lower1 = np.array(card_lower1)
        self.card_upper1 = np.array(card_upper1)
        self.card_lower2 = np.array(card_lower2)
        self.card_upper2 = np.array(card_upper2)
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def get_masks(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Mango
        mask_mango = cv2.inRange(hsv, self.mango_lower, self.mango_upper)
        mask_mango = cv2.morphologyEx(mask_mango, cv2.MORPH_OPEN, self.kernel)
        # Card
        mask1 = cv2.inRange(hsv, self.card_lower1, self.card_upper1)
        mask2 = cv2.inRange(hsv, self.card_lower2, self.card_upper2)
        mask_card = cv2.bitwise_or(mask1, mask2)
        mask_card = cv2.morphologyEx(mask_card, cv2.MORPH_OPEN, self.kernel)
        mask_card = cv2.morphologyEx(mask_card, cv2.MORPH_CLOSE, self.kernel)
        return {'mango': mask_mango, 'card': mask_card}

    def get_contours(self, image):
        masks = self.get_masks(image)
        # (appel à une fonction utilitaire pour extraire le plus grand contour)
        return {k: get_largest_contour(v) for k, v in masks.items()}
