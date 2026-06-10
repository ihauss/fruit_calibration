import cv2
import numpy as np

def get_mango_mask(image):
    """Masque pour la mangue (jaune) avec ouverture."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Plages jaune (ajustez selon votre éclairage)
    lower = np.array([10, 50, 50])
    upper = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # Ouverture : érosion puis dilatation pour supprimer le bruit
    kernel = np.ones((25,25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def get_card_mask(image):
    """Masque pour la carte (rouge) avec ouverture."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Le rouge est aux extrémités de la teinte (0-10 et 170-179)
    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 50, 50])
    upper2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    # Ouverture
    kernel = np.ones((25,25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def get_largest_contour(mask):
    """Retourne le plus grand contour du masque (supposé être l'objet principal)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def get_contours(image):
    contours = {}
    mango_mask = get_mango_mask(image)
    card_mask = get_card_mask(image)
    contours["mango"] = get_largest_contour(mango_mask)
    contours["card"] = get_largest_contour(card_mask)
    return contours
