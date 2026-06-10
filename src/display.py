import cv2
import numpy as np

def display_contour(image, contour, color=(0, 255, 0), thickness=15, fill=False):
    """
    Affiche un contour sur une copie de l'image.
    - fill=True : remplit l'intérieur du contour (utile pour masque)
    """
    img_copy = image.copy()
    if fill:
        cv2.drawContours(img_copy, [contour], -1, color, thickness=cv2.FILLED)
    else:
        cv2.drawContours(img_copy, [contour], -1, color, thickness)
    return img_copy

def display_contours(image, contours):
    display = image.copy()
    image_cp = image.copy()
    display = display_contour(display, contours["mango"], color=(255, 0, 0), fill=True)
    display = display_contour(display, contours["card"], color=(255, 255, 0), fill=True)
    image_cp = display_contour(image_cp, contours["mango"], color=(255, 0, 0))
    image_cp = display_contour(image_cp, contours["card"], color=(255, 255, 0))
    display = cv2.addWeighted(display, 0.4, image_cp, 0.6, 5)
    return display

def draw_dimensions_cm(image, dimensions_cm, rect_color=(0, 255, 0), thickness=15):
    """
    Dessine les rectangles orientés des objets et affiche leurs dimensions en cm
    (blanc sur fond noir) au centre de chaque forme.
    
    Paramètres:
        image (numpy.ndarray): Image sur laquelle dessiner (copie modifiée).
        dimensions_cm (dict): Dictionnaire avec clés 'mango' et 'card'.
                              Chaque valeur est (pts, (longueur_cm, largeur_cm)).
        rect_color (tuple): Couleur du rectangle (BGR).
        thickness (int): Épaisseur des traits.
    
    Retourne:
        numpy.ndarray: Image annotée.
    """
    img_out = image.copy()
    
    for obj_name, (pts, (long_cm, larg_cm)) in dimensions_cm.items():
        # pts est un numpy array de shape (4,2) issu de cv2.boxPoints
        pts = np.array(pts, dtype=np.int32)
        # Dessiner le rectangle orienté
        cv2.polylines(img_out, [pts], isClosed=True, color=rect_color, thickness=thickness)
        
        # Calcul du centre (moyenne des quatre points)
        center = np.mean(pts, axis=0).astype(int)
        cx, cy = center[0], center[1]
        
        # Préparer le texte
        text = f"{long_cm:.1f} x {larg_cm:.1f} cm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness=2)
        
        # Coordonnées du rectangle noir de fond
        rect_x1 = cx - text_w // 2 - 5
        rect_y1 = cy - text_h // 2 - 5
        rect_x2 = cx + text_w // 2 + 5
        rect_y2 = cy + text_h // 2 + 5
        
        # Dessiner le fond noir
        cv2.rectangle(img_out, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        # Dessiner le texte blanc (centré)
        cv2.putText(img_out, text, (cx - text_w // 2, cy + text_h // 2),
                    font, font_scale, (255, 255, 255), 5)
    
    return img_out


def display_results(image, contours, dimensions_cm):
    display = display_contours(image, contours)
    display = draw_dimensions_cm(display, dimensions_cm)
    return display
