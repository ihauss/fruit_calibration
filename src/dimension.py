import cv2
import numpy as np

def px_dimensions(contour):
    rect = cv2.minAreaRect(contour)
    (w, h) = rect[1]
    longueur = max(w, h)
    largeur = min(w, h)
    
    # Obtenir les 4 coins
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)  # conversion robuste en int32
    
    # Tracer les 4 côtés
    pts = []
    for i in range(4):
        pts.append(tuple(box[i]))

    return pts, (longueur, largeur)

def px2cm(dimensions_px):
    """
    Convertit les dimensions en pixels en centimètres en utilisant la carte comme référence.
    
    Paramètres:
        dimensions_px (dict): Dictionnaire avec clés "mango" et "card".
                              Chaque valeur est un tuple (pts, (longueur_px, largeur_px))
    
    Retourne:
        dict: Même structure avec les dimensions en cm (longueur_cm, largeur_cm)
    """
    # Dimensions réelles de la carte de crédit (cm)
    CARD_LONGUEUR_CM = 8.56    # 85.6 mm
    CARD_LARGEUR_CM  = 5.398   # 53.98 mm
    
    # Extraire les dimensions en pixels de la carte
    _, (card_long_px, card_larg_px) = dimensions_px["card"]
    
    # Calculer les facteurs d'échelle (pixels par cm)
    scale_long = card_long_px / CARD_LONGUEUR_CM   # pixels/cm pour la longueur
    scale_larg = card_larg_px / CARD_LARGEUR_CM    # pixels/cm pour la largeur
    
    # Dictionnaire des résultats
    dimensions_cm = {}
    
    for key in ["mango", "card"]:
        pts, (long_px, larg_px) = dimensions_px[key]
        if key == "mango":
            # Appliquer les facteurs correspondants
            long_cm = long_px / scale_long
            larg_cm = larg_px / scale_larg
        else:  # "card" : on peut aussi vérifier la cohérence
            long_cm = long_px / scale_long   # devrait donner ~8.56
            larg_cm = larg_px / scale_larg   # devrait donner ~5.398
        dimensions_cm[key] = (pts, (long_cm, larg_cm))
    
    return dimensions_cm

def get_dimension(contours):
    dimensions_px = {}
    dimensions_px["mango"] = px_dimensions(contours["mango"])
    dimensions_px["card"] = px_dimensions(contours["card"])
    dimensions_cm = px2cm(dimensions_px)
    return dimensions_cm
