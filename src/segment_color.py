import cv2
import numpy as np
import torch
from albumentations import Compose, Normalize, Resize, ToTensorV2
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights=None,  # pas besoin de poids pré-entraînés pour l'inférence
    in_channels=3,
    classes=3
)

mode = "unet"
#mode = "hsv"

def smooth_mask_morphological(mask, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Fermeture : comble les petits trous
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Ouverture : enlève les petits bruits
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return cleaned

def smooth_mask_gaussian(mask, kernel_size=15, sigma=1.0):
    # Le masque doit être en float pour le flou
    mask_float = mask.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), sigma)
    # Seuillage pour revenir au binaire (seuil à 0.5 par exemple)
    _, smoothed = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)
    return (smoothed * 255).astype(np.uint8)

def smooth_masks(mask1, mask2):
    mask1 = smooth_mask_morphological(mask1)
    mask2 = smooth_mask_morphological(mask2)
    mask1 = smooth_mask_gaussian(mask1)
    mask2 = smooth_mask_gaussian(mask2)
    return mask1, mask2

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

def remove_model_prefix(state_dict):
    return {key[6:] if key.startswith('model.') else key: value for key, value in state_dict.items()}


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

def preprocess_image(image, target_size=(256, 256)):
    # Lire l'image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]  # (H, W)

    # Transformer
    transform = Compose([
        Resize(*target_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    transformed = transform(image=image)
    tensor = transformed["image"].unsqueeze(0)  # ajouter la dimension batch

    return tensor, original_size, image

def predict_mask(model, image_tensor, original_size, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)          # shape: [1, 3, 256, 256]
        pred = output.argmax(dim=1)           # [1, 256, 256]
        pred = pred.squeeze(0).cpu().numpy()  # [256, 256]

    # Redimensionner à la taille originale (utiliser NEAREST pour préserver les classes)
    mask_original = cv2.resize(
        pred.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST
    )
    return mask_original

def get_contours(image):
    contours = {}
    if mode == "hsv":
        mango_mask = get_mango_mask(image)
        card_mask = get_card_mask(image)
    elif mode == "unet":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load("src/models/Unet-98.ckpt", map_location=device)
        state_dict = remove_model_prefix(checkpoint['state_dict'])
        model.load_state_dict(state_dict)  # selon votre format Lightning
        model.to(device)
        model.eval()
        tensor, original_size, original_image = preprocess_image(image)
        mask = predict_mask(model, tensor, original_size, device)
        mango_mask = (mask == 1).astype(np.uint8) * 255
        card_mask = (mask == 2).astype(np.uint8) * 255
    mango_mask, card_mask = smooth_masks(mango_mask, card_mask) 
    contours["mango"] = get_largest_contour(mango_mask)
    contours["card"] = get_largest_contour(card_mask)
    return contours
