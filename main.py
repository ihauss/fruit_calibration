import cv2

from src.display import display_results
from src.dimension import get_dimension
from src.segment_color import get_contours

if __name__ == "__main__":
    while(True):
        image = cv2.imread("./data/IMG_20260609_182110.jpg")
        contours = get_contours(image)
        dimensions_cm = get_dimension(contours)
        display = display_results(image, contours, dimensions_cm)
        cv2.imshow("Masque", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
