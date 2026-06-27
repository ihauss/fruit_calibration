import cv2

from src.segmentation.factory import SegmentationFactory
from src.dimension import DimensionCalculator
from src.display import Visualizer

# Création du moteur
#engine = SegmentationFactory.create(engine_type="unet", checkpoint_path="src/models/Unet-98.ckpt")
engine = SegmentationFactory.create(engine_type="hsv")
dim_calc = DimensionCalculator()
vis = Visualizer(contour_thickness=15, text_scale=2.0, text_thickness=5)

if __name__ == "__main__":
    while(True):
        image = cv2.imread("./data/images/IMG_20260609_182110.jpg")
        contours = engine.get_contours(image)
        dimensions_cm = dim_calc.compute_dimensions(contours)
        display = vis.display_results(image, contours, dimensions_cm)
        cv2.imshow("Masque", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
