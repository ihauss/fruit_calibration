# Fruit Calibration – Mango Size Measurement

A computer vision tool that measures the length and width of a mango (in centimeters) using a red credit‑card as a scale reference.  
The segmentation is based on HSV color filtering, and the results are displayed directly on the image.

![Original image](./data/IMG_20260609_182110.jpg)  
*Figure 1: Original image with mango and red card*

![Final annotated image](./output.png)  
*Figure 2: Detected objects with dimensions in cm*

## Features

- **HSV segmentation** for a yellow mango and a red card.
- **Morphological opening** to clean the masks.
- **Oriented bounding boxes** (`cv2.minAreaRect`) to get precise length/width in pixels.
- **Automatic scaling** using a real credit‑card size (85.6 mm × 53.98 mm).
- **Visual output** – rectangles, filled masks, and dimensions in cm (white text on black background, centered).

## Project Structure
