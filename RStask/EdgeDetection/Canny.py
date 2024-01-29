from PIL import Image
import cv2
import numpy as np
class Image2Canny:
    def __init__(self):
        print("Initializing Image2Canny")
        self.low_threshold = 100
        self.high_threshold = 200

    def inference(self, inputs,new_image_name):
        image = Image.open(inputs)
        image = np.array(image)
        canny = cv2.Canny(image, self.low_threshold, self.high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        updated_image_path = new_image_name
        canny.save(updated_image_path)
        print(f"\nProcessed Image2Canny, Input Image: {inputs}, Output Text: {updated_image_path}")
        return None