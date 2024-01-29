from RStask.InstanceSegmentation.model import SwinUPer
import torch
from skimage import io
from PIL import Image
import numpy as np
class SwinInstance:
    def __init__(self, device):
        print("Initializing InstanceSegmentation")
        self.model = SwinUPer()
        self.device = device
        try:
            trained = torch.load('./checkpoints/last_swint_upernet_finetune.pth')
        except:
            trained = torch.load('../../checkpoints/last_swint_upernet_finetune.pth')
        self.model.load_state_dict(trained["state_dict"])
        self.model = self.model.to(device)
        self.model.eval()
        self.mean, self.std = torch.tensor([123.675, 116.28, 103.53]).reshape((1, 3, 1, 1)), torch.tensor(
            [58.395, 57.12, 57.375]).reshape((1, 3, 1, 1))
        self.all_dict = {'plane': 1, 'ship': 2, 'storage tank': 3, 'baseball diamond': 4, 'tennis court': 5,
                         'basketball court': 6, 'ground track field': 7, 'harbor': 8, 'bridge': 9,
                         'large vehicle': 10, 'small vehicle': 11, 'helicopter': 12, 'roundabout': 13,
                         'soccer ball field': 14, 'swimming pool': 15}
    def inference(self, image_path, det_prompt ,updated_image_path):
        image = torch.from_numpy(io.imread(image_path))
        image = (image.permute(2, 0, 1).unsqueeze(0) - self.mean) / self.std
        with torch.no_grad():
            pred = self.model(image.to(self.device))
        pred = pred.argmax(1).cpu().squeeze().int().numpy()

        if det_prompt.strip().lower() in [i.strip().lower()  for i in self.all_dict.keys()]:
            idx=[i.replace(' ', '_').lower() for i in self.all_dict.keys()].index(det_prompt.strip().lower())+1
            pred=(pred==idx)*255
            pred = Image.fromarray(np.stack([pred, pred, pred], -1).astype(np.uint8))
            pred.save(updated_image_path)
            print(f"\nProcessed Instance Segmentation, Input Image: {image_path + ',' + det_prompt}, Output SegMap: {updated_image_path}")
            return updated_image_path
        else:
            print(f"\nCategory: { det_prompt} is not supported. Please use other tools.")
            return f"Category {det_prompt} is not supported. Please use other tools."



