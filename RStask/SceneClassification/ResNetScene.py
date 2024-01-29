import torch
from skimage import io

class ResNetAID:
    def __init__(self, device=None):
        print("Initializing SceneClassification")
        from torchvision import models
        self.model = models.resnet34(pretrained=False, num_classes=30)
        self.device = device
        try:
            trained = torch.load('./checkpoints/Res34_AID_best.pth')
        except:
            trained = torch.load('../../checkpoints/Res34_AID_best.pth')

        self.model.load_state_dict(trained)
        self.model = self.model.to(device)
        self.model.eval()
        self.mean, self.std = torch.tensor([123.675, 116.28, 103.53]).reshape((1, 3, 1, 1)), torch.tensor(
            [58.395, 57.12, 57.375]).reshape((1, 3, 1, 1))
        self.all_dict = {'Bridge': 0, 'Medium Residential': 1, 'Park': 2, 'Stadium': 3, 'Church': 4,
                         'Dense Residential': 5, 'Farmland': 6,
                         'River': 7, 'School': 8, 'Sparse Residential': 9, 'Viaduct': 10, 'Beach': 11, 'Forest': 12,
                         'Baseball Field': 13, 'Desert': 14, 'BareLand': 15,
                         'Railway Station': 16, 'Center': 17, 'Industrial': 18, 'Meadow': 19, 'Airport': 20,
                         'Storage Tanks': 21, 'Pond': 22, 'Commercial': 23, 'Resort': 24,
                         'Parking': 25, 'Port': 26, 'Square': 27, 'Mountain': 28, 'Playground': 29}


    def inference(self, inputs):
        image_path = inputs
        image = torch.from_numpy(io.imread(image_path))
        image = (image.permute(2, 0, 1).unsqueeze(0) - self.mean) / self.std
        with torch.no_grad():
            pred = self.model(image.to(self.device))

        values, indices = torch.softmax(pred, 1).topk(2, dim=1, largest=True, sorted=True)
        output_txt = image_path + ' has ' + str(
            torch.round(values[0][0] * 10000).item() / 100) + '% probability being ' + list(self.all_dict.keys())[
                         indices[0][0]] + ' and ' + str(
            torch.round(values[0][1] * 10000).item() / 100) + '% probability being ' + list(self.all_dict.keys())[
                         indices[0][1]]+'.'
        print(f"\nProcessed Scene Classification, Input Image: {inputs}, Output Scene: {output_txt}")
        return output_txt