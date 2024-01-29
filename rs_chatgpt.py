import os
import random

import openai
import torch
import re
import uuid
from PIL import Image
from skimage import io
import argparse
import inspect
import time
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

import cv2
import numpy as np

import torchvision
import torch.nn.functional as F

RS_CHATGPT_PREFIX = """Remote Sensing ChatGPT is designed to assist with a wide range of remote sensing image related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of remote sensing applications. Remote Sensing ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Remote Sensing ChatGPT can process and understand large amounts of  remote sensing images, knowledge, and text. As a expertized language model, Remote Sensing ChatGPT can not directly read remote sensing images, but it has a list of tools to finish different remote sensing tasks. Each input remote sensing image will have a file name formed as "image/xxx.png", and Remote Sensing ChatGPT can invoke different tools to indirectly understand the remote sensing image. When talking about images, Remote Sensing ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Remote Sesning ChatGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Remote Sensing ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new remote sensing images to Remote Sensing ChatGPT with a description. The description helps Remote Sensing ChatGPT to understand this image, but Remote Sensing ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Remote Sensing ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of remote sensing tasks and provide valuable insights and information on a wide range of remote sensing applicatinos. 


TOOLS:
------

Remote Sensing ChatGPT  has access to the following tools:"""

RS_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

RS_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Remote Sensing ChatGPT is a text language model, Remote Sensing ChatGPT must use tools to observe remote sensing images rather than imagination.
The thoughts and observations are only visible for Remote Sensing ChatGPT, Remote Sensing ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""

os.makedirs('image', exist_ok=True)

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]

    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}.png'.replace('__','_')
    return os.path.join(head, new_file_name)


class Image2Canny:
    def __init__(self, device):
        print("Initializing Image2Canny")
        self.low_threshold = 100
        self.high_threshold = 200

    @prompts(name="Edge Detection On Image",
             description="useful when you want to detect the edge of the remote sensing image. "
                         "like: detect the edges of this image, or canny detection on image, "
                         "or perform edge detection on this image, or detect the canny image of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        image = np.array(image)
        canny = cv2.Canny(image, self.low_threshold, self.high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        updated_image_path = get_new_image_name(inputs, func_name="edge")
        canny.save(updated_image_path)
        print(f"\nProcessed Image2Canny, Input Image: {inputs}, Output Text: {updated_image_path}")
        return updated_image_path


class ObjectCounting:
    def __init__(self, device):
        from ObjectDetection.models.common import DetectMultiBackend
        self.model = DetectMultiBackend('./checkpoints/yolov5_best.pt', device=torch.device(device), dnn=False,
                                        data='dota_data/dota_name.yaml', fp16=False)

        self.category = ['small vehicle', 'large vehicle', 'plane', 'storage tank', 'ship', 'harbor',
                         'ground track field',
                         'soccer ball field', 'tennis court', 'swimming pool', 'baseball diamond', 'roundabout',
                         'basketball court', 'bridge', 'helicopter']

    @prompts(name="Count object",
             description="useful when you want to the number of the certain object in the remote sensing image. "
                         "like: count the number of bridges, or how many planes are there in the image, "
                         "or perform vehicle on this image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be found")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        image = torch.from_numpy(io.imread(image_path))
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        _, _, h, w = image.shape
        with torch.no_grad():
            out, _ = self.model(F.interpolate(image.to(self.device), size=(640, 640), mode='bilinear'), augment=False,
                                val=True)
            predn = self.non_max_suppression(out, conf_thres=0.001, iou_thres=0.75, labels=[], multi_label=True,
                                             agnostic=False)[0]
            detections = predn.clone().cpu()
            detections = detections[predn[:, 4] > 0.75]
            # detections_box = (detections[:, :4] / (640 / h)).int().numpy()
            detection_classes = detections[:, 5].int().numpy()
        log_text = 'Results:'
        if len(detection_classes) > 0:
            det = np.zeros((h, w, 3))
            for i in range(len(self.category)):
                if (detection_classes == i).sum() > 0:
                    log_text += str((detection_classes == i).sum()) + ' ' + self.category[i] + ','
            log_text = log_text[:-1] + ' detected.'
        log_text = log_text + ' 0 objects detected for other categories.'
        print(f"\nProcessed Object Counting, Input Image: {inputs}, Output text: {log_text}")
        return log_text


class InstanceSegmentation:
    def __init__(self, device):
        print("Initializing InstanceSegmentation")
        from InstanceSegmentation.model import SwinUPer
        self.model = SwinUPer()
        self.device = device
        trained = torch.load('./checkpoints/last_swint_upernet_finetune.pth')
        self.model.load_state_dict(trained["state_dict"])
        self.model = self.model.to(device)
        self.model.eval()
        self.mean, self.std = torch.tensor([123.675, 116.28, 103.53]).reshape((1, 3, 1, 1)), torch.tensor(
            [58.395, 57.12, 57.375]).reshape((1, 3, 1, 1))
        self.all_dict = {'plane': 1, 'ship': 2, 'storage tank': 3, 'baseball diamond': 4, 'tennis court': 5,
                         'basketball court': 6, 'ground track field': 7, 'harbor': 8, 'bridge': 9,
                         'large vehicle': 10, 'small vehicle': 11, 'helicopter': 12, 'roundabout': 13,
                         'soccer ball field': 14, 'swimming pool': 15}

    @prompts(name="Instance Segmentation for Remote Sensing Image",
             description="useful when you want to apply man-made instance segmentation for the image. The expected input category include plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, vehicle, helicopter, roundabout, soccer ball field, and swimming pool."
                         "like: extract plane from this image, "
                         "or predict the ship in this image, or extract tennis court from this image, segment harbor from this image, Extract the vehicle in the image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text of the category,selected from plane, or ship, or storage tank, or baseball diamond, or tennis court, or basketball court, or ground track field, or harbor, or bridge, or vehicle, or helicopter, or roundabout, or soccer ball field, or  swimming pool. ")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        image = torch.from_numpy(io.imread(image_path))
        image = (image.permute(2, 0, 1).unsqueeze(0) - self.mean) / self.std
        with torch.no_grad():
            pred = self.model(image.to(self.device))
        pred = pred.argmax(1).cpu().squeeze().int().numpy()
        pred = Image.fromarray(np.stack([pred, pred, pred], -1).astype(np.uint8))
        updated_image_path = get_new_image_name(image_path, func_name="instance_" + det_prompt)
        pred.save(updated_image_path)
        print(f"\nProcessed Instance Segmentation, Input Image: {inputs}, Output SegMap: {updated_image_path}")
        return updated_image_path


class SceneClassification:
    def __init__(self, device):
        print("Initializing SceneClassification")
        from torchvision import models
        self.model = models.resnet34(pretrained=False, num_classes=30)
        self.device = device
        trained = torch.load('./checkpoints/Res34_AID_best.pth')
        self.model.load_state_dict(trained)
        self.model = self.model.to(device)
        self.model.eval()
        self.mean, self.std = torch.tensor([123.675, 116.28, 103.53]).reshape((1, 3, 1, 1)), torch.tensor(
            [58.395, 57.12, 57.375]).reshape((1, 3, 1, 1))
        self.all_dict = {'Bridge': 0, 'MediumResidential': 1, 'Park': 2, 'Stadium': 3, 'Church': 4,
                         'DenseResidential': 5, 'Farmland': 6,
                         'River': 7, 'School': 8, 'SparseResidential': 9, 'Viaduct': 10, 'Beach': 11, 'Forest': 12,
                         'BaseballField': 13, 'Desert': 14, 'BareLand': 15,
                         'RailwayStation': 16, 'Center': 17, 'Industrial': 18, 'Meadow': 19, 'Airport': 20,
                         'StorageTanks': 21, 'Pond': 22, 'Commercial': 23, 'Resort': 24,
                         'Parking': 25, 'Port': 26, 'Square': 27, 'Mountain': 28, 'Playground': 29}

    @prompts(name="Scene Classification for Remote Sensing Image",
             description="useful when you want to know the type of scene or function for the image. "
                         "like: what is the category of this image?, "
                         "or classify the scene of this image, or predict the scene category of this image, or what is the function of this image. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, inputs):
        image_path = inputs
        image = torch.from_numpy(io.imread(image_path))
        image = (image.permute(2, 0, 1).unsqueeze(0) - self.mean) / self.std
        with torch.no_grad():
            pred = self.model(image.to(self.device))
        # pred= pred.argmax(1).cpu().squeeze().int().numpy()
        values, indices = torch.softmax(pred, 1).topk(2, dim=1, largest=True, sorted=True)
        output_txt = image_path + ' has ' + str(
            torch.round(values[0][0] * 10000).item() / 100) + '% probability being ' + list(self.all_dict.keys())[
                         indices[0][0]] + ' and ' + str(
            torch.round(values[0][1] * 10000).item() / 100) + '% probability being ' + list(self.all_dict.keys())[
                         indices[0][1]]+'.'

        print(f"\nProcessed Scene Classification, Input Image: {inputs}, Output Scene: {output_txt}")
        return output_txt


class LandUseSegmentation:
    def __init__(self, device):
        print("Initializing LandUseSegmentation")
        from LandUseClassification.seg_hrnet import HRNet48
        self.model = HRNet48()
        self.device = device
        trained = torch.load('./checkpoints/HRNET_LoveDA_best.pth')
        # rename = {k.replace('backbone.', 'model.').replace('decode_head.', 'model.'): v for k, v in
        #           trained['state_dict'].items()}
        self.model.load_state_dict(trained)
        self.model = self.model.to(device)
        self.model.eval()
        self.mean, self.std = torch.tensor([123.675, 116.28, 103.53]).reshape((1, 3, 1, 1)), torch.tensor(
            [58.395, 57.12, 57.375]).reshape((1, 3, 1, 1))

    @prompts(name="Land Use Segmentation for Remote Sensing Image",
             description="useful when you want to apply land use gegmentation for the image. The expected input category include Lnad Use, Building, Road, Water, Barren, Forest, and Architecture."
                         "like: generate land use map from this image, "
                         "or predict the land use on this image, or extract building from this image, segment roads from this image, Extract the water bodies in the image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text of the category,selected from Lnad Use, or Building, or Road, or Water, or Barren, or Forest, or and Architecture. ")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        image = torch.from_numpy(io.imread(image_path))
        image = (image.permute(2, 0, 1).unsqueeze(0) - self.mean) / self.std
        with torch.no_grad():
            pred = self.model(image.to(self.device))
        pred = pred.argmax(1).cpu().squeeze().int().numpy()
        pred = Image.fromarray(np.stack([pred, pred, pred], -1).astype(np.uint8))
        updated_image_path = get_new_image_name(image_path, func_name="landuse")
        pred.save(updated_image_path)
        print(f"\nProcessed Landuse Classification, Input Image: {inputs}, Output SegMap: {updated_image_path}")
        return updated_image_path


class ObjectDetection:
    def __init__(self, device):
        self.device = device
        from ObjectDetection.models.common import DetectMultiBackend
        self.model = DetectMultiBackend('./checkpoints/yolov5_best.pt', device=torch.device(device), dnn=False,
                                        data='dota_data/dota_name.yaml', fp16=False)

        self.category = ['small vehicle', 'large vehicle', 'plane', 'storage tank', 'ship', 'harbor',
                         'ground track field',
                         'soccer ball field', 'tennis court', 'swimming pool', 'baseball diamond', 'roundabout',
                         'basketball court', 'bridge', 'helicopter']

    @prompts(name="Detect the given object",
             description="useful when you only want to detect the bounding box of the certain objects in the picture"
                         "according to the given text"
                         "like: detect the cat,"
                         "or can you detect an object for me, or can you locate an object for me."
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be found")

    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        image = torch.from_numpy(io.imread(image_path))
        image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
        _, _, h, w = image.shape
        with torch.no_grad():
            out, _ = self.model(image.to(self.device), augment=False,val=True)
            predn = self.non_max_suppression(out, conf_thres=0.001, iou_thres=0.75, labels=[], multi_label=True,
                                             agnostic=False)[0]
            detections = predn.clone()
            detections = detections[predn[:, 4] > 0.75]
            detections_box = (detections[:, :4] / (640 / h)).int().cpu().numpy()
            detection_classes = detections[:, 5].int().cpu().numpy()
        if len(detection_classes) > 0:

            det = np.zeros((h, w, 3))
            for i in range(len(detections_box)):
                x1, y1, x2, y2 = detections_box[i]
                det[y1:y2, x1:x2] = detection_classes[i] + 1
            log_text = 'Results:'
            for i in range(len(self.category)):
                if (detection_classes == i).sum() > 0:
                    log_text += str((detection_classes == i).sum()) + ' ' + self.category[i] + ','
            log_text = log_text[:-1] + ' detected.'
            updated_image_path = get_new_image_name(image_path, func_name="detection_" + det_prompt.replace(' ', '_'))
            self.visualize(image_path,updated_image_path,detections)
            # pred = Image.fromarray(det).astype(np.uint8).save(updated_image_path)
            # pred.save(updated_image_path)
            print(
                f"\nProcessed Object Detection, Input Image: {inputs}, Output Bounding box: {updated_image_path},Output text: {log_text}")
            return updated_image_path + ',' + log_text
    def visualize(self,image_path, newpic_path,detections):
        font = cv2.FONT_HERSHEY_SIMPLEX
        im = io.imread(image_path)
        boxes = detections.int().cpu().numpy()
        for i in range(len(boxes)):
            cv2.rectangle(im, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 255), 2)
            cv2.rectangle(im, (boxes[i][0], boxes[i][1] - 15), (boxes[i][0] + 45, boxes[i][1] - 2), (0, 0, 255),thickness=-1)
            cv2.putText(im, self.category[boxes[i][-1]], (boxes[i][0], boxes[i][1] - 2), font, 0.5, (255, 255, 255),1)
        Image.fromarray(im.astype(np.uint8)).save(newpic_path)
        with open(newpic_path[:-4]+'.txt','w') as f:
            for i in range(len(boxes)):
                f.write(str(list(boxes[i,:4]))[1:-1]+', '+self.category[boxes[i][-1]]+'\n')
    def non_max_suppression(self, prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=300):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        def box_iou(box1, box2):
            def box_area(box):
                # box = xyxy(4,n)
                return (box[2] - box[0]) * (box[3] - box[1])

            # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
            """
            Return intersection-over-union (Jaccard index) of boxes.
            Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
            Arguments:
                box1 (Tensor[N, 4])
                box2 (Tensor[M, 4])
            Returns:
                iou (Tensor[N, M]): the NxM matrix containing the pairwise
                    IoU values for every element in boxes1 and boxes2
            """

            # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
            (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
            inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

            # IoU = inter / (area1 + area2 - inter)
            return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)

        def xywh2xyxy(x):
            # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

            y = x.clone()
            y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
            y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
            y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
            y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
            return y

        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.1 + 0.03 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]

        return output


class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype).to(self.device)

    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = 'A satellite image of ' + self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions

class RSChatGPT:
    def __init__(self, gpt_name,load_dict,openai_key):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing RSChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict or 'SceneClassification' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for RSChatGPT")

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if
                                           k != 'self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})

        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.llm = OpenAI(model_name=gpt_name,temperature=0,openai_api_key=openai_key)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def initialize(self):
        self.memory.clear()  # clear previous history
        PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = RS_CHATGPT_PREFIX, RS_CHATGPT_FORMAT_INSTRUCTIONS, RS_CHATGPT_SUFFIX
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX}, )

    def run_text(self, text, state):
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state
    def init_image(self, image_dir, state):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        img = io.imread(image_dir)
        width, height = img.shape[1],img.shape[0]
        ratio = min(640 / width, 640 / height)
        if ratio<1:
            width_new, height_new = (round(width * ratio), round(height * ratio))
        else:
            width_new, height_new =width,height
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64

        if width_new!=width or height_new!=height:
            img = cv2.resize(img,(width_new, height_new))
            print(f"======>Auto Resizing Image from {height,width} to {height_new,width_new}...")
        else:
            print(f"======>Auto Renaming Image...")
        io.imsave(image_filename, img.astype(np.uint8))
        scene_prior = self.models['SceneClassification'].inference(image_filename)
        caption_prior = self.models['ImageCaptioning'].inference(image_filename)
        Human_prompt = f' Provide a remote sensing image named {image_filename}.The description is: {caption_prior}. {scene_prior} This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\".'
        AI_prompt = "Received."
        self.memory.chat_memory.add_user_message(Human_prompt)
        self.memory.chat_memory.add_ai_message(AI_prompt)

        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('openai_key', type=str)
    parser.add_argument('--image-dir', type=str,default="./test_image.png")
    parser.add_argument('--gpt_name', type=str, default="gpt-3.5-turbo",choices=['gpt-3.5-turbo','gpt-4'])
    parser.add_argument('--load', type=str,help='Image Captioning and Scene Classification are basic models that are required. You can select from [ObjectDetection,LandUseSegmentation,InstanceSegmentation,Image2Canny]',
                        default="ImageCaptioning_cuda:0,SceneClassification_cuda:0,ObjectDetection_cuda:0,LandUseSegmentation_cuda:0,InstanceSegmentation_cuda:0,SceneClassification_cuda:0,Image2Canny_cpu")
    #,ObjectDetection_cuda:0,LandUseClassfication_cuda:0,InstanceSegmentation_cuda:0,SceneClassification_cuda:0,Image2Canny_cpu
    args = parser.parse_args()
    state = []
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    bot = RSChatGPT(gpt_name=args.gpt_name,load_dict=load_dict,openai_key=args.openai_key)
    bot.initialize()
    state=bot.init_image(args.image_dir,state)
    print('RSChatGPT initialization done, you can now chat with RSChatGPT~')

    while 1:
        txt = input('You can now input your question?(e.g. How many planes are there in the image?)\n')
        state = bot.run_text(txt,state)
