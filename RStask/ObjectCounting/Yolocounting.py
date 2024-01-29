from RStask.ObjectDetection.models.common import DetectMultiBackend
import torch
from skimage import io
import numpy as np
import torchvision
import torch.nn.functional as F
class YoloCounting:
    def __init__(self, device):
        from RStask.ObjectDetection.models.common import DetectMultiBackend
        self.device = device
        try:
            self.model = DetectMultiBackend('./checkpoints/yolov5_best.pt', device=torch.device(device), dnn=False, fp16=False)
        except:
            self.model = DetectMultiBackend('../../checkpoints/yolov5_best.pt', device=torch.device(device), dnn=False,fp16=False)
        self.category = ['small vehicle', 'large vehicle', 'plane', 'storage tank', 'ship', 'harbor',
                         'ground track field',
                         'soccer ball field', 'tennis court', 'swimming pool', 'baseball diamond', 'roundabout',
                         'basketball court', 'bridge', 'helicopter']


    def inference(self, image_path, det_prompt):
        supported_class=False
        for i in range(len(self.category)):
            if self.category[i] == det_prompt or self.category[i] == det_prompt[:-1] or self.category[i] == det_prompt[:-3]:
                supported_class=True
        if supported_class is False:
            log_text=det_prompt+' is not a supported category for the model.'
            print(f"\nProcessed Object Counting, Input Image: {image_path}, Output text: {log_text}")
            return log_text

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
        log_text = ''

        for i in range(len(self.category)):
            if (detection_classes == i).sum() > 0 and (
                    self.category[i] == det_prompt or self.category[i] == det_prompt[:-1] or self.category[
                i] == det_prompt[:-3]):
                log_text += str((detection_classes == i).sum()) + ' ' + self.category[i] + ','
        if log_text != '':
            log_text = log_text[:-1] + ' detected.'
        else:
            log_text = 'No ' + self.category[i] + ' detected.'

        print(f"\nProcessed Object Counting, Input Image: {image_path}, Output text: {log_text}")
        return log_text

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
