from RStask.ObjectDetection.YOLOv5 import YoloDetection
model=YoloDetection('cuda:0')
det=model.inference('/data/haonan.guo/RSChatGPT/test.tif',None,'/data/haonan.guo/RSChatGPT/output.png')