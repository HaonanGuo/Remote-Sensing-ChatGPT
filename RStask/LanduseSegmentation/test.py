from RStask import LanduseFunction
model=LanduseFunction('cuda:0')
model.inference('/data/haonan.guo/LoveDA/Train/Urban/images_png/1367.png','road','/data/haonan.guo/RSChatGPT/output.png')