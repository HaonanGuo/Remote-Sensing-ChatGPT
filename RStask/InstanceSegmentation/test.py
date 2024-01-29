from RStask import InstanceFunction
model=InstanceFunction('cuda:0')
model.inference('/data/haonan.guo/LoveDA/Train/Urban/images_png/1367.png','bike','/data/haonan.guo/RSChatGPT/output.png')