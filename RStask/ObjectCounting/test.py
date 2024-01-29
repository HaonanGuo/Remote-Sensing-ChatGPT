from RStask import CountingFuncnction
model=CountingFuncnction('cuda:0')
txt='/data/haonan.guo/RSChatGPT/test.tif,small vehicles'
p,t=txt.split(",")
model.inference(p,t)
