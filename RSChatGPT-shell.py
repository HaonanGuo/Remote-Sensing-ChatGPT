import os

import re
import uuid
from skimage import io
import argparse
import inspect
from langchain.chat_models import ChatOpenAI
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
import numpy as np
from Prefix import  RS_CHATGPT_PREFIX, RS_CHATGPT_FORMAT_INSTRUCTIONS, RS_CHATGPT_SUFFIX
from RStask import ImageEdgeFunction,CaptionFunction,LanduseFunction,DetectionFunction,CountingFuncnction,SceneFunction,InstanceFunction



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
    recent_prev_file_name = name_split[0]
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}.png'.replace('__','_')
    return os.path.join(head, new_file_name)

class EdgeDetection:
    def __init__(self, device):
        print("Initializing Edge Detection Function....")
        self.func = ImageEdgeFunction()
    @prompts(name="Edge Detection On Image",
             description="useful when you want to detect the edge of the remote sensing image. "
                         "like: detect the edges of this image, or canny detection on image, "
                         "or perform edge detection on this image, or detect the  edge of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        updated_image_path=get_new_image_name(inputs, func_name="edge")
        self.func.inference(inputs,updated_image_path)
        return updated_image_path

class ObjectCounting:
    def __init__(self, device):
        self.func=CountingFuncnction(device)
    @prompts(name="Count object",
             description="useful when you want to count the number of the  object in the image. "
                         "like: how many planes are there in the image? or count the number of bridges"
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be counted")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        log_text=self.func.inference(image_path,det_prompt)
        return log_text


class InstanceSegmentation:
    def __init__(self, device):
        print("Initializing InstanceSegmentation")
        self.func=InstanceFunction(device)
    @prompts(name="Instance Segmentation for Remote Sensing Image",
             description="useful when you want to apply man-made instance segmentation for the image. The expected input category include plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, vehicle, helicopter, roundabout, soccer ball field, and swimming pool."
                         "like: extract plane from this image, "
                         "or predict the ship in this image, or extract tennis court from this image, segment harbor from this image, Extract the vehicle in the image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text of the category,selected from plane, or ship, or storage tank, or baseball diamond, or tennis court, or basketball court, or ground track field, or harbor, or bridge, or vehicle, or helicopter, or roundabout, or soccer ball field, or  swimming pool. ")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        updated_image_path = get_new_image_name(image_path, func_name="instance_" + det_prompt)
        text=self.func.inference(image_path, det_prompt,updated_image_path)
        return text

class SceneClassification:
    def __init__(self, device):
        print("Initializing SceneClassification")
        self.func=SceneFunction(device)
    @prompts(name="Scene Classification for Remote Sensing Image",
             description="useful when you want to know the type of scene or function for the image. "
                         "like: what is the category of this image?, "
                         "or classify the scene of this image, or predict the scene category of this image, or what is the function of this image. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, inputs):
        output_txt=self.func.inference(inputs)
        return output_txt


class LandUseSegmentation:
    def __init__(self, device):
        print("Initializing LandUseSegmentation")
        self.func=LanduseFunction(device)

    @prompts(name="Land Use Segmentation for Remote Sensing Image",
             description="useful when you want to apply land use gegmentation for the image. The expected input category include Building, Road, Water, Barren, Forest, Farmland, Landuse."
                         "like: generate landuse map from this image, "
                         "or predict the landuse on this image, or extract building from this image, segment roads from this image, Extract the water bodies in the image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text of the category,selected from Lnad Use, or Building, or Road, or Water, or Barren, or Forest, or Farmland, or Landuse.")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        updated_image_path = get_new_image_name(image_path, func_name="landuse")
        text=self.func.inference(image_path, det_prompt,updated_image_path)
        return text

class ObjectDetection:
    def __init__(self, device):
        self.func=DetectionFunction(device)


    @prompts(name="Detect the given object",
             description="useful when you only want to detect the bounding box of the certain objects in the picture according to the given text."
                         "like: detect the plane, or can you locate an object for me."
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be found")

    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        updated_image_path = get_new_image_name(image_path, func_name="detection_" + det_prompt.replace(' ', '_'))
        log_text=self.func.inference(image_path, det_prompt,updated_image_path)
        return log_text

class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.func=CaptionFunction(device)
    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        captions = self.func.inference(image_path)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions

class RSChatGPT:
    def __init__(self, gpt_name,load_dict,openai_key,proxy_url):
        print(f"Initializing RSChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
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

        self.llm = ChatOpenAI(api_key=openai_key, base_url=proxy_url, model_name=gpt_name,temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def initialize(self):
        self.memory.clear() #clear previous history
        PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = RS_CHATGPT_PREFIX, RS_CHATGPT_FORMAT_INSTRUCTIONS, RS_CHATGPT_SUFFIX
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,stop=["\nObservation:", "\n\tObservation:"],
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,'suffix': SUFFIX}, )

    def run_text(self, text, state):
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state
    def run_image(self, image_dir, state, txt=None):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        img = io.imread(image_dir)
        # width, height = img.shape[1],img.shape[0]
        # ratio = min(640 / width, 640 / height)
        # if ratio<1:
        #     width_new, height_new = (round(width * ratio), round(height * ratio))
        # else:
        #     width_new, height_new =width,height
        # width_new = int(np.round(width_new / 64.0)) * 64
        # height_new = int(np.round(height_new / 64.0)) * 64
        #
        # if width_new!=width or height_new!=height:
        #     img = cv2.resize(img,(width_new, height_new))
        #     print(f"======>Auto Resizing Image from {height,width} to {height_new,width_new}...")
        # else:
        #     print(f"======>Auto Renaming Image...")
        io.imsave(image_filename, img.astype(np.uint8))
        description = self.models['ImageCaptioning'].inference(image_filename)
        Human_prompt = f' Provide a remote sensing image named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\".'
        AI_prompt = "Received."
        self.memory.chat_memory.add_user_message(Human_prompt)
        self.memory.chat_memory.add_ai_message(AI_prompt)

        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        state=self.run_text(f'{txt} {image_filename} ', state)
        return state



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_key', type=str,required=True)
    parser.add_argument('--image_dir', type=str,required=True)
    parser.add_argument('--gpt_name', type=str, default="gpt-3.5-turbo",choices=['gpt-3.5-turbo-1106','gpt-3.5-turbo','gpt-4','gpt-4-0125-preview','gpt-4-turbo-preview','gpt-4-1106-preview'])
    parser.add_argument('--proxy_url', type=str, default=None)
    parser.add_argument('--load', type=str,help='Image Captioning is basic models that is required. You can select from [ImageCaptioning,ObjectDetection,LandUseSegmentation,InstanceSegmentation,ObjectCounting,SceneClassification,EdgeDetection]',
                        default="ImageCaptioning_cuda:0,SceneClassification_cuda:0,ObjectDetection_cuda:0,LandUseSegmentation_cuda:0,InstanceSegmentation_cuda:0,ObjectCounting_cuda:0,EdgeDetection_cpu")
    args = parser.parse_args()
    state = []
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    bot = RSChatGPT(gpt_name=args.gpt_name,load_dict=load_dict,openai_key=args.openai_key,proxy_url=args.proxy_url)
    bot.initialize()
    print('RSChatGPT initialization done, you can now chat with RSChatGPT~')
    bot.initialize()
    txt='Count the number of plane in the image.'
    state=bot.run_image(args.image_dir, [], txt)

    while 1:
        txt = input('You can now input your question.(e.g. Extract buildings from the image)\n')
        state = bot.run_image(args.image_dir, state, txt)


