#################################################################################
import torch
from controlnet_aux import CannyDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import os
from dotenv import load_dotenv

load_dotenv()
MMIX_MODEL_PATH = os.getenv('MMIX_MODEL_PATH')
IP_ADAPTER_PATH = os.getenv('IP_ADAPTER')
CANNY_MODEL_PATH=os.getenv('CANNY_MODEL')

img1 = 'C:\\Users\\dovsy\\Downloads\\image-4.png'
img2 = 'C:\\Users\\dovsy\\Downloads\\image-3.png'

controlnet = ControlNetModel.from_pretrained(
    CANNY_MODEL_PATH,
    torch_dtype=torch.float32,
    varient="fp32"
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    MMIX_MODEL_PATH,
    controlnet=controlnet,
    torch_dtype=torch.float32)

pipe.load_ip_adapter(
                     IP_ADAPTER_PATH,
                     subfolder="models",
                     weight_name="ip-adapter_sd15.bin"
)

ip_adap_img = load_image(img2)
img = load_image(img1).resize((768, 768))

canny = CannyDetector()
canny_img = canny(img, detect_resolution=512, image_resolution=768)

prompt = """(photorealistic:1.2), raw, masterpiece, high quality, 8k, girl wearing hoodie, headphones, hands in pocket"""
negative_prompt = "low quality, ugly, mutated"

pipe.set_ip_adapter_scale(0.5)

images = pipe(prompt = prompt,
              negative_prompt = negative_prompt,
              height = 768,
              width = 768,
              ip_adapter_image = ip_adap_img,
              image = canny_img,
              guidance_scale = 6,
              controlnet_conditioning_scale = 0.7,
              num_inference_steps = 20
              ).images[0]
images.save('canny.png')
##########################################################################