import rembg
import torch
import numpy as np
from PIL import Image, ImageOps
from diffusers import AutoPipelineForInpainting, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
import os
from dotenv import load_dotenv
from optimum.quanto import freeze, qfloat8, quantize

load_dotenv()
SDV5_MODEL_PATH = os.getenv('SDV5_MODEL_PATH')
REV_ANIMATED_MODEL_PATH = os.getenv('REV_ANIMATED_MODEL_PATH')
KANDINSKY_MODEL_PATH = os.getenv('KANDINSKY_MODEL_PATH')
MMIX_MODEL_PATH = os.getenv('MMIX_MODEL_PATH')
CONTROLNET_MODEL_PATH = os.getenv('CONTROLNET_MODEL_PATH')
VAE_MODEL_PATH = os.getenv('VAE_MODEL_PATH')

init_image = load_image("C:\\Users\\dovsy\\Downloads\\ddw.png").resize((512, 768))
# mask = load_image("C:\\Users\\dovsy\\Downloads\\ddw1.png").resize((512, 768))

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Convert the input image to a numpy array
input_array = np.array(init_image)

# # Extract mask using rembg
# mask_array = rembg.remove(input_array, only_mask=True)
#
# # Create a PIL Image from the output array
# mask_image = Image.fromarray(mask_array)
#
# mask_image_inverted = ImageOps.invert(mask_image)
mask_image_inverted = load_image("C:\\Users\\dovsy\\Downloads\\ddw1.png").resize((512, 768))
mask_image_inverted = np.array(mask_image_inverted)
# torch.cuda.memory_summary(device=None, abbreviated=False)
# torch.cuda.empty_cache()

pipeline = AutoPipelineForInpainting.from_pretrained(
    MMIX_MODEL_PATH,
    torch_dtype=torch.float32,
    requires_safety_checker=False,
    safety_checker=None
    # vae=AutoencoderKL.from_pretrained(
    #     VAE_MODEL_PATH,
    #     subfolder=None)
).to('cpu')
pipeline.requires_safety_checker = False
pipeline.safety_checker = None
# pipeline.enable_model_cpu_offload()
# quantize(pipeline.unet, weights=qfloat8)
# freeze(pipeline.unet)
# pipeline.enable_xformers_memory_efficient_attention()
prompt = "best quality, highres, high definition masterpiece, photorealistic, girl on casino table"

negative_prompt = "nsfw, worst quality, low quality, normal quality, lowres,watermark, monochrome, low resolutio, extra limbs, people, ugly face, people, girl"

image = pipeline(prompt=prompt,
                 negative_prompt=negative_prompt,
                 width=512,
                 height=768,
                 num_inference_steps=25,
                 image=init_image,
                 #scheduler=EulerDiscreteScheduler.from_config(pipeline.scheduler.config),
                 mask_image=mask_image_inverted,
                 guidance_scale=13,
                 strength=0.8
                 # generator=torch.manual_seed(189061)
                 ).images[0]

image.save('test11_img.jpg')
