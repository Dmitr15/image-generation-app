import os
from dotenv import load_dotenv
import torch
import numpy as np
from torch import autocast
from diffusers import AutoPipelineForInpainting, EulerDiscreteScheduler, ControlNetModel, StableDiffusionControlNetInpaintPipeline
from prompt_engineering import art_styles
import tiktoken
from prompt_engineering import art_styles
from PIL import Image, ImageOps
from io import BytesIO
from skimage import io
import tiktoken
from img2img import num_of_tokens, uniquify
from txt2img import removesplits
from diffusers.utils import load_image, make_image_grid
import rembg

load_dotenv()
SDV5_MODEL_PATH = os.getenv('SDV5_MODEL_PATH')
SAVE_PATH = os.getenv('SAVE_PATH')
INPAINTING_MODEL_PATH=os.getenv('INPAINTING_MODEL_PATH')
REV_ANIMATED_MODEL_PATH=os.getenv('REV_ANIMATED_MODEL_PATH')
CONTROLNET_MODEL_PATH=os.getenv('CONTROLNET_MODEL_PATH')
MMIX_MODEL_PATH=os.getenv('MMIX_MODEL_PATH')

prompt = 'best quality, highres, high definition masterpiece, photorealistic, a girl on the casino table'
negative_prompt = 'worst quality, low quality, normal quality, lowres,watermark, monochrome, light color, low resolution'
num_inference_steps = 44
base_img = 'C:\\Users\\dovsy\\Downloads\\ddw.jpg'
num_of_img_per_prompt = 1

# def make_inpaint_condition(init_img, mask):
#     init_image = np.array(init_img.convert("RGB")).astype(np.float32) / 255.0
#     mask_image = np.array(mask.convert("L")).astype(np.float32) / 255.0
#
#     init_image[mask_image > 0.5] = -1.0  # set as masked pixel
#     init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
#     init_image = torch.from_numpy(init_image)
#     return init_image

if __name__ == '__main__':
    # controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_PATH, torch_dtype=torch.float32)
    #
    # pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    #     SDV5_MODEL_PATH,
    #     controlnet=controlnet,
    #     # use_safetensors=True,
    #     torch_dtype=torch.float32
    # ).to('cpu')
    #
    # base_img = 'C:\\Users\\dovsy\\Downloads\\photo_2025-01-29_18-30-39.jpg'
    # mask = 'test1_img.jpg'
    #
    # init_image = Image.open(base_img).convert("RGB")
    # mask_image = Image.open(mask).convert("RGB")
    #
    # print(type(mask_image))
    # print(type(init_image))
    #
    # image = pipeline(
    #     prompt=prompt,
    #     # scheduler=scheduler,
    #     negative_prompt=negative_prompt,
    #     width=512,
    #     height=768,
    #     num_inference_steps=50,
    #     image=init_image,
    #     mask_image=mask_image,
    #     control_image=control_image,
    #     guidance_scale=10,
    #     strength=0.7
    #     # generator=torch.manual_seed(189123)
    # ).images[0]
    #
    # image.save('test16_img.jpg')
    init_image = load_image(base_img).resize((512, 768))

    # Convert the input image to a numpy array
    input_array = np.array(init_image)

    # Extract mask using rembg
    mask_array = rembg.remove(input_array, only_mask=True)

    # Create a PIL Image from the output array
    mask_image = Image.fromarray(mask_array)
    mask_image_inverted = ImageOps.invert(mask_image)

    pipeline = AutoPipelineForInpainting.from_pretrained(
        REV_ANIMATED_MODEL_PATH,
        torch_dtype=torch.float32
    )

    image = pipeline(prompt=prompt,
                     negative_prompt=negative_prompt,
                     width=512,
                     height=768,
                     num_inference_steps=20,
                     image=init_image,
                     mask_image=mask_image_inverted,
                     guidance_scale=1,
                     strength=0.7
                     #generator=torch.manual_seed(189018)
                     ).images[0]

    image.save('test24_img.jpg')