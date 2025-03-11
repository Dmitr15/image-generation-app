#######
import rembg
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from diffusers import AutoPipelineForInpainting, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
import os
from dotenv import load_dotenv

load_dotenv()
REV_ANIMATED_MODEL_PATH = os.getenv('REV_ANIMATED_MODEL_PATH')
KANDINSKY_MODEL_PATH = os.getenv('KANDINSKY_MODEL_PATH')
VAE_MODEL_PATH = os.getenv('VAE_MODEL_PATH')

def outpaint(prompt, negative_prompt, input_image, mask_img):
    pipe = AutoPipelineForInpainting.from_pretrained(
        REV_ANIMATED_MODEL_PATH,
        torch_dtype=torch.float32,
        vae=AutoencoderKL.from_pretrained(
             VAE_MODEL_PATH,
             subfolder=None)
    )

    result = pipe(prompt=prompt,
                  negative_prompt=negative_prompt,
                  width=768,
                  height=512,
                  num_inference_steps=50,
                  image=input_image,
                  mask_image=mask_img,
                  guidance_scale=15,
                  scheduler=EulerDiscreteScheduler.from_config(pipe.scheduler.config),
                  strength=1,
                  # generator=torch.manual_seed(seed_value)
                  ).images[0]
    return result


def create_canvas(width, height, img_path):
    image = Image.open(img_path)
    mask = Image.new("RGB", (width, height), "white")

    left = 256
    top = 128

    right = left + image.size[0] - 5
    bottom = top + image.size[1] - 5

    draw = ImageDraw.Draw(mask)
    draw.rectangle([left, top, right, bottom], fill="black")

    input_image = mask.copy()
    input_image.paste(image, (left, top))

    return input_image, mask


if __name__ == '__main__':
    prompt = """(best quality,realistic,highres:1.2), soft toned, realistic countryside, peaceful scene, detailed houses and trees, scenic countryside road, sunlit fields, tranquil atmosphere, warm sunlight, lush greenery, rustic charm, homely cottages, serene landscape, subtle shadows, bucolic setting, calm and quiet ambiance, soft light and shadows, immaculate details"""

    negative_prompt = "painting, digital art, 3d art, low quality, poor drown hands"

    img, mask = create_canvas(768, 512, 'C:\\Users\\dovsy\\Downloads\\hiker.jpg')

    res = outpaint(prompt, negative_prompt, img, mask)
    res.save('test5.jpg')
    res.show()
###############