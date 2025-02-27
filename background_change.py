import os
from dotenv import load_dotenv
import torch
import cv2
import numpy as np
from torch import autocast
from diffusers import AutoPipelineForInpainting, EulerDiscreteScheduler, ControlNetModel, \
    StableDiffusionControlNetInpaintPipeline, EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, AutoencoderKL
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
KANDINSKY_MODEL_PATH=os.getenv('KANDINSKY_MODEL_PATH')
REV_ANIMATED_MODEL_PATH=os.getenv('REV_ANIMATED_MODEL_PATH')
CONTROLNET_MODEL_PATH=os.getenv('CONTROLNET_MODEL_PATH')
MMIX_MODEL_PATH=os.getenv('MMIX_MODEL_PATH')
SAM_MODEL_PATH = os.getenv('SAM_MODEL_PATH')
VAE_MODEL_PATH = os.getenv('VAE_MODEL_PATH')
CANNY_MODEL_PATH=os.getenv('CANNY_MODEL')

#best quality, highres, high definition masterpiece, trending on artstation, unrealistic volumetric fog on background, girl in
#ugly, poorly drawn eyes, poorly drawn hands, mutation,
prompt = 'best quality, highres, high definition masterpiece, photorealistic, foggy winter forest'
negative_prompt = 'nsfw, people, worst quality, low quality, normal quality, lowres,watermark, monochrome, low resolutio, extra limbs'
num_inference_steps = 40
base_img = 'C:\\Users\\dovsy\\Downloads\\cat.png'
num_of_img_per_prompt = 1

controlnet = ControlNetModel.from_pretrained(
    CANNY_MODEL_PATH,
    torch_dtype=torch.float32,
    varient="fp32"
)

def binary_mask(init_image):
    #init_image = load_image(url).resize((512, 768))
    input_array = np.array(init_image)
    mask_array = rembg.remove(input_array, only_mask=True)
    mask_image = Image.fromarray(mask_array)
    mask_image = ImageOps.invert(mask_image)
    return mask_image


def make_inpaint_condition(init_img, mask):
    init_image = np.array(init_img.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask.convert("L")).astype(np.float32) / 255.0

    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image

# def render_prompt():
#     shorted_prompt = (prompt[:25] + '...') if len(prompt) > 25 else prompt
#     shorted_prompt = removesplits(shorted_prompt)
#     shorted_prompt = shorted_prompt.replace(' ', '_')
#     generation_path = os.path.join(SAVE_PATH, shorted_prompt.removesuffix('...'))
#
#     if not os.path.exists(SAVE_PATH):
#         os.mkdir(SAVE_PATH)
#     if not os.path.exists(generation_path):
#         os.mkdir(generation_path)
#
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     pipe = StableDiffusionPipeline.from_pretrained(SDV5_MODEL_PATH, use_safetensors=True, torch_dtype=torch.float32).to(device)
#
#     for style_type, style_prompt in art_styles.items():
#         prompt_stylized = f'{prompt}, {style_prompt}'
#
#         print(f'Full prompt:\n{prompt_stylized}\n\n')
#         print(f'Characters in prompt: {len(prompt_stylized)}, limit: 200')
#         print(f'Tokens:{num_of_tokens(prompt_stylized)}\n')
#
#         for i in range(num_of_img_per_prompt):
#             img = pipe(prompt_stylized, negative_prompt=negative_prompt, height=height, width=width,
#                        num_inference_steps=num_inference_steps, generator=torch.manual_seed(seed)).images[0]
#             img_path = uniquify(os.path.join(SAVE_PATH, generation_path, style_type + ' - ' + shorted_prompt) + '.png')
#             print(img_path)
#
#             img.save(img_path)
#
#         print('\nFINISHED\n')

def toPill(image):
    minv = np.amin(image)
    maxv = np.amax(image)
    image = image - minv
    image = image / (maxv - minv)
    image = image * 255
    img = Image.fromarray(image.astype(np.uint8))
    return img

if __name__ == '__main__':
    # render_prompt()
    #controlnet = ControlNetModel.from_pretrained(REV_ANIMATED_MODEL_PATH, torch_dtype=torch.float32)
    #scheduler = EulerDiscreteScheduler.from_pretrained(REV_ANIMATED_MODEL_PATH, subfolder="scheduler")

    pipeline = AutoPipelineForInpainting.from_pretrained(
        REV_ANIMATED_MODEL_PATH,
        controlnet=controlnet,
        # use_safetensors=True,
        torch_dtype=torch.float32,
        vae=AutoencoderKL.from_pretrained(
                 VAE_MODEL_PATH,
                 subfolder=None)
        #requires_safety_checker=False,
        #safety_checker=None
    ).to('cpu')
    #pipeline.requires_safety_checker = False
    #pipeline.safety_checker = None


    size = (768, 512)
    init_image = Image.open(base_img)
    init_image.thumbnail(size)
    #print(type(init_image))
    init_image = toPill(init_image)
    #print(type(init_image))

    #mask = 'C:\\Users\\dovsy\\Downloads\\ddw1.png'
    #mask_image = load_image(mask).resize((512, 768))

    bin_mask=binary_mask(init_image)
    #print(type(bin_mask))
    # Or
    #bin_mask = Image.open(mask)
    #bin_mask.thumbnail(size)

    #control_image = make_inpaint_condition(init_image, bin_mask)

    #blurred_mask.save('mask_test1_img.jpg')
    #print(type(bin_mask))
    #print(type(init_image))
    #mask_test = binary_mask(init_image)
    #mask_test.save('test_mask_img.jpg')
    print(type(init_image))
    print(type(bin_mask))
    print(init_image.size)
    print(bin_mask.size)
    #init_image.show()
    #bin_mask.show()
    #bin_mask.show()
    #init_image.show()
    image = pipeline(
                     prompt=prompt,
                     scheduler=EulerDiscreteScheduler.from_config(pipeline.scheduler.config),
                     negative_prompt=negative_prompt,
                     width=1024,
                     height=704,
                     num_inference_steps=70,
                     image=init_image,
                     mask_image=bin_mask,
                     control_image=init_image,
                     guidance_scale=15,
                     strength=0.8
                     #generator=torch.manual_seed(189123)
                     ).images[0]

    image.save('test1.jpg')
