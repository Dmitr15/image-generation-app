import os
from dotenv import load_dotenv
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image, ControlNetModel, \
    EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, AutoencoderKL, DDIMScheduler, AutoPipelineForInpainting
# from prompt_engineering import art_styles
import tiktoken
from diffusers.callbacks import SDCFGCutoffCallback
from PIL import Image, ImageOps
import rembg
import numpy as np

load_dotenv()
SAVE_PATH = os.getenv('SAVE_PATH')
KANDINSKY_MODEL_PATH = os.getenv('KANDINSKY_MODEL_PATH')
REV_ANIMATED_MODEL_PATH = os.getenv('REV_ANIMATED_MODEL_PATH')
CONTROLNET_MODEL_PATH = os.getenv('CONTROLNET_MODEL_PATH')
MMIX_MODEL_PATH = os.getenv('MMIX_MODEL_PATH')
DREAMLIKE_MODEL_PATH = os.getenv('DREAMLIKE')
LORA_REALISTIC_PATH = os.getenv('LORA_REALISTIC')
LORA_PIXELS_PATH = os.getenv('LORA_PIXELS')
VAE_MODEL_PATH = os.getenv('VAE_MODEL_PATH')
LORA_NIGHTTIME_PATH = os.getenv('LORA_NIGHTTIME')


def num_of_tokens(str):
    encoding = tiktoken.get_encoding('r50k_base')
    num_tokens = len(encoding.encode(str))
    return num_tokens


def binary_mask(init_image):
    # init_image = load_image(url).resize((512, 768))
    input_array = np.array(init_image)
    mask_array = rembg.remove(input_array, only_mask=True)
    mask_image = Image.fromarray(mask_array)
    # mask_image = ImageOps.invert(mask_image)
    return mask_image


callback = SDCFGCutoffCallback(cutoff_step_ratio=0.4)
prompt = 'concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k'
negative_prompt = 'cartoon, cgi, render, illustration, painting, drawing, bad quality, grainy, low resolution'
base_img = 'C:\\Users\\dovsy\\Pictures\\inpaint.png'
num_inference_steps = 150
height = 512
width = 512
num_of_img_per_prompt = 1


def toPill(image):
    minv = np.amin(image)
    maxv = np.amax(image)
    image = image - minv
    image = image / (maxv - minv)
    image = image * 255
    img = Image.fromarray(image.astype(np.uint8))
    return img


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + '(' + str(counter) + ')' + extension
        counter += 1

    return path


def create_pipeline():
    pipeline = AutoPipelineForInpainting.from_pretrained(
        REV_ANIMATED_MODEL_PATH,
        # custom_pipeline="lpw_stable_diffusion",
        # use_safetensors=True,
        torch_dtype=torch.float32,
        vae=AutoencoderKL.from_pretrained(
            VAE_MODEL_PATH,
            subfolder=None)
    ).to('cpu')
    # pipeline.enable_xformers_memory_efficient_attention()
    return pipeline


def render_prompt():
    init_image = Image.open(base_img)
    # init_image.thumbnail(size)
    init_image = toPill(init_image)

    bin_mask = binary_mask(init_image)

    shorted_prompt = (prompt[:25] + '...') if len(prompt) > 25 else prompt
    # shorted_prompt = removesplits(shorted_prompt)
    shorted_prompt = shorted_prompt.replace(' ', '_')
    generation_path = os.path.join(SAVE_PATH, shorted_prompt.removesuffix('...'))

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    if not os.path.exists(generation_path):
        os.mkdir(generation_path)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_PATH, torch_dtype=torch.float32)
    pipe = create_pipeline()

    # for style_type, style_prompt in art_styles.items():
    prompt_stylized = f'{prompt}'
    # pipe.load_lora_weights(LORA_PIXELS_PATH, weight_name="Pixhell_15.safetensors", adapter_name="pixel")
    # pipe.load_lora_weights(LORA_REALISTIC_PATH)
    # pipe.load_lora_weights(LORA_NIGHTTIME_PATH)
    print(f'Full prompt:\n{prompt_stylized}\n\n')
    # print(f'Characters in prompt: {len(prompt_stylized)}, limit: 200')
    print(f'Tokens:{num_of_tokens(prompt_stylized)}, limit: 75\n')

    for i in range(num_of_img_per_prompt):
        img = pipe(prompt_stylized,
                   negative_prompt=negative_prompt,
                   scheduler=DDIMScheduler.from_config(pipe.scheduler.config),
                   num_inference_steps=25,
                   image=init_image,
                   mask_image=bin_mask,
                   height=init_image.height,
                   width=init_image.width,
                   # control_image=controlnet,
                   guidance_scale=15,
                   # cross_attention_kwargs={"scale": 0.6},
                   strength=1,
                   # callback_on_step_end=callback
                   ).images[0]
        # del pipe
        img_path = uniquify(os.path.join(SAVE_PATH, generation_path + ' - ' + shorted_prompt) + '.png')
        print(img_path)

        img.save(img_path)

    print('\nFINISHED\n')


if __name__ == '__main__':
    render_prompt()
