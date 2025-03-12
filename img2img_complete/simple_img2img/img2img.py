import os
from diffusers.callbacks import SDCFGCutoffCallback
from dotenv import load_dotenv
import torch
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    DDIMScheduler, AutoencoderKL
# from prompt_engineering import art_styles
from PIL import Image
from io import BytesIO
from skimage import io
import tiktoken

load_dotenv()
# SDV5_MODEL_PATH = os.getenv('SDV5_MODEL_PATH')
SAVE_PATH = os.getenv('SAVE_PATH')
INPAINTING_MODEL_PATH = os.getenv('INPAINTING_MODEL_PATH')
REV_ANIMATED_MODEL_PATH = os.getenv('REV_ANIMATED_MODEL_PATH')
MMIX_MODEL_PATH = os.getenv('MMIX_MODEL_PATH')
DREAMLIKE_MODEL_PATH = os.getenv('DREAMLIKE')
VAE_MODEL_PATH = os.getenv('VAE_MODEL_PATH')
LORA_REALISTIC_PATH = os.getenv('LORA_REALISTIC')

prompt = 'cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k'
negative_prompt = 'cartoon, cgi, render, illustration, painting, drawing, bad quality, grainy, low resolution'
# ugly, tiling, mutation, extra limbs, disfigured, deformed, body out of frame, blurry, bad art, bad anatomy
num_inference_steps = 25
base_img = 'C:\\Users\\dovsy\\Downloads\\cat.png'

callback = SDCFGCutoffCallback(cutoff_step_ratio=0.4)
def num_of_tokens(str):
    encoding = tiktoken.get_encoding('r50k_base')
    num_tokens = len(encoding.encode(str))
    if num_tokens > 75:
        print('For a better picture, create a prompt containing less than 75 tokens!')
    return num_tokens


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + '(' + str(counter) + ')' + extension
        counter += 1

    return path


def img2img():
    shorted_prompt = (prompt[:25] + '...') if len(prompt) > 25 else prompt
    # shorted_prompt = removesplits(shorted_prompt)
    shorted_prompt = shorted_prompt.replace(' ', '_')
    generation_path = os.path.join(SAVE_PATH, shorted_prompt.removesuffix('...'))

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    if not os.path.exists(generation_path):
        os.mkdir(generation_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # use_safetensors=True,
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        DREAMLIKE_MODEL_PATH,
        use_safetensors=True,
        torch_dtype=torch.float32,
        vae=AutoencoderKL.from_pretrained(
            VAE_MODEL_PATH,
            subfolder=None)
        ).to('cpu')
    # pipe = pipe.to(device)
    # , use_safetensors=True

    # init_image = Image.open(BytesIO(base_img.content)).convert("RGB")
    # init_image = io.imread(base_img)
    # init_image = init_image.
    init_image = Image.open(base_img).convert("RGB")
    init_image = init_image.resize((768, 512))
    # scheduler = EulerDiscreteScheduler.from_pretrained(INPAINTING_MODEL_PATH, subfolder="scheduler")
    # for style_type, style_prompt in art_styles.items():
    prompt_stylized = f'{prompt}'

    print(f'Full prompt:\n{prompt_stylized}\n\n')
    # print(f'Characters in prompt: {len(prompt_stylized)}, limit: 200')
    print(f'Tokens:{num_of_tokens(prompt_stylized)}\n')

    pipe.load_lora_weights(LORA_REALISTIC_PATH)

    # for i in range(num_of_img_per_prompt):
    img = pipe(
        prompt_stylized,
        image=init_image,
        height=768,
        width=512,
        negative_prompt=negative_prompt,
        scheduler=DDIMScheduler.from_config(pipe.scheduler.config),
        strength=1,
        num_inference_steps=num_inference_steps,
        callback_on_step_end=callback,
        guidance_scale=25
    ).images[0]
    img_path = uniquify(os.path.join(SAVE_PATH, generation_path, ' - ' + shorted_prompt) + '.png')
    print(img_path)

    img.save(img_path)

    print('\nFINISHED\n')


if __name__ == '__main__':
    # print(num_of_tokens(prompt))
    img2img()
