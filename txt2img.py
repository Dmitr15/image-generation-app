import os
from dotenv import load_dotenv
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image, ControlNetModel, \
    EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
from prompt_engineering import art_styles
import tiktoken

list_prompt = [',', ':', ';', ']', '[', '|']

load_dotenv()
SDV5_MODEL_PATH = os.getenv('SDV5_MODEL_PATH')
SAVE_PATH = os.getenv('SAVE_PATH')
KANDINSKY_MODEL_PATH = os.getenv('KANDINSKY_MODEL_PATH')
REV_ANIMATED_MODEL_PATH = os.getenv('REV_ANIMATED_MODEL_PATH')
CONTROLNET_MODEL_PATH = os.getenv('CONTROLNET_MODEL_PATH')
MMIX_MODEL_PATH = os.getenv('MMIX_MODEL_PATH')


def num_of_tokens(str):
    encoding = tiktoken.get_encoding('r50k_base')
    num_tokens = len(encoding.encode(str))
    return num_tokens


def removesplits(prompt1):
    # prompt1 = prompt1.replace()
    prompt1 = prompt1.translate({ord(x): '' for x in list_prompt})
    return prompt1


prompt = 'best quality, highres, high definition masterpiece, photorealistic'
negative_prompt = 'nsfw, worst quality, low quality, normal quality, lowres,watermark, monochrome, low resolutio, extra limbs, people, ugly face, anime'
num_inference_steps = 150
height = 768
width = 512
num_of_img_per_prompt = 1

# low_vram = True
# device_type = 'cpu'
seed = 42


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + '(' + str(counter) + ')' + extension
        counter += 1

    return path


def render_prompt():
    shorted_prompt = (prompt[:25] + '...') if len(prompt) > 25 else prompt
    shorted_prompt = removesplits(shorted_prompt)
    shorted_prompt = shorted_prompt.replace(' ', '_')
    generation_path = os.path.join(SAVE_PATH, shorted_prompt.removesuffix('...'))

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    if not os.path.exists(generation_path):
        os.mkdir(generation_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_PATH, torch_dtype=torch.float32)
    pipe = StableDiffusionPipeline.from_pretrained(MMIX_MODEL_PATH, use_safetensors=True, torch_dtype=torch.float32).to('cpu')

    for style_type, style_prompt in art_styles.items():
        prompt_stylized = f'{prompt}, {style_prompt}'

        print(f'Full prompt:\n{prompt_stylized}\n\n')
        print(f'Characters in prompt: {len(prompt_stylized)}, limit: 200')
        print(f'Tokens:{num_of_tokens(prompt_stylized)}, limit: 75\n')

        for i in range(num_of_img_per_prompt):
            img = pipe(prompt_stylized,
                       negative_prompt=negative_prompt,
                       height=height,
                       width=width,
                       scheduler=EulerDiscreteScheduler.from_config(pipe.scheduler.config),
                       num_inference_steps=50,
                       control_image=controlnet,
                       guidance_scale=40,
                       strength=1
                       # generator=torch.manual_seed(seed)
                       ).images[0]
            img_path = uniquify(os.path.join(SAVE_PATH, generation_path, style_type + ' - ' + shorted_prompt) + '.png')
            print(img_path)

            img.save(img_path)

        print('\nFINISHED\n')


if __name__ == '__main__':
    render_prompt()
