import itertools
import math
import os
import random
import diffusers
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, \
    AutoPipelineForInpainting
from diffusers.utils import load_image
import numpy as np
import torch
from dotenv import load_dotenv
# from tqdm import tqdm
# import set_seed
# from set_seed import seed_everything, resize_for_condition_image
#
# seed: int = 8888
# seed_everything(seed)
# out_dir: str = "lynlapat_snooker"
# negative_prompt: str = "nsfw, worst quality, low quality, normal quality, lowres,watermark, monochrome, light color, low resolution"
# device: str = "mps" if torch.backends.mps.is_available() else "cpu"
# print(device)
#
# schedulers = [
#     ("LMSDiscreteScheduler", diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler),
#     ("DDIMScheduler", diffusers.schedulers.scheduling_ddim.DDIMScheduler),
#     ("DPMSolverMultistepScheduler", diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler),
#     ("EulerDiscreteScheduler", diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler),
#     ("PNDMScheduler", diffusers.schedulers.scheduling_pndm.PNDMScheduler),
#     ("DDPMScheduler", diffusers.schedulers.scheduling_ddpm.DDPMScheduler),
#     ("EulerAncestralDiscreteScheduler", diffusers.schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteScheduler)
# ]
# models = [
#     ("stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),  # Reference bad
#     ("majicmixRealistic_v6", "../ai_directory/majicmixRealistic_v6"),  # Quite good
#     # ("MeinaV10", "../ai_directory/MeinaV10"),   # Anime
#     # ("perfectWorld_v4Baked", "../ai_directory/perfectWorld_v4Baked"),  # Ordinary
#     ("chilloutmix_NiPrunedFp32Fix", "../ai_directory/chilloutmix_NiPrunedFp32Fix"),
# ]
# # init_image = init_image.resize((512, 512))
# strengths = [1, ]
# # guidance_scales = [round(0.5 * _, 3) for _ in range(50, 71, 5)]
# guidance_scales = [10, 20, 30, 40]
# # eta_list = [round(0.2 * _, 3) for _ in range(0, 11, 1)]
# prompt = "best quality, highres, high definition masterpiece, photorealistic, a girl on the casino table."
# combined_list = list(itertools.product(models, strengths, guidance_scales, schedulers))
# random.shuffle(combined_list)
# init_image = load_image("sources/lynlapat.jpeg")
# width, height = init_image.size
# size_factor: float = 0.8
# new_width, new_height = math.floor(width * size_factor / 8) * 8, math.floor(height * size_factor / 8) * 8
# # new_width, new_height = width, height
# print(width, height)
# print(new_width, new_height)
# init_image = init_image.resize((new_width, new_height))
#
# generator = torch.Generator(device=device).manual_seed(seed)
# mask_image = load_image("sources/masked_lynlapat.png").resize((new_width, new_height))
#
#
# def make_inpaint_condition(image, image_mask):
#     image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
#     image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
#
#     assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
#     image[image_mask > 0.5] = 1  # set as masked pixel
#     image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
#     image = torch.from_numpy(image)
#     return image
#
#
# def main() -> None:
#     """Run main function."""
#     # strength = 1
#     # guidance_scale = 7.5
#     eta = 0
#     for item in tqdm(combined_list, total=len(combined_list)):
#         (model_name, model_id), strength, guidance_scale, (scheduler_name, scheduler) = item
#         add_prompt = prompt
#         print(strength, guidance_scale, eta, add_prompt)
#         filename: str = f"{out_dir}/{model_name}_{scheduler_name}_{add_prompt}_{strength}_{guidance_scale}_{eta}_0.png"
#         try:
#             if not os.path.exists(filename):
#                 control_image = make_inpaint_condition(init_image, mask_image)
#
#                 controlnet = ControlNetModel.from_pretrained(
#                     "lllyasviel/control_v11p_sd15_inpaint",
#                 ).to(device)
#                 pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#                     model_id,
#                     controlnet=controlnet,
#                     requires_safety_checker=False,
#                     safety_checker=None
#                 ).to(device)
#                 pipe.requires_safety_checker = False
#                 pipe.safety_checker = None
#
#                 pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
#
#                 # generate image
#                 result = pipe(
#                     prompt=add_prompt,
#                     negative_prompt=negative_prompt,
#                     num_inference_steps=150,
#                     generator=generator,
#                     image=init_image,
#                     mask_image=mask_image,
#                     control_image=control_image,
#                     num_images_per_prompt=1,
#                     width=new_width,
#                     height=new_height,
#                     strength=strength,
#                     guidance_scale=guidance_scale,
#                     # eta=eta,
#                 )
#                 for idx, image in enumerate(result.images):
#                     filename: str = f"{out_dir}/{model_name}_{scheduler_name}_{add_prompt}_{strength}_{guidance_scale}_{eta}_{idx}.png"
#                     image.save(filename)
#             else:
#                 print("File exists.")
#         except Exception as err:
#             print(err)
#             print(f"{filename} is impossible")
#             with open(f"{out_dir}/log.txt", "a") as file:
#                 file.write(str(err) + "\n")
#                 file.write(filename + "\n")
#                 file.write("====================================" + "\n")

load_dotenv()
REV_ANIMATED_MODEL_PATH = os.getenv('REV_ANIMATED_MODEL_PATH')
img = 'C:\\Users\\dovsy\\Downloads\\inpaint-example.png'
mask = 'mask.png'
init_image = Image.open(img)
mask_image = Image.open(mask)

pipe = AutoPipelineForInpainting.from_pretrained(
        REV_ANIMATED_MODEL_PATH,
        #use_safetensors=True,
        torch_dtype=torch.float16
    )#.to('cpu')

negative_prompt = 'ugly'
prompt = "a grey cat sitting on a bench, high resolution"

image = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                 image=init_image,
                 mask_image=mask_image
                 #guidance_scale=15,
                 #strength=1,
                 #num_inference_steps=50
                 ).images[0]
image.save('output.png')

