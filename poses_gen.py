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
CONTROLNET_MODEL_PATH=os.getenv('CONTROLNET_MODEL_PATH')

