import torch
import numpy as np
from numpy import asarray
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting, EulerDiscreteScheduler
import cv2
import os
from dotenv import load_dotenv
from segment_anything import sam_model_registry, SamPredictor

load_dotenv()
REV_ANIMATED_MODEL_PATH = os.getenv('REV_ANIMATED_MODEL_PATH')
KANDINSKY_MODEL_PATH = os.getenv('KANDINSKY_MODEL_PATH')
VAE_MODEL_PATH = os.getenv('VAE_MODEL_PATH')
SAM_MODEL_PATH = os.getenv("SAM_MODEL_PATH")
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = 'sam_vit_b_01ec64.pth'
SDV5_MODEL_PATH = os.getenv('SDV5_MODEL_PATH')
img = 'C:\\Users\\dovsy\\Downloads\\inpaint-example.png'


# mask generation function
def mask_generator(img):
    image = cv2.imread(img)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device='cuda')

    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image_rgb)
    input_point = np.array([[250, 250]])
    input_label = np.array([1])
    masks, scores, logits = mask_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    mask = masks.astype(float) * 255
    mask = np.transpose(mask, (1, 2, 0))
    _, bw_image = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    cv2.imwrite('mask.png', bw_image)


def chanel2channels(img_path):
    # img = cv2.imread(img_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img2 = np.zeros_like(img)
    # img2[:, :, 0] = gray
    # img2[:, :, 1] = gray
    # img2[:, :, 2] = gray
    # cv2.imwrite('new_mask.png', img2)
    #image = Image.open(img_path).convert("RGB")
    #print(image.sh)
    # img = Image.open(img_path)
    # numpydata = asarray(img)
    # img2 = cv2.merge((numpydata,numpydata,numpydata))
    # return img2
    img = cv2.imread(img_path)
    img = Image.open(img_path)
    #img.size()
    # if img.size()[1] == 1:  # check if it's single channel
    #     image = img.expand(-1, 3, -1, -1)

def outpaint_mask(IMAGE_PATH):
    image = cv2.imread(IMAGE_PATH)
    height, width = image.shape[:2]#????
    padding = 100
    mask = np.ones((height + 2 * padding, width + 2 * padding), dtype=np.uint8) * 255
    mask[padding:-padding, padding:-padding] = 0
    cv2.imwrite("mask.png", mask)
    image_extended = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant',constant_values=128)
    #cv2_imshow(image_extended)
    cv2.imwrite("image_extended.png", image_extended)

#def outpainting()
# inpainting function
def inpaint(init_img, mask):
    init_image = Image.open(init_img).convert('RGB')
    mask_image = Image.open(mask).convert('RGB')

    # if mask_image.size()[1] == 1:  # check if it's single channel
    #     mask_image = mask_image.expand(-1, 3, -1, -1)

    pipe = AutoPipelineForInpainting.from_pretrained(
        KANDINSKY_MODEL_PATH,
        use_safetensors=True,
        torch_dtype=torch.float32
    ).to('cpu')

    negative_prompt = 'ugly, mutated, disformed'
    prompt = "a grey cat sitting on a bench, high resolution"
    image = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                 image=init_image,
                 mask_image=mask_image,
                 num_inference_steps=100,
                 scheduler=EulerDiscreteScheduler.from_config(pipe.scheduler.config),
                 ).images[0]
    image.save('output.png')


# mask_generator(img)

#mask = chanel2channels('mask.png')
outpaint_mask("inpaint-example.png")
#inpaint(img, 'mask.png')
