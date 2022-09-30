from email import base64mime
from re import U
from fastapi import FastAPI, UploadFile, File
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import requests
import base64
from omegaconf import OmegaConf
from PIL import Image
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

app = FastAPI()

def make_batch(image, mask, device, img_size=512):
    image = image.convert("RGB")
    image = image.resize((img_size, img_size))
    image = np.array(image)
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)
    mask = mask.convert("L")
    mask = mask.resize((img_size, img_size))
    mask = np.array(mask)
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)
    masked_image = (1-mask)*image
    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch

def inpaintin_image(image, mask, steps=2):
    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
                          strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    with torch.no_grad():
        with model.ema_scope():
            batch = make_batch(image, mask, device=device)
            # encode masked image and concat downsample
            c = model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(batch["mask"],
                                                    size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)
            shape = (c.shape[1]-1,)+c.shape[2:]
            samples_ddim, _ = sampler.sample(S= steps,
                                                conditioning=c,
                                                batch_size=c.shape[0],
                                                shape=shape,
                                                verbose=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            image = torch.clamp((batch["image"]+1.0)/2.0,
                                min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                min=0.0, max=1.0)
            predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                            min=0.0, max=1.0)
            inpainted = (1-mask)*image+mask*predicted_image
            inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
            inpainted = Image.fromarray(inpainted.astype(np.uint8))

    return inpainted

@app.post("/predict/")  
async def inpainting_img(files: List[UploadFile] = File(...)):
    img = Image.open(io.BytesIO(files[0].file.read()))
    mask = Image.open(io.BytesIO(files[1].file.read()))

    inpainted_img = inpaintin_image(img, mask)

    with io.BytesIO() as output:
        inpainted_img.save(output, format='PNG')
        bytes_inpinted_img = output.getvalue()
        bytes_inpinted_img = base64.b64encode(bytes_inpinted_img)    

    return bytes_inpinted_img

