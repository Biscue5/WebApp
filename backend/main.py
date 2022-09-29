from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urldefrag
from PIL import Image
import numpy as np
import requests
from omegaconf import OmegaConf
from PIL import Image
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

app = FastAPI()
origins = ['http://localhost:3000']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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

@app.get("/")
def inpainting_image(image, mask, steps=30):
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
    return inpainted