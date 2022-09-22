from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np



from omegaconf import OmegaConf
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.transforms import ToPILImage, CenterCrop, Resize
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

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

col1, col2 = st.beta_columns(2)


drawing_mode = "freedraw"
canvas_result = None

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 11)
stroke_color = '#ffffff' #white
with col1:
    st.header('Input Image')
    bg_image = st.file_uploader("Background image:", type=['jpg', 'jpeg', 'png'], use_column_width=True)

if bg_image:
    img = Image.open(bg_image)
    img_ = np.array(img)
    height, width = img_.shape[:-1]

realtime_update = st.sidebar.checkbox("Update in realtime", True)

if bg_image:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        #background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        point_display_radius=0,
        key="canvas",
    )

if canvas_result:
    try:
        mask = canvas_result.image_data.transpose(2,0,1)[0]
        mask = Image.fromarray(mask)
        with col2:
            st.header('Masked Region')
            st.image(mask, use_column_width=True)

        if st.button('generate', key=1):
            with st.spinner('Generating'):
                inpainted_img = inpainting_image(img, mask)

            st.markdown('Generated Image')
            inpainted_img = Image.fromarray(inpainted_img.astype(np.uint8))
            inpainted_img = inpainted_img.resize((width, height))
            st.image(inpainted_img, use_column_width=True)
            
    except:
        pass



