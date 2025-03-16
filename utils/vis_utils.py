import math
import os
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch

from utils import ptp_utils
import matplotlib.pyplot as plt


def save_binary_masks(
    attention_masks,
    word: str,
    res: int = 16,
    orig_image=None,
    save_path=None,
    txt_under_img:bool=False,
):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(attention_masks, torch.Tensor):
        attention_masks = attention_masks.squeeze().cpu().numpy()
    elif isinstance(attention_masks, np.ndarray):
        attention_masks = attention_masks.squeeze()
    else:
        raise TypeError("attention_masks must be torch.Tensor or np.ndarray")

    mask = (attention_masks > 0).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode='L')
    mask_image = mask_image.resize((256, 256), resample=Image.NEAREST)  #
    mask_image = mask_image.convert('RGB')
    mask_np = np.array(mask_image)
    if txt_under_img:
        mask_with_text = ptp_utils.text_under_image(mask_np, word)
        final_image = Image.fromarray(mask_with_text)
    else:
        final_image = Image.fromarray(mask_np)
    final_image = final_image.resize((256, 256), resample=Image.BILINEAR)
    if save_path:
        final_image.save(save_path)




def show_cross_attention(prompt: str,
                         attention_store,
                         tokenizer,
                         res: int,
                         from_where: List[str],
                         subject_words: List[str],
                         bs:int=2,
                         select: int = 0,
                         orig_image=None,
                         text_under_img:bool=True):


    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, bs).detach().cpu()
    images = []
    token_texts = [decoder(int(token)) for token in tokens]
    token_indices = [i for i, text in enumerate(token_texts) if text in subject_words]
    last_idx = len(token_texts) - 1

    # show spatial attention for indices of tokens to strengthen
    for i in token_indices:
        image = attention_maps[:, :, i]         # （32,32）
        image = show_image_relevance(image, orig_image)
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
        if text_under_img:
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)

    ptp_utils.view_images(np.stack(images, axis=0))
        



def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=32):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])       # （1，1，16，16）
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')    # (1,1,256,256)
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)                         # (256,256)   
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image

def aggregate_attention(attention_store, res: int, from_where: List[str], is_cross: bool, select: int, bs:int = 2):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(bs, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def show_self_attention(attention_stores, from_where: str, layers:int):
    top_components = []
    # the first self attention map
    first_attention_map = attention_stores[0].get_average_attention()[from_where][layers][:8].mean(dim=0)
    U, S, V = torch.svd(first_attention_map.to(torch.float32))
    top_U = U[:, :6]
    top_components.append(top_U)
    for i, attention_store in enumerate(attention_stores, start=0):
        attention_map = (attention_store.get_average_attention()[from_where][layers][8:]).mean(dim=0).to(torch.float32)
        U, S, V = torch.svd(attention_map)
        top_U = U[:,:6]
        top_components.append(top_U)

    for batch_idx, components in enumerate(top_components):
        plt.figure(figsize=(24, 4))
        for comp_idx in range(6):
            plt.subplot(1, 6, comp_idx + 1)
            component = components[:,comp_idx].reshape(16,16).to('cpu')
            plt.imshow(component, cmap='viridis')
            # plt.colorbar()
            plt.axis('off')
            plt.title(f'prompt {batch_idx + 1} Top {comp_idx + 1}')
        plt.tight_layout()
        plt.show()