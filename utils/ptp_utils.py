import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from einops import rearrange
import math
from typing import Union, Tuple, List

def aggregate_attention(attention_store, res: List[int], from_where: List[str], is_cross: bool, select: int, batch_size: int = 1):
    out = []
    attention_maps = attention_store.get_average_attention()
    res_W, res_H = res
    num_pixels = res_H*res_W
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(batch_size, -1, res_W, res_H, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def aggregate_attention_intermediate(
        attention_store,
        res: int,
        from_where: List[str],
        from_res: List[int],
        is_cross: bool,
        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = [r ** 2 for r in from_res]
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] in num_pixels:
                cur_res = int(math.sqrt(item.shape[1]))
                cross_maps = item.reshape(1, -1, cur_res, cur_res, item.shape[-1])[select]
                cross_maps = rearrange(cross_maps, 'b h w c -> b c h w')
                cross_maps = torch.nn.functional.interpolate(cross_maps, size=(res,res),mode='nearest', )
                cross_maps = rearrange(cross_maps, 'b c h w -> b h w c')
                out.append(cross_maps)
    out = torch.cat(out, dim=0) #[40,16,16,77]
    out = out.sum(0) / out.shape[0]
    return out

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        if isinstance(images[0], Image.Image):
            h, w = images[0].size
            images = [np.array(img) for img in images]
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)
