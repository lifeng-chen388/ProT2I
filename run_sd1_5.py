import os
import torch
from prot2i_pipeline_sd_1_5 import ProT2IPipeline
from processors import create_controller


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

if __name__ == "__main__":
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == "cuda":
        pipe = ProT2IPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE",
                                                    torch_dtype=torch.float16, use_safetensors=True).to(device)
    else:
        pipe = ProT2IPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE",
                                                    torch_dtype=torch.float32, use_safetensors=False).to(device)
    seed = 3254
    g_cpu = torch.Generator().manual_seed(seed)
    prompt = "a man in a red tracksuit wearing sunglasses and a brown hat stand in front of a blue sport car with a rainbow in the background"
    image = pipe(prompt, num_inference_steps=50, num_images_per_prompt=1,generator=g_cpu)[0]

    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(image['images']):
        img.save(os.path.join(output_dir, f"REAL_{i+1}_SD_1_5.jpg"))
        print(f"Saved image: {os.path.join(output_dir, f'REAL_{i+1}_SD1_5.jpg')}")

