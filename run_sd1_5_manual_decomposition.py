import os
import torch
from ProT2I.prot2i_pipeline_sd_1_5 import ProT2IPipeline
from ProT2I.processors import create_controller


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
    # sentences decomposition
    nps = [
    "a man stand in front of a car", 
    "a man in a red tracksuit stand in front of a car",
    "a man in a red tracksuit stand in front of a blue car", 
    "a man in a red tracksuit stand in front of a blue sport car", 
    "a man in a red tracksuit wearing sunglasses stand in front of a blue sport car",
    "a man in a red tracksuit wearing sunglasses stand in front of a blue sport car with a rainbow in the background",
    "a man in a red tracksuit wearing sunglasses and a hat stand in front of a blue sport car with a rainbow in the background",
    "a man in a red tracksuit wearing sunglasses and a brown hat stand in front of a blue sport car with a rainbow in the background",
    ]
    lb_words_list = [("man","man"),  ("car","car"), ("car","car"), ("man","man"), None, ("man","man"), ("hat","hat")]

    prompts = ["a man in a red tracksuit wearing sunglasses and a brown hat stand in front of a blue sport car with a rainbow in the background"]

    assert len(lb_words_list)+1 == len(nps), f"The length of controller list is {len(lb_words_list)}, but nps {len(nps)}"

    # make controllers
    n_cross_replace = 0.0
    n_self_replace = 0.8
    attn_res = (32,32)
    lb_threshold = 0.4
    controller_list = []
    controller_np = [[nps[i-1],nps[i]] for i in range(1, len(nps))]
    for i in range(len(lb_words_list)):
        controller_kwargs = {
            "edit_type": "refine",
            "local_blend_words": lb_words_list[i],
            "equalizer_words": None,
            "equalizer_strengths": None,
            "n_cross_replace": {"default_": n_cross_replace},
            "n_self_replace": n_self_replace,
            "lb_threshold": lb_threshold,
        }
        controller = create_controller(prompts=controller_np[i], cross_attention_kwargs=controller_kwargs, num_inference_steps=50,tokenizer=pipe.tokenizer, device=device, attn_res=attn_res, structured_cond=None)
        controller_list.append(controller)

    # if you mannualy decompose the complex prompt, you can use `cross_attention_kwargs` to input the nps and controllers
    cross_attention_kwargs = {
        "nps": nps,
        "set_controller": controller_list,
        }

    print(type(pipe))
    image = pipe(prompts, cross_attention_kwargs=cross_attention_kwargs, num_inference_steps=50, num_images_per_prompt=1,generator=g_cpu)[0]

    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(image['images']):
        img.save(os.path.join(output_dir, f"P2P_REAL_{i+1}_SD_1_5.jpg"))
        print(f"Saved image: {os.path.join(output_dir, f'P2P_REAL_{i+1}_SD1_5.jpg')}")

