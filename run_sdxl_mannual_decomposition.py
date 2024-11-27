import os
import torch
from ProT2I.prot2i_pipeline_sdxl import ProT2IPipeline
from ProT2I.processors import create_controller


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

if __name__ == "__main__":
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
    # stabilityai/stable-diffusion-xl-base-1.0
    # SG161222/RealVisXL_V4.0
    # SG161222/Realistic_Vision_V5.1_noVAE
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == "cuda":
        pipe = ProT2IPipeline.from_pretrained("SG161222/RealVisXL_V4.0",
                                                    torch_dtype=torch.float16, use_safetensors=True).to(device)
    else:
        pipe = ProT2IPipeline.from_pretrained("SG161222/RealVisXL_V4.0",
                                                    torch_dtype=torch.float32, use_safetensors=False).to(device)
    # pipe.enable_model_cpu_offload()
    seed = 3172
    g_cpu = torch.Generator().manual_seed(seed)

    # 分词结果
    nps = [
        "A golden beige tiger mascot with a sleek body, donning a dreamy lavender jacket , sky-blue sunglasses , and a coral-pink belt , lounging near a pool of glowing turquoise water in a fantasy jungle.",
        "A golden beige tiger mascot with a sleek body, donning a jacket , sunglasses , and a belt , lounging near a pool of glowing turquoise water in a fantasy jungle.",
        "A golden beige tiger mascot with a sleek body, donning a dreamy lavender jacket , sunglasses , and a belt , lounging near a pool of glowing turquoise water in a fantasy jungle.",
        "A golden beige tiger mascot with a sleek body, donning a dreamy lavender jacket , sky-blue sunglasses , and a belt , lounging near a pool of glowing turquoise water in a fantasy jungle.",
        "A golden beige tiger mascot with a sleek body, donning a dreamy lavender jacket , sky-blue sunglasses , and a coral-pink belt , lounging near a pool of glowing turquoise water in a fantasy jungle."
    ]
    prompts="A golden beige tiger mascot with a sleek body, donning a dreamy lavender jacket , sky-blue sunglasses , and a coral-pink belt , lounging near a pool of glowing turquoise water in a fantasy jungle."
    lb_words_list = [None,["jacket,","jacket"],["sunglasses","sunglasses"],["belt","belt"]]



    n_self_replace = 0.8
    n_cross_replace = 0.0
    # 定制controller
    attn_res = (32,32)      # good
    lb_threshold = 0.5
    controller_np = [[nps[i-1],nps[i]] for i in range(1, len(nps))]
    controller_list = []
    for i, np_pair in enumerate(controller_np):
        if i == 0:
            controller_kwargs = {
                "edit_type": "refine",
                "local_blend_words": lb_words_list[i],
                "equalizer_words": None,
                "equalizer_strengths": None,
                "n_cross_replace": {"default_": 0.5},
                "n_self_replace": 0.4,
                "lb_threshold": lb_threshold,
            }
            controller = create_controller(prompts=np_pair, cross_attention_kwargs=controller_kwargs, num_inference_steps=50,tokenizer=pipe.tokenizer, device=device, attn_res=attn_res)
            controller_list.append(controller)
        else:
            controller_kwargs = {
                "edit_type": "refine",
                "local_blend_words":lb_words_list[i],
                "equalizer_words": None,
                "equalizer_strengths": None,
                "n_cross_replace": {"default_": n_cross_replace},
                "n_self_replace": n_self_replace,
                "lb_threshold": lb_threshold,
            }
            controller = create_controller(prompts=np_pair, cross_attention_kwargs=controller_kwargs, num_inference_steps=50,tokenizer=pipe.tokenizer, device=device, attn_res=attn_res)
            controller_list.append(controller)

    assert len(controller_list)+1 == len(nps), "Manually setting controller numbers are not right!"

    # if you mannualy decompose the complex prompt, you can use `cross_attention_kwargs` to input the nps and controllers
    cross_attention_kwargs = {
        "nps": nps,
        "set_controller": controller_list,
        }

    print(type(pipe))
    image = pipe(prompts, cross_attention_kwargs=cross_attention_kwargs, 
                num_inference_steps=50, num_images_per_prompt=1,
                width=1024, height=1024, generator=g_cpu)[0]

    output_dir = f"generated_images"
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(image['images']):
        img.save(os.path.join(output_dir, f"{i+1}.jpg"))
        print(f"Saved image: {os.path.join(output_dir, f'{i+1}.jpg')}")

