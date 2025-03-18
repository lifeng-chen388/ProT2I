import os
import random
import time
import torch
from ProT2I.prot2i_pipeline_sdxl import ProT2IPipeline
from ProT2I.processors import create_controller
from utils.nlp_utils import split_prompt
from config import *



if __name__ == "__main__":
    config = NLPConfig()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pipe = ProT2IPipeline.from_pretrained(config.model_path, use_safetensors=True, variant='fp16').to(torch.float16)
    pipe.enable_model_cpu_offload() 
    
    seeds = config.seeds

    for seed in seeds:
        if seed == 0:
            seed = random.randint(0,1000000) 
        g_cpu = torch.Generator().manual_seed(seed)
        if config.use_nlp:
            sps, nps = split_prompt(config.prompt)
        elif config.sps and config.nps:
            sps = config.sps
            nps = config.nps
        
        if config.use_remove_then_add and config.use_nlp:
            sps = [config.prompt] + sps
            nps = [None] + nps


        assert len(nps)+1 == len(sps)

        if config.use_nurse:
            nursing_tokens = nps
        else:
            nursing_tokens = None

        # make controllers
        controller_list = []
        run_name = config.output_path+f'/{time.strftime("%Y%m%d-%H%M%S")}-{seed}'
        controller_np = [[sps[i-1],sps[i]] for i in range(1, len(sps))]
        for i in range(len(controller_np)):
            controller_kwargs = {
                "edit_type": "refine",
                "local_blend_words": nps[i],
                "n_cross_replace": {"default_": config.n_cross},
                "n_self_replace": config.n_self,
                "lb_threshold": config.lb_t+1,
                "lb_prompt": [sps[0]]*2,
                "is_nursing": config.use_nurse,
                "lb_res": config.attention_res,
                "run_name": run_name,
                "save_map": config.save_cross_attention_maps,
            }
            controller = create_controller(
                prompts=controller_np[i],
                cross_attention_kwargs=controller_kwargs,
                num_inference_steps=config.n_inference_steps,
                tokenizer=pipe.tokenizer,
                device=device, 
                attn_res=config.attention_res
            )
            controller_list.append(controller)


        # if you mannualy decompose the complex prompt, you can use `cross_attention_kwargs` to input the sps and controllers
        cross_attention_kwargs = {
            "subprompts": sps,
            "set_controller": controller_list,
            "subject_words": nursing_tokens,
            "nursing_threshold": config.nursing_thresholds,
            "max_refinement_steps": config.max_refinement_steps,
            "scale_factor": config.scale_factor,
            "scale_range":config.scale_range,
            "centroid_alignment":config.centroid_alignment,
            "angle_loss_weight":config.angle_loss_weight,
            }

        print(type(pipe))
        image = pipe(prompt=config.prompt, 
                    width = config.width,
                    height = config.height,
                    cross_attention_kwargs=cross_attention_kwargs, 
                    num_inference_steps=config.n_inference_steps, 
                    num_images_per_prompt=1,
                    generator=g_cpu,
                    attn_res = config.attention_res,
                    )[0]

        output_dir = f"./{run_name}/generated_images/"
        os.makedirs(output_dir, exist_ok=True)
        for i, img in enumerate(image['images']):
            img.save(os.path.join(output_dir, f"Generated{i+1}_SDXL.jpg"))
            print(f"Saved image: {os.path.join(output_dir, f'Generated{i+1}_SDXL.jpg')}")

