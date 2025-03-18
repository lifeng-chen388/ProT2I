import os
import random
import time
import torch
import gradio as gr
from ProT2I.prot2i_pipeline_sdxl import ProT2IPipeline
from ProT2I.processors import create_controller
from PIL import Image
import numpy as np
import difflib

_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">ProT2I for SDXL</h1>
</div>

⭐⭐⭐**Tips:**
- ⭐`Sub-prompts:` Enter the decomposed sub-prompts, one per line.
- ⭐`Subject Masking Words:` Enter the subject words for each sub-prompt, one per line. (Leave it a blank line, if you want to remove all attributes firstly.)
- ⭐We provide an example at the bottom that you can try.
- ⭐For attributes overflow, you can adaptively increase the `Threshold Value` for mask extraction.
'''

def create_placeholder_image():
    return Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 255)

def get_diff_string(str1, str2):
    """
    `str1` and `str2` are two strings.
    This function returns the difference between the two strings as a string.
    """
    diff = difflib.ndiff(str1.split(), str2.split())
    added_parts = [word[2:] for word in diff if word.startswith('+ ')]  # get added parts
    return ' '.join(added_parts)

def init_pipeline():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    if device.type == "cuda":
        pipe = ProT2IPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            torch_dtype=torch.float16, 
            use_safetensors=True,
            variant='fp16'
        ).to(device)
        pipe.enable_model_cpu_offload()
    else:
        pipe = ProT2IPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            torch_dtype=torch.float32, 
            use_safetensors=False
        ).to(device)
    return pipe, device

def process_image(
    sub_prompts,
    lb_words,
    n_self_replace,
    lb_threshold,
    attention_res,
    use_nurse,
    centroid_alignment,
    width,
    height,
    inference_steps,
    seed
):
    try:
        # Initialize pipeline
        pipe, device = init_pipeline()
        
        # Process sub-prompts
        sps = [prompt.strip() for prompt in sub_prompts.split('\n') if prompt.strip()]
        
        # Process semantic masking words
        nps = []
        for word in lb_words.split('\n'):
            if word.strip():
                nps.append(word.strip())
            else:
                nps.append(None)
        
        # Validate inputs
        if len(nps) + 1 != len(sps):
            placeholder_image = create_placeholder_image()
            return placeholder_image, [placeholder_image] * 3, f"Error: Number of semantic masks ({len(nps)}) should be one less than number of sub-prompts ({len(sps)})"
        
        # Set fixed parameters from config
        guidance_scale = 7.5
        n_cross = 0.0
        scale_factor = 1750
        scale_range = (1.0, 0.0)
        angle_loss_weight = 0.0
        max_refinement_steps = [6, 3]
        nursing_thresholds = {
            0: 26, 1: 25, 2: 24, 3: 23, 4: 22.5,
            5: 22, 6: 21.5, 7: 21, 8: 21, 9: 21
        }
        save_cross_attention_maps = False
        
        if seed == -1:
            seed = random.randint(0, 1000000)
        g_cpu = torch.Generator().manual_seed(seed)
        
        # Create controllers
        controller_list = []
        run_name = f'runs-SDXL/{time.strftime("%Y%m%d-%H%M%S")}-{seed}'
        controller_np = [[sps[i-1], sps[i]] for i in range(1, len(sps))]
        
        # Prepare status messages
        status_messages = [f"seed: {seed}"]
        
        for i in range(len(controller_np)):
            controller_kwargs = {
                "edit_type": "refine",
                "local_blend_words": nps[i],
                "n_cross_replace": {"default_": n_cross},
                "n_self_replace": float(n_self_replace),
                "lb_threshold": float(lb_threshold)+1,
                "lb_prompt": [sps[0]]*2,
                "is_nursing": use_nurse,
                "lb_res": (int(attention_res), int(attention_res)),
                "run_name": run_name,
                "save_map": save_cross_attention_maps,
            }
            
            # Get difference between sps[i+1] and sps[i]
            if nps[i] is None:
                subject_strig = ",".join(nps[1:])
                status_messages.append(f"Remove attributes from {subject_strig}")
            else:
                diff_str = get_diff_string(sps[i], sps[i+1])
                if diff_str:
                    status_messages.append(f"Add {diff_str} to {nps[i]}")
            
            controller = create_controller(
                prompts=controller_np[i],
                cross_attention_kwargs=controller_kwargs,
                num_inference_steps=inference_steps,
                tokenizer=pipe.tokenizer,
                device=device,
                attn_res=(int(attention_res), int(attention_res))
            )
            controller_list.append(controller)
        
        # Set up cross attention kwargs
        cross_attention_kwargs = {
            "subprompts": sps,
            "set_controller": controller_list,
            "subject_words": nps if use_nurse else None,
            "nursing_threshold": nursing_thresholds,
            "max_refinement_steps": max_refinement_steps,
            "scale_factor": scale_factor,
            "scale_range": scale_range,
            "centroid_alignment": centroid_alignment,
            "angle_loss_weight": angle_loss_weight,
        }
        
        # Generate images
        output = pipe(
            prompt=sps[-1],  # Use the last sub-prompt as the final prompt
            width=width,
            height=height,
            cross_attention_kwargs=cross_attention_kwargs,
            num_inference_steps=inference_steps,
            num_images_per_prompt=1,
            generator=g_cpu,
            attn_res=(int(attention_res), int(attention_res)),
        )[0]
        
        return output["images"][-1], output["images"], "\n".join(status_messages)
        
    except Exception as e:
        placeholder_image = create_placeholder_image()
        return placeholder_image, [placeholder_image] * 3, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks() as iface:
    gr.Markdown(_HEADER_)
    
    with gr.Row():
        with gr.Column(scale=1):
            sub_prompts = gr.Textbox(
                lines=5,
                label="Sub-prompts",
                placeholder="Enter sub-prompts, one per line..."
            )
            
            lb_words = gr.Textbox(
                lines=4,
                label="Subject masking words",
                placeholder="Enter subject words, one per line..."
            )
            
            n_self_replace = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.8,
                step=0.1,
                label="Percetange of self-attention map substitution steps"
            )
            
            lb_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Threshold for latent mask extraction of subject words"
            )
            
            attention_res = gr.Number(
                label="Attention map resolution",
                value=32
            )
            
            with gr.Row():
                use_nurse = gr.Checkbox(
                    label="Use attention nursing",
                    value=True
                )
                
                centroid_alignment = gr.Checkbox(
                    label="Use centroid alignment",
                    value=True
                )
            
            with gr.Row():
                width = gr.Number(
                    label="Width",
                    value=1024
                )
                
                height = gr.Number(
                    label="Height",
                    value=1024
                )
            
            inference_steps = gr.Number(
                label="Inference steps",
                value=20
            )
            
            seed = gr.Number(
                label="Seed (-1 for random)",
                value=-1
            )
            
            generate_btn = gr.Button("Generate Image")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image")
            
            with gr.Accordion("Progressive Generation Process", open=False):
                gallery = gr.Gallery(
                    label="Generation Steps",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=3,
                    height="auto"
                )
            
            output_status = gr.Textbox(label="Status", lines=4)
    
    # Connect the generate button to the process_image function
    generate_btn.click(
        fn=process_image,
        inputs=[
            sub_prompts,
            lb_words,
            n_self_replace,
            lb_threshold,
            attention_res,
            use_nurse,
            centroid_alignment,
            width,
            height,
            inference_steps,
            seed
        ],
        outputs=[output_image, gallery, output_status]
    )
    
    # Examples
    example_data = [
        [
            "a car and a bench\na blue car and a bench\na car and a red bench",
            "car\nbench",
            0.1,
            20,
            0
        ],
        [
            "In a cyberpunk style city night, a hound dog is standing in front of a sports car\nVan Gogh style hound dog\nLego-style sports car",
            "dog\ncar",
            0.25,
            20,
            -1
        ],
        [
            "A sketch-style robot is leaning a oil-painting style tree\nA robot is leaning a tree\nA sketch-style robot is leaning a tree\nA robot is leaning a oil-painting style tree",
            "\nrobot\ntree",
            0.25,
            20,
            -1
        ],
        [
            "a man wearing a red hat and blue tracksuit is standing in front of a green sports car\na man wearing a hat and tracksuit is standing in front of a sports car\na man wearing a red hat and tracksuit is standing in front of a sports car\na man wearing a hat and blue tracksuit is standing in front of a sports car\na man wearing a hat and tracksuit is standing in front of a green sports car",
            "\nrobot\ntree",
            0.25,
            20,
            6
        ],

    ]
    
    gr.Examples(
        examples=example_data,
        inputs=[
            sub_prompts,
            lb_words,
            lb_threshold,
            inference_steps,
            seed
        ]
    )

if __name__ == "__main__":
    iface.launch(share=True, server_port=8080)