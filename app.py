import os
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
- ⭐`Semantic Masking Words:` Enter the subject words of the corresponding added attributes of the sub-prompts, one subject per line. (Enter None if no specific subject is needed for circumstances like adding background or layout initializaiton)
- ⭐we provide some examples in the bottom, you can try these example prompts first.
- ⭐If invovled context-aware layout initialization, you can unfold the "Progressive Generation Process" to compare with the first image, which is the baseline output.
'''

def create_placeholder_image():
    # create placeholder image
    return Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 255)

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
            use_safetensors=True
        ).to(device)
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
    n_cross_replace,
    n_self_replace,
    attn_scale,
    inference_steps,
    lb_threshold,
    seed
):
    try:
        # Initialize pipeline
        pipe, device = init_pipeline()
        
        # Process sub-prompts
        nps = [prompt.strip() for prompt in sub_prompts.split('\n') if prompt.strip()]
        
        # Process local blend word pairs
        lb_words_list = []
        for word in lb_words.split('\n'):
            if word.strip() == "None" or word.strip() == "none" or word.strip() == "NONE":
                lb_words_list.append(None)
            elif word.strip():
                lb_words_list.append((word.strip(), word.strip()))
        print(lb_words_list)

        
         # Validate inputs
        if len(lb_words_list) + 1 != len(nps):
            placeholder_image = create_placeholder_image()  # create placeholder image
            return placeholder_image, [placeholder_image] * len(nps), f"Error: Number of word pairs ({len(lb_words_list)}) should be one less than number of sub-prompts ({len(nps)})"
        
        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()
        # Set up generator
        g_cpu = torch.Generator().manual_seed(int(seed))
        
        # Create controllers
        controller_list = []
        controller_np = [[nps[i-1], nps[i]] for i in range(1, len(nps))]
        status_messages = []
        for i in range(len(lb_words_list)):
            controller_kwargs = {
                "edit_type": "refine",
                "local_blend_words": lb_words_list[i],
                "equalizer_words": None,
                "equalizer_strengths": None,
                "n_cross_replace": {"default_": float(n_cross_replace)},
                "n_self_replace": float(n_self_replace),
                "lb_threshold": float(lb_threshold),
            }

            # Get difference between nps[i+1] and nps[i]
            diff_str = get_diff_string(nps[i], nps[i+1])
            if diff_str and lb_words_list[i] is not None:
                status_messages.append(f"Add '{diff_str}' to '{lb_words_list[i][0]}'.")
            elif diff_str:
                status_messages.append(f"Add '{diff_str}'.")
            elif lb_words_list[0] is None and i == 0:
                status_messages.append("Context-aware layout initialization.")

            # context-aware layout initialization branch
            if lb_words_list[0] is None and i == 0:
                controller_kwargs = {
                    "edit_type": "refine",
                    "local_blend_words": lb_words_list[i],
                    "equalizer_words": None,
                    "equalizer_strengths": None,
                    "n_cross_replace": {"default_": 0.5},
                    "n_self_replace": 0.4,
                    "lb_threshold": lb_threshold,
                }

            controller = create_controller(
                prompts=controller_np[i],
                cross_attention_kwargs=controller_kwargs,
                num_inference_steps=inference_steps,
                tokenizer=pipe.tokenizer,
                device=device,
                attn_res=(int(attn_scale), int(attn_scale)),
                structured_cond=None
            )
            controller_list.append(controller)

        # Set up cross attention kwargs
        cross_attention_kwargs = {
            "nps": nps,
            "set_controller": controller_list,
        }

        # Generate images
        prompts = [nps[-1]]  # Use the last sub-prompt as the final prompt
        output = pipe(
            prompts,
            cross_attention_kwargs=cross_attention_kwargs,
            num_inference_steps=inference_steps,
            num_images_per_prompt=1,
            generator=g_cpu
        )[0]
        images = output["images"]
        
        return images[-1], images, "\n".join(status_messages)
        
    except Exception as e:
        placeholder_image = create_placeholder_image()
        return placeholder_image, [placeholder_image] * len(nps), f"Error: {str(e)}"



# Create Gradio interface
with gr.Blocks() as iface:
    gr.Markdown(_HEADER_)
    
    with gr.Row():
        with gr.Column():
            sub_prompts = gr.Textbox(
                lines=8,
                label="Sub-prompts",
                placeholder="Enter sub-prompts, one per line..."
            )
            lb_words = gr.Textbox(
                lines=7,
                label="Semantic Masking Words",
                placeholder="subject1\nsubject2\nsubject3\n..."
            )
            with gr.Accordion("Advanced Settings", open=False):
                n_cross_replace = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="Percetage of Cross-Attention Subsitution steps"
                )
                n_self_replace = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label="Percetage of Self-Attention Subsitution steps"
                )

                attn_scale = gr.Number(label="Manipulation Attention Resolution", value=32)

                inference_steps = gr.Number(label="Number of Inference Steps", value=50)
                    
                lb_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.55,
                    step=0.05,
                    label="Threshold for Instance-wise Semantic Mask Extraction"
                )
                seed = gr.Number(label="Seed (-1 for random)", value=-1)
        
        with gr.Column():
            final_output = gr.Image(label="Generated Image")
            with gr.Accordion("Progressive Generation Process", open=False):
                gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=3,
                    height="auto"
                )
                status_output = gr.Textbox(label="Status", lines=4)
    
    generate_btn = gr.Button("Generate Images")
    generate_btn.click(
        fn=process_image,
        inputs=[
            sub_prompts,
            lb_words,
            n_cross_replace,
            n_self_replace,
            attn_scale,
            inference_steps,
            lb_threshold,
            seed
        ],
        outputs=[final_output ,gallery, status_output]
    )
    
    with gr.Row(), gr.Column():
        gr.Markdown("## Examples")
        # Example data for the Gradio interface
        example_data = [
            # Example 1
            [
                "A magical deer with iridescent fur in shades of mint green, lilac, and pearl white, antlers adorned with glowing golden vines, standing in an enchanted forest of coral pink trees under a teal sky.\n"
                "A magical deer with iridescent fur in shades of mint green, lilac, and pearl white, antlers adorned with vines, standing in an enchanted forest of trees under a sky.\n"
                "A magical deer with iridescent fur in shades of mint green, lilac, and pearl white, antlers adorned with glowing golden vines, standing in an enchanted forest of trees under a sky.\n"
                "A magical deer with iridescent fur in shades of mint green, lilac, and pearl white, antlers adorned with glowing golden vines, standing in an enchanted forest of coral pink trees under a sky.\n"
                "A magical deer with iridescent fur in shades of mint green, lilac, and pearl white, antlers adorned with glowing golden vines, standing in an enchanted forest of coral pink trees under a teal sky.",
                "None\n"
                "antlers\n"
                "trees\n"
                "sky.",
                3163

            ],
            # Example 2
            [
                "A cream-colored bunny mascot with floppy ears, wearing a pastel teal scarf , rose gold sunglasses , and a lavender backpack , standing in a glowing coral-pink meadow under a twilight sky.\n"
                "A cream-colored bunny mascot with floppy ears, wearing a scarf ,sunglasses , and a backpack , standing in a glowing coral-pink meadow under a twilight sky.\n"
                "A cream-colored bunny mascot with floppy ears, wearing a pastel teal scarf ,sunglasses , and a backpack , standing in a glowing coral-pink meadow under a twilight sky.\n"
                "A cream-colored bunny mascot with floppy ears, wearing a pastel teal scarf , rose gold sunglasses , and a backpack , standing in a glowing coral-pink meadow under a twilight sky.\n"
                "A cream-colored bunny mascot with floppy ears, wearing a pastel teal scarf , rose gold sunglasses , and a lavender backpack , standing in a glowing coral-pink meadow under a twilight sky.",
                "None\n"
                "scarf\n"
                "sunglasses\n"
                "backpack",
                3176

            ],
            # Example 3
            [
                "A snow-white penguin mascot with a smooth body, wearing a light yellow beanie , a pearlescent green satchel , and a shiny magenta bowtie , waddling through an icy landscape under an aurora of pastel colors.\n"
                "A snow-white penguin mascot with a smooth body, wearing a beanie , a satchel , and a bowtie , waddling through an icy landscape under an aurora of pastel colors.\n"
                "A snow-white penguin mascot with a smooth body, wearing a light yellow beanie , a satchel , and a bowtie , waddling through an icy landscape under an aurora of pastel colors.\n"
                "A snow-white penguin mascot with a smooth body, wearing a light yellow beanie , a pearlescent green satchel , and a bowtie , waddling through an icy landscape under an aurora of pastel colors.\n"
                "A snow-white penguin mascot with a smooth body, wearing a light yellow beanie , a pearlescent green satchel , and a shiny magenta bowtie , waddling through an icy landscape under an aurora of pastel colors.",
                "None\n"
                "beanie\n"
                "satchel\n"
                "bowtie",
                3169
            ],
            # Example 4
            [
                "A golden beige tiger mascot with a sleek body, donning a dreamy lavender jacket , sky-blue sunglasses , and a coral-pink belt , lounging near a pool of glowing turquoise water in a fantasy jungle.\n"
                "A golden beige tiger mascot with a sleek body, donning a jacket , sunglasses , and a belt , lounging near a pool of glowing turquoise water in a fantasy jungle.\n"
                "A golden beige tiger mascot with a sleek body, donning a dreamy lavender jacket , sunglasses , and a belt , lounging near a pool of glowing turquoise water in a fantasy jungle.\n"
                "A golden beige tiger mascot with a sleek body, donning a dreamy lavender jacket , sky-blue sunglasses , and a belt , lounging near a pool of glowing turquoise water in a fantasy jungle.\n"
                "A golden beige tiger mascot with a sleek body, donning a dreamy lavender jacket , sky-blue sunglasses , and a coral-pink belt , lounging near a pool of glowing turquoise water in a fantasy jungle.",
                "None\n"
                "jacket\n"
                "sunglasses\n"
                "belt",
                3173
            ],
            # Example 5
            [
            "a red teddy bear wearing a green trucksuit and yellow hat is eating a blue cake with a black cat in the background\n"
            "a red teddy bear wearing a trucksuit and hat is eating a cake with a black cat in the background\n"
            "a red teddy bear wearing a green trucksuit and hat is eating a cake with a black cat in the background\n"
            "a red teddy bear wearing a green trucksuit and yellow hat is eating a cake with a black cat in the background\n"
            "a red teddy bear wearing a green trucksuit and yellow hat is eating a blue cake with a black cat in the background",
            "None\ntrucksuit\nhat\ncake",
            3127
        ],
        ]
        example_data_one_per_one = [
            # Example 1
            [
                "A man in a blue jersey and black helmet is riding a mountain bike with a pink frame and purple wheels on the road\n"
                "A man is riding a mountain bike on the road\n"
                "A man is riding a mountain bike with a pink frame and purple wheels on the road\n"
                "A man in a blue jersey and black helmet is riding a mountain bike with a pink frame and purple wheels on the road",
                "None\nbike\nman",
                906840

            ],
            # Example 2
            [
                "A brown Ragdoll cat wearing a grey hat is eating a pink cake with blue icing accents\n"
                "A Ragdoll cat is eating a cake\n"
                "A brown Ragdoll cat wearing a grey hat is eating a pink cake\n"
                "A brown Ragdoll cat wearing a grey hat is eating a pink cake with blue icing accents",
                "None\ncat\ncake",
                487915
            ]
        ]

        examples = gr.Examples(
            examples=example_data,
            label="One attribute added one time",
            inputs=[sub_prompts, lb_words, seed],
        )
        examples2 = gr.Examples(
            examples=example_data_one_per_one,
            label="Multiple attributes belonging to one subject added  one time",
            inputs=[sub_prompts, lb_words, seed],
        )



if __name__ == "__main__":
    iface.launch(share=True, server_port=6006)