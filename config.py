from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List

@dataclass
class StyleConfig:
    prompt: str = "A sketch-style robot is leaning a oil-painting style tree"

    width: int = 1024

    height: int = 1024

    model_path: str = "SG161222/RealVisXL_V4.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False

    # decompose the prompt freely, only worked when use_nlp set False
    sps:List[str] = field(
        default_factory=lambda: [
            "A sketch-style robot is leaning a oil-painting style tree",
            "A robot is leaning a tree",
            "A sketch-style robot is leaning a tree",
            "A robot is leaning a oil-painting style tree"
        ]
    )
    
    # 'None' represents don't use semantic masks
    nps:List[str] = field(
        default_factory=lambda: [
            None,
            "robot",
            "tree"
        ]
    )
    # Activate it when you want to remove all attributes firstly, then add them one by one. Use it with use_nlp=True
    # This would add one more branch, means more memory cost.
    use_remove_then_add:bool = False

    # 0 represents random
    seeds: List[int] = field(default_factory=lambda:[0]*10)
    # Path to save all outputs to
    output_path:str = "runs-SDXL/style-test"
    # Number of denoising steps
    n_inference_steps: int = 20
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: tuple[int] = (32,32)
    # percentage of inference steps with cross attention map swap, from 0.0 to 1.0
    n_cross: float = 0.0
    # percentage of inference steps with self attention map swap, from 0.0 to 1.0.
    # Higher n_self means more consistent image layout, but lower fielity.
    n_self: float = 0.8
    # threshold for instance-wise semantic masking, from -1.0 to 1.0
    lb_t: float = 0.25

    # attention nursing work only when use_nurse=True
    use_nurse:bool = True
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    nursing_thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
        }
    )
    # maximum attention refinement steps
    max_refinement_steps: List[int] = field(default_factory=lambda: [6,3])

    use_adapose: bool = True
    # angular loss weight for avoiding attention map overlap
    angle_loss_weight: float = 0.0
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 1750
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = True


@dataclass
class ColorConfig:
    prompt: str = "a man wearing a red hat and blue tracksuit is standing in front of a green sports car"

    width: int = 1024

    height: int = 1024

    model_path: str = "SG161222/RealVisXL_V4.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = False

    # decompose the prompt freely, only worked when use_nlp set False
    sps:List[str] = field(
        default_factory=lambda: [
            "a man wearing a red hat and blue tracksuit is standing in front of a green sports car",
            "a man wearing a hat and tracksuit is standing in front of a sports car",
            "a man wearing a red hat and tracksuit is standing in front of a sports car",
            "a man wearing a hat and blue tracksuit is standing in front of a sports car",
            "a man wearing a hat and tracksuit is standing in front of a green sports car",
        ]
    )
    
    # 'None' represents don't use semantic masks
    nps:List[str] = field(
        default_factory=lambda: [
            None,
            "hat",
            "tracksuit",
            "car"
        ]
    )
    # Activate it when you want to remove all attributes firstly, then add them one by one. Use it with use_nlp=True
    # This would add one more branch, means more memory cost.
    use_remove_then_add:bool = False

    # 0 represents random
    seeds: List[int] = field(default_factory=lambda: [1,2,3,4,5,6])
    # Path to save all outputs to
    output_path:str = "runs-SDXL/color-test"
    # Number of denoising steps
    n_inference_steps: int = 20
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: tuple[int] = (32,32)
    # percentage of inference steps with cross attention map swap, from 0.0 to 1.0
    n_cross: float = 0.0
    # percentage of inference steps with self attention map swap, from 0.0 to 1.0.
    # Higher n_self means more consistent image layout, but lower fielity.
    n_self: float = 0.8
    # threshold for instance-wise semantic masking, from -1.0 to 1.0
    lb_t: float = 0.25

    # attention nursing work only when use_nurse=True
    use_nurse:bool = True
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    nursing_thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
        }
    )
    # maximum attention refinement steps
    max_refinement_steps: List[int] = field(default_factory=lambda: [6,3])

    use_adapose: bool = True
    # angular loss weight for avoiding attention map overlap
    angle_loss_weight: float = 0.0
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 1750
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = True



@dataclass
# a config example for use nlp toolkit for prompt decomposition
# But only support easy prompt!
class NLPConfig:
    prompt: str = "A blue bench and a red car"

    width: int = 1024

    height: int = 1024

    model_path: str = "SG161222/RealVisXL_V4.0"
    # whether use nlp model to spilt the prompt
    use_nlp: bool = True

    # decompose the prompt freely, only worked when use_nlp set False
    sps:List[str] = field(
        default_factory=lambda: []
    )
    
    # 'None' represents don't use semantic masks
    nps:List[str] = field(
        default_factory=lambda: []
    )
    # Activate it when you want to remove all attributes firstly, then add them one by one.
    # This would add one more branch, means more memory cost.
    use_remove_then_add:bool = True

    # 0 represents random
    seeds: List[int] = field(default_factory=lambda: range(0,10))
    # Path to save all outputs to
    output_path:str = "runs-SDXL/nlp"
    # Number of denoising steps
    n_inference_steps: int = 20
    # Text guidance scale
    guidance_scale: float = 7.5
    # Resolution of UNet to compute attention maps over
    attention_res: tuple[int] = (32,32)
    # percentage of inference steps with cross attention map swap, from 0.0 to 1.0
    n_cross: float = 0.0
    # percentage of inference steps with self attention map swap, from 0.0 to 1.0
    n_self: float = 0.8
    # threshold for instance-wise semantic masking, from 0 to 1.0
    lb_t: float = 0.25

    # attention nursing work only when use_nurse=True
    use_nurse:bool = True
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    nursing_thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            0: 26,
            1: 25,
            2: 24,
            3: 23,
            4: 22.5,
            5: 22,
        }
    )
    # maximum attention refinement steps
    max_refinement_steps: List[int] = field(default_factory=lambda: [6,3])

    use_adapose: bool = True
    # angular loss weight for avoiding attention map overlap
    angle_loss_weight: float = 0.0
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 1750
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.0))
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = True


