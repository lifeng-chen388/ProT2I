import ast
from typing import Any, Callable, List
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
from torchvision import transforms as T
from ProT2I.processors import *
from utils.gaussian_smoothing import GaussianSmoothing
import numpy as np



def get_centroid(attn_map: torch.Tensor) -> torch.Tensor:
    """
    attn_map: h*w*token_len
    """
    h, w, seq_len = attn_map.shape

    attn_x, attn_y = attn_map.sum(0), attn_map.sum(1)  # w|h seq_len
    x = torch.linspace(0, 1, w).to(attn_map.device).reshape(w, 1)
    y = torch.linspace(0, 1, h).to(attn_map.device).reshape(h, 1)

    centroid_x = (x * attn_x).sum(0) / attn_x.sum(0)  # seq_len
    centroid_y = (y * attn_y).sum(0) / attn_y.sum(0)  # bs seq_len
    centroid = torch.stack((centroid_x, centroid_y), -1)  # (seq_len, 2)
    return centroid

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def preprocess_prompts(prompts):
    if isinstance(prompts, (list, tuple)):
        return [p.lower().strip().strip(".").strip() for p in prompts]
    elif isinstance(prompts, str):
        return prompts.lower().strip().strip(".").strip()
    else:
        raise NotImplementedError


class ProT2IPipeline(StableDiffusionXLPipeline):
    r"""
    Args:
    Prompt-to-Prompt-Pipeline for text-to-image generation using Stable Diffusion. This model inherits from
    [`StableDiffusionPipeline`]. Check the superclass documentation for the generic methods the library implements for
    all the pipelines (such as downloading or saving, running on a particular device, etc.)
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents. scheduler
        ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    _optional_components = ["safety_checker", "feature_extractor"]

    def _entropy_loss(
        self,
        prompt,
        attention_store: AttentionStore,
        subject_words: List[str],
        attention_res: Tuple[int] = (32, 32),
        Use_AdaPose: bool = False,
        angle_loss_weight: float = 1.0,
    ):
        """
        Aggregates the attention for each token and computes the max activation value for each token to alter.
        If Use_AdaPose=True, then constrain the coordinates of the selected subattention maps.
        Instead of the manually specified points, we use “find the coordinates of the brightest area of each patch”.
        """
        # ============  1. Attention map aggregation ============
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0,
        )  # [h, w, 77]

        loss = 0

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        attention_for_texts = attention_maps[:,:,1:len(tokens)+1]
        text_cross_map = F.softmax(attention_for_texts / 0.5, dim=-1)
        
        subject_word_indices_list = []
        for w in subject_words:
            if w:
                w_token_ids = self.tokenizer.encode(w, add_special_tokens=False)
                i = 0
                matched_indices = []
                while i < len(tokens):
                    if tokens[i : i + len(w_token_ids)] == w_token_ids:
                        matched_indices.extend(range(i, i + len(w_token_ids)))
                        i += len(w_token_ids)
                        break
                    else:
                        i += 1
                subject_word_indices_list.append(matched_indices)

        word_attn_maps = []
        for idx_list in subject_word_indices_list:
            if len(idx_list) == 0:

                word_attn_maps.append(torch.zeros_like(text_cross_map[:, :, 0]))
            else:
                sub_map = text_cross_map[:, :, idx_list]          # [h, w, n_subtokens]
                word_map = sub_map.mean(dim=-1)                   # [h, w]
                word_attn_maps.append(word_map)

        cross_map = torch.stack(word_attn_maps, dim=-1)
        # --------------------------------------------------------------------

        cross_map = (cross_map - cross_map.amin(dim=(0, 1), keepdim=True)) / (
            cross_map.amax(dim=(0, 1), keepdim=True) - cross_map.amin(dim=(0, 1), keepdim=True)
        )
        cross_map = cross_map / cross_map.sum(dim=(0, 1), keepdim=True)

        # Entropy loss
        loss = loss - 2 * (cross_map * torch.log(cross_map + 1e-5)).sum()

        # ============ 2. Adapos loss, coordinate constraints on the attention graph of the specified Token ============
        # if Use_AdaPose:
        #     vis_map = cross_map.permute(2, 0, 1) 

        #     n_positions = vis_map.shape[0]

        #     brightest_patches = []
        #     max_vals = []

        #     for i in range(n_positions):
        #         sub_map = vis_map[i]
        #         # Gaussian Smoothing
        #         sub_map_smoothed = cv2.GaussianBlur(sub_map.detach().cpu().float().numpy(), (5, 5), 0)
        #         sub_map_smoothed = torch.tensor(sub_map_smoothed, device=sub_map.device)

        #         # Find the brightest area (max value and max position).
        #         max_val, idx = sub_map_smoothed.view(-1).max(dim=0)
        #         y, x = divmod(idx.item(), sub_map.shape[1])
        #         pos = torch.tensor([x, y], device=sub_map.device, dtype=torch.float32)
        #         brightest_patches.append(pos)
        #         max_vals.append(max_val)

        #     # ============ 3. Calculate the center of mass ============
        #     curr_map = torch.stack([vis_map[i] for i in range(n_positions)])  # [K, h, w]
        #     curr_map = curr_map.permute(1, 2, 0)  # [h, w, K]
        #     pair_pos = (get_centroid(curr_map) * attention_res[0]).to(vis_map.device)

        #     for i, pos in enumerate(brightest_patches):
        #         loss += (0.1 * (pair_pos[i] - pos) ** 2).mean()

            # ============  4 Add angular similarity constraints to prevent overlap (worked not remarkbly)============ 
            # for i in range(n_positions):
            #     for j in range(i + 1, n_positions):
            #         vec_i = brightest_patches[i] / (torch.norm(brightest_patches[i])+1e-8)
            #         vec_j = brightest_patches[j] / (torch.norm(brightest_patches[j])+1e-8)
                    
            #         cos_sim = torch.dot(vec_i, vec_j) 
                    
            #         angle_penalty = torch.clamp(cos_sim, min=0)  
            #         loss += angle_loss_weight * (1 - angle_penalty)

        return loss
    

    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=False, allow_unused=True
        )[0]
        if grad_cond is None:
            grad_cond = torch.zeros_like(latents)
        latents = latents - step_size * grad_cond
        del grad_cond
        return latents


    def _perform_iterative_refinement_step(
        self,
        prompt,
        latents: torch.Tensor,
        subject_words: List[str],
        threshold: float,
        text_embeddings: torch.Tensor,
        attention_store: AttentionStore,
        step_size: float,
        t: int,
        attention_res: Tuple[int] = (32,32),
        max_refinement_steps: List[int] = [3, 3],
        Use_AdaPose: bool = False,
        angle_loss_weight: float = 1.0,
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code and text embedding according to our loss objective until the given threshold is reached for all tokens.
        """
        ratio = t / 1000
        if ratio > 0.9:
            max_refinement_steps = max_refinement_steps[0]
        if ratio <= 0.9:
            max_refinement_steps = max_refinement_steps[1]
        print("=====================================================================")
        iteration = 0
        while True:
            iteration += 1
            torch.cuda.empty_cache()
            with torch.enable_grad():
                latents = latents.clone().detach().requires_grad_(True)

                cross_attention_kwargs={"is_nursing":True}
                noise_pred_text = self.unet(
                    latents.unsqueeze(0),
                    t,
                    encoder_hidden_states=text_embeddings,
                    added_cond_kwargs=self.added_cond_kwargs2,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                self.unet.zero_grad()
                del noise_pred_text

                loss = self._entropy_loss(prompt,attention_store,subject_words,attention_res,Use_AdaPose=Use_AdaPose, angle_loss_weight=angle_loss_weight)

                if loss != 0: 
                    latents = self._update_latent(latents, loss, step_size)
                
                print(f"Iteration {iteration}, Loss:{loss:.4f}")
                # print cuda memory usage
                # print(torch.cuda.memory_summary(device=self.device, abbreviated=True))
                if loss < threshold:
                    break
                if iteration >= max_refinement_steps:
                    print(
                        f"Entropy loss optimization Exceeded max number of iterations ({max_refinement_steps}) "
                    )
                    break
        # print Step iteration and loss
        print(f"Step Interation: {iteration} | loss: {loss}")
        print("=====================================================================")
        return latents.detach()

    def check_inputs(
            self,
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = {},
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        attn_res=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

                The keyword arguments to configure the edit are:
                - edit_type (`str`). The edit type to apply. Can be either of `replace`, `refine`, `reweight`.
                - n_cross_replace (`int`): Number of diffusion steps in which cross attention should be replaced
                - n_self_replace (`int`): Number of diffusion steps in which self attention should be replaced
                - local_blend_words(`List[str]`, *optional*, default to `None`): Determines which area should be
                  changed. If None, then the whole image can be changed.
                - equalizer_words(`List[str]`, *optional*, default to `None`): Required for edit type `reweight`.
                  Determines which words should be enhanced.
                - equalizer_strengths (`List[float]`, *optional*, default to `None`) Required for edit type `reweight`.
                  Determines which how much the words in `equalizer_words` should be enhanced.

            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        nursing_threshold = cross_attention_kwargs.get("nursing_threshold",None)
        scale_range = cross_attention_kwargs.get("scale_range",None)
        scale_factor = cross_attention_kwargs.get("scale_factor",None)
        attention_refinement_steps = cross_attention_kwargs.get("max_refinement_steps",None)
        subject_words = cross_attention_kwargs.get("subject_words",None)
        is_nursing = True if subject_words else False
        use_adapose = cross_attention_kwargs.get("use_AdaPose",None)
        angle_loss_weight = cross_attention_kwargs.get("angle_loss_weight",None)


        # get prompts   
        if cross_attention_kwargs.get("subprompts", None):
            prompt = preprocess_prompts(cross_attention_kwargs.get("subprompts", None))

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attn_res = attn_res

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        device = self._execution_device
        scale_range = np.linspace(
            scale_range[0], scale_range[1], len(self.scheduler.timesteps)
        )


        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]


        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if len(latents)>1:
            # init all latents to be the same
            latents = latents[0].repeat(batch_size, 1, 1, 1)


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim # if none should be changed to enc1
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        added_cond_kwargs2 = {
            "text_embeds": add_text_embeds[batch_size:batch_size+1],
            "time_ids": add_time_ids[batch_size:batch_size+1],
        }

        self.added_cond_kwargs2 = added_cond_kwargs2

        # Add
        self.controller_list = cross_attention_kwargs.get("set_controller", None)
        attention_store = self.register_attention_control(self.controller_list, do_classifier_free_guidance, is_nursing=is_nursing)  # add attention controller

        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i in nursing_threshold.keys() and is_nursing:
                    nursed_latents = self._perform_iterative_refinement_step(
                        prompt=prompt[0],
                        latents=latents[0],
                        subject_words=subject_words,
                        threshold=nursing_threshold[i],
                        text_embeddings=prompt_embeds[batch_size:batch_size+1] if do_classifier_free_guidance else prompt_embeds[0],
                        attention_store=attention_store,
                        step_size=scale_factor*np.sqrt(scale_range[i]),
                        t=t,
                        attention_res=attn_res,
                        max_refinement_steps=attention_refinement_steps,
                        Use_AdaPose=use_adapose,
                        angle_loss_weight=angle_loss_weight,
                    ).unsqueeze(0)
                    latents = torch.cat([nursed_latents,latents[1:]], dim=0)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds,
                                       added_cond_kwargs=added_cond_kwargs, ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # step callback
                for i, controller in enumerate(self.controller_list):
                    if is_nursing:
                        if controller.local_blend is not None:
                            controller.local_blend.attn_res = self.attn_res 
                        latents[i:i+2,:] = controller.step_callback(latents[i:i+2,:], self.controller_list[0])
                    else:
                        latents[i:i+2,:] = controller.step_callback(latents[i:i+2,:])

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 8. Post-processing
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return (StableDiffusionXLPipelineOutput(images=image), self.controller_list, attention_store)


    def register_attention_control(self, controller_list, do_classifier_free_guidance, is_nursing=False):
        attn_procs = {}
        cross_att_count = 0
        if is_nursing:
            attentionStore = AttentionStore(self.attn_res)
        else:
            attentionStore = None
        for name in self.unet.attn_processors.keys():
            None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PAttnProcessor(controller_list=controller_list, place_in_unet=place_in_unet, do_cfg=do_classifier_free_guidance,attention_store=attentionStore)

        self.unet.set_attn_processor(attn_procs)
        if controller_list:
            for controller in controller_list:
                controller.num_att_layers = cross_att_count
        if is_nursing:
            attentionStore.num_att_layers = cross_att_count
        return attentionStore

