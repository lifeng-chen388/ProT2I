from __future__ import annotations

import abc
import math
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
import cv2
from utils.ptp_utils import aggregate_attention
from utils import vis_utils


class P2PAttnProcessor:
    def __init__(self, controller_list: List[AttentionControl], place_in_unet: str, do_cfg: bool, attention_store:AttentionStore=None):
        super().__init__()
        self.controller_list = controller_list
        self.place_in_unet = place_in_unet
        self.do_cfg = do_cfg
        self.attention_store = attention_store

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        is_nursing:bool=False,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # debug used
        if encoder_hidden_states is not None:
            pass


        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Added
        if is_nursing and self.attention_store:
            self.attention_store(attention_probs,is_cross,self.place_in_unet)
        elif self.controller_list:
            for i, controller in enumerate(self.controller_list):
                if self.do_cfg:
                    h = attention_probs.shape[0]
                    head_dim = attention_probs.shape[0] // 2 // (len(self.controller_list)+1)
                    controller(attention_probs[h//2+i*head_dim: h//2+(i+2)*head_dim], is_cross, self.place_in_unet)
                else:
                    h = attention_probs.shape[0]
                    head_dim = attention_probs.shape[0] // (len(self.controller_list)+1)
                    controller(attention_probs[i*head_dim: (i+2)*head_dim], is_cross, self.place_in_unet)
                
        value = attn.to_v(encoder_hidden_states)
        value = attn.head_to_batch_dim(value)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def create_controller(
    prompts: List[str], cross_attention_kwargs: Dict, num_inference_steps: int,
      tokenizer, device, attn_res) -> AttentionControl:
      
    edit_type = cross_attention_kwargs.get("edit_type", None)
    local_blend_words = cross_attention_kwargs.get("local_blend_words", None)
    n_cross_replace = cross_attention_kwargs.get("n_cross_replace", 0.4)
    n_self_replace = cross_attention_kwargs.get("n_self_replace", 0.4)
    lb_threshold = cross_attention_kwargs.get("lb_threshold", 0.6)
    lb_res = cross_attention_kwargs.get("lb_res",(32,32))
    lb_prompt = cross_attention_kwargs.get("lb_prompt",None)
    run_name = cross_attention_kwargs.get("run_name","masks")
    save_map = cross_attention_kwargs.get("save_map",False)
    is_nursing = cross_attention_kwargs.get("is_nursing", False)

    # only refine
    if edit_type == "refine" and local_blend_words is None:
        return AttentionRefine(
            prompts, num_inference_steps, n_cross_replace, n_self_replace, tokenizer=tokenizer, device=device, attn_res=attn_res
        )

    # refine + localblend
    if edit_type == "refine" and local_blend_words is not None:
        if is_nursing and lb_prompt:
            lb = LocalBlend(lb_prompt, local_blend_words, tokenizer=tokenizer, device=device, attn_res=lb_res,threshold=lb_threshold, run_name=run_name, save_map=save_map)
        else:
            lb = LocalBlend(prompts, local_blend_words, tokenizer=tokenizer, device=device, attn_res=lb_res, threshold=lb_threshold, run_name=run_name, save_map=save_map)
        return AttentionRefine(
            prompts, num_inference_steps, n_cross_replace, n_self_replace, lb, tokenizer=tokenizer, device=device, attn_res=attn_res
        )

    return EmptyControl(attn_res=attn_res)


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):                    # attn represents the attention maps
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, attn_res=None):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.attn_res = attn_res



class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= self.attn_res[0]**2:  # avoid memory overhead
            if is_cross:                       # only save the cross-attention maps, avoiding memory overhead
                self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=None):
        super(AttentionStore, self).__init__(attn_res)
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        

class EmptyControl(AttentionStore):
    def __init__(self, attn_res=None):
        super(EmptyControl, self).__init__(attn_res=attn_res)


def test_mask(mask):
    contains_false = not torch.all(mask.bool())
    print(f"Contains false: {contains_false}")


class LocalBlend:
    def __init__(
        self, prompts: List[str], words: str, device, tokenizer, threshold=0.6, attn_res=(32, 32), run_name="masks", save_map:bool=False,
    ):

        self.max_num_words = 77
        self.attn_res = attn_res
        self.words = words
        self.threshold = threshold
        self.run_name = run_name
        self.save_map = save_map

        prompt_ids = tokenizer.encode(prompts[0], add_special_tokens=False)
        w_token_ids = tokenizer.encode(words, add_special_tokens=False)
        self.token_indices = []
        i = 0
        while i < len(prompt_ids):
            if prompt_ids[i : i + len(w_token_ids)] == w_token_ids:
                self.token_indices.extend(range(i, i + len(w_token_ids)))
                i += len(w_token_ids)
                break
            else:
                i += 1

        if not self.token_indices:
            print(f"Warning: cannot find sub-token(s) for '{words}' in the prompt.")

    def __call__(self, x_t, attention_store: AttentionStore, cur_step: int, bs:int):
        print(f"Processing words: {self.words}")
        from_where = ("up", "mid","down")
        select = 0

        attention_maps = aggregate_attention(
            attention_store, 
            self.attn_res, 
            from_where, 
            True, 
            select, 
            bs,
        ).detach().cpu()

        if len(self.token_indices) > 0:
            cross_map = attention_maps[:, :, [x+1 for x in self.token_indices]].mean(dim=-1)  # +1 is <sot> token, shape = (H, W)
        else:
            cross_map = torch.zeros_like(attention_maps[:, :, 0])  # shape = (H, W)

        cross_map = cross_map.reshape(1, 1, *cross_map.shape)  # (1,1,H,W)
        cross_map = cross_map.cuda()

        mask = F.interpolate(cross_map, size=(x_t.shape[2], x_t.shape[3]), mode='bilinear')  # (1,1,h,w)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8) 
        mean_mask = mask.mean()
        mask = mask.gt(mask.min() + (mask.max()-mask.min())* self.threshold).float()

        if self.save_map:
            vis_utils.save_binary_masks(
                attention_masks=mask.squeeze().cpu(),
                word=self.words,
                res=self.attn_res[0],
                save_path=f"./{self.run_name}/intermediate/mask_{cur_step}-{self.words}.jpg",
                txt_under_img=True,
            )

        true_rate = mask.sum() / (mask.numel())
        print(f"True rate: {true_rate}")

        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t, attention_store:AttentionStore=None,):
        if self.local_blend is not None and attention_store is not None:
            if self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
                x_t = self.local_blend(x_t, attention_store,self.cur_step, bs=2)         # after a step, apply local blend
        elif self.local_blend is not None:
            if self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
                x_t = self.local_blend(x_t, self,self.cur_step, bs=2)      
        return x_t

    def replace_self_attention(self, attn_base, attn_replace):
        if attn_replace.shape[2] <= self.attn_res[0]**2:
        # if attn_replace.shape[2] == 32**2:
            return attn_base.unsqueeze(0).expand(attn_replace.shape[0], *attn_base.shape)
        else:
            return attn_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]       # (len(des_prompts),1,1,77), means the whether each word is replaced at cur step
                attn_replace_new = (
                    self.replace_cross_attention(attn_base, attn_replace) * alpha_words
                    + (1 - alpha_words) * attn_replace      # alpha_words replace happens in specific words and steps, others keep the original dest attention
                )
                attn[1:] = attn_replace_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_replace)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
        tokenizer,
        device,
        attn_res=None,
    ):
        super(AttentionControlEdit, self).__init__(attn_res=attn_res)
        # add tokenizer and device here

        self.tokenizer = tokenizer
        self.device = device

        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, self.tokenizer
        ).to(self.device)
        if isinstance(self_replace_steps, float):
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend



class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):                              # not used most time
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)               
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        tokenizer=None,
        device=None,
        attn_res=None
    ):
        super(AttentionRefine, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device, attn_res
        )
        self.mapper, alphas = get_refinement_mapper(prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])



### util functions for all Edits
def update_alpha_time_word(
    alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor] = None
):
    if isinstance(bounds, float):
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts, num_steps, cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]], tokenizer, max_num_words=77
):
    if not isinstance(cross_replace_steps, dict):
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():           # different replace steps for different words
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]  # get the indices of the words in setting bound
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


### util functions for LocalBlend and ReplacementEdit
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if isinstance(word_place, str):
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif isinstance(word_place, int):
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):         # if  the word is not split into multiple tokens
                ptr += 1
                cur_len = 0
    return np.array(out)

### util functions for RefinementEdit
class ScoreParams:
    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match


def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i - 1])
            y_seq.append(y[j - 1])
            i = i - 1
            j = j - 1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append("-")
            y_seq.append(y[j - 1])
            j = j - 1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i - 1])
            y_seq.append("-")
            i = i - 1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x: str, y: str, tokenizer, max_len=77):
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[: mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0] :] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas


def get_refinement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers, alphas = [], []
    for i in range(1, len(prompts)):
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)






