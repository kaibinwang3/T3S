import re
import os
import argparse
import torch
import math
import numpy as np
from typing import Dict, Optional, Sequence, List, Union
import transformers
from transformers import AutoConfig
from PIL import Image
from decord import VideoReader, cpu
import copy

from oryx.conversation import conv_templates, SeparatorStyle
from oryx.model.builder import load_pretrained_model
from oryx.utils import disable_torch_init
from oryx.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_anyres_video_genli
from oryx.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from oryx.model.language_model.oryx_qwen import (
    GenerateOutput
)

from ..base import BaseModel
from ...smp import *


from thop import profile
import json


class ForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids,
                past_key_values, inputs_embeds, use_cache, cache_position,
                cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k):
        all_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
        )
        return all_outputs


def pack_embeds_with_position_ids_for_flash_attn(embeds_list, device):
    """
    将embeds列表pack成一个序列，并生成对应的position_ids，用于Flash Attention
    
    Args:
        embeds_list: List[torch.Tensor] - 每个tensor是一个序列的embeds [seq_len, hidden_size]
        device: 设备
    
    Returns:
        dict: 包含packed inputs和Flash Attention所需参数的字典
    """
    # 获取每个序列的长度
    seq_lengths = [embeds.shape[0] for embeds in embeds_list]
    original_lengths = seq_lengths.copy()
    
    # 计算cumulative lengths (从0开始)
    cu_seq_lens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device=device)
    cu_seq_lens[1:] = torch.cumsum(torch.tensor(seq_lengths, device=device), dim=0)
    
    # Pack inputs_embeds - 直接连接所有embeds
    packed_embeds = torch.cat(embeds_list, dim=0)  # [total_length, hidden_size]
    
    # 生成position_ids - 每个序列从0开始
    position_ids_list = []
    for seq_len in seq_lengths:
        position_ids_list.append(torch.arange(seq_len, dtype=torch.long, device=device))
    
    # Pack position_ids
    packed_position_ids = torch.cat(position_ids_list, dim=0)  # [total_length]
    
    # 创建packed attention_mask - 全1，因为Flash Attention通过cu_seq_lens处理边界
    total_length = packed_embeds.shape[0]
    packed_attention_mask = torch.ones(total_length, dtype=torch.long, device=device)
    
    # 最大序列长度
    max_length = max(seq_lengths)
    
    return {
        'inputs_embeds': packed_embeds.unsqueeze(0),          # [1, total_length, hidden_size]
        'position_ids': packed_position_ids.unsqueeze(0),     # [1, total_length]
        'attention_mask': packed_attention_mask.unsqueeze(0), # [1, total_length]
        'cu_seq_lens_q': cu_seq_lens,                         # [num_seqs + 1]
        'cu_seq_lens_k': cu_seq_lens,                         # [num_seqs + 1]
        'max_length_q': max_length,                           # int
        'max_length_k': max_length,                           # int
        'original_lengths': original_lengths                  # List[int]
    }


def unpack_model_outputs(outputs, original_lengths):
    """
    将packed的模型输出解包回原始的list格式
    
    Args:
        outputs: 模型输出，包含last_hidden_state等
        original_lengths: 原始序列长度列表
    
    Returns:
        List[torch.Tensor]: 解包后的输出列表
    """
    # 获取last_hidden_state
    if hasattr(outputs, 'last_hidden_state'):
        hidden_states = outputs.last_hidden_state
    else:
        # 如果是tuple或其他格式，取第一个元素
        hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    
    # 移除batch维度
    hidden_states = hidden_states.squeeze(0)  # [total_length, hidden_size]
    
    unpacked_outputs = []
    start_idx = 0
    
    for length in original_lengths:
        end_idx = start_idx + length
        seq_hidden = hidden_states[start_idx:end_idx]  # [seq_len, hidden_size]
        unpacked_outputs.append(seq_hidden)
        start_idx = end_idx
    
    return unpacked_outputs


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


def uniform_sample_video(video_file, nframe):
    vr = VideoReader(video_file, ctx=cpu(0))
    total_frame_num = len(vr)
    nframe = min(total_frame_num, nframe)
    uniform_sampled_frames = np.linspace(
        0, total_frame_num - 1, nframe,
        dtype=int
    )
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(frame) for frame in spare_frames]
    return video


def random_sample_video(video_file, nframe):
    vr = VideoReader(video_file, ctx=cpu(0))
    total_frame_num = len(vr)
    nframe = min(total_frame_num, nframe)
    frame_idx = np.random.choice(total_frame_num, nframe, replace=False)
    frame_idx.sort()
    frame_idx = frame_idx.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(frame) for frame in spare_frames]
    return video


class ProdecodeOryxRandframeLogitsPackTimeit(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(
        self,
        model_path="/mnt/afs/wangkaibin/models/Oryx-1.5-7B",
        model_config=None,
        **kwargs
    ):
        super().__init__()
        self.model_config = model_config

        model_path = "/mnt/afs/wangkaibin/models/Oryx-1.5-7B"
        model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map="cuda:0",
            overwrite_config=dict(
                mm_resampler_type="dynamic_compressor",
            )
        )
        self.model.to('cuda').eval()

    @torch.no_grad
    def generate_inner_video(self, message, dataset=None):
        all_alpha = self.model_config.alpha
        bsz = len(all_alpha)

        video_file = None
        for item in message:
            if item['type'] == 'video':
                video_file = item['value']
                break
        assert video_file is not None

        nframe = self.model_config.nframe
        all_video = []
        for _ in range(bsz):
            video = random_sample_video(video_file, nframe)
            nframe = len(video)
            all_video.extend(video)

        all_video_processed = []
        all_modalities = ["video_long"] * len(all_video)
        for idx, frame in enumerate(all_video):
            self.image_processor.do_resize = False
            self.image_processor.do_center_crop = False
            frame = process_anyres_video_genli(frame, self.image_processor)
            all_video_processed.append(frame.unsqueeze(0))
        all_video_processed = torch.cat(all_video_processed, dim=0).bfloat16().cuda()

        question = ''
        system_prompt = "You are a helpful assistant."
        for item in message:
            if item['type'] == 'video':
                question += '<image>' * nframe
            else:
                question += item['value']
        qwen_message = []
        qwen_message.append({'from': 'human','value': question})
        qwen_message.append({'from': 'gpt','value': None})

        input_ids = preprocess_qwen(
            qwen_message, self.tokenizer, has_image=True, system_message=system_prompt
        ).cuda()

        all_input_ids = [input_ids.clone().squeeze(0) for _ in range(bsz)]

        if dataset == "LongVideoBench":
            num_options = 5
        else:
            num_options = 4
        option_ids = self.tokenizer(
            [chr(ord('A') + i) for i in range(num_options)],
            return_tensors="pt"
        ).input_ids.flatten().to(input_ids.device)

        packed_inputs = self.OryxMetaForCausalLM_prepare_inputs_labels_for_multimodal(
            self.model,
            all_input_ids, all_alpha, 
            all_modalities, all_video_processed, all_video_processed
        )

        if not hasattr(self, "perf_record"):
            self.perf_record = []
            self.forward_wrapper = ForwardWrapper(self.model)

        inputs = (
            None,
            packed_inputs['attention_mask'],
            packed_inputs['position_ids'],
            None,
            packed_inputs['inputs_embeds'],
            True,
            packed_inputs['position_ids'],
            packed_inputs['cu_seq_lens_q'],
            packed_inputs['cu_seq_lens_k'],
            packed_inputs['max_length_q'],
            packed_inputs['max_length_k'],
        )

        torch.cuda.synchronize(0)
        start_time = time.perf_counter()
        logits = self.forward_wrapper(*inputs)
        torch.cuda.synchronize(0)
        end_time = time.perf_counter()

        flops, _ = profile(self.forward_wrapper, inputs=inputs, verbose=False)

        self.perf_record.append({
            "wall_clock": end_time - start_time,
            "flops": flops
        })

        warmup_step = self.model_config.timeit.warmup_step
        test_step = self.model_config.timeit.test_step
        if len(self.perf_record) == warmup_step + test_step:
            self.perf_record = self.perf_record[warmup_step:]
            timeit_result = {
                "wall_clock": sum(it["wall_clock"] for it in self.perf_record) / test_step,
                "gflops": sum(it["flops"] for it in self.perf_record) / test_step / 1e9
            }
            with open(self.model_config.timeit.output_path, 'w') as f:
                json.dump(timeit_result, f, indent=2)
            sys.exit(0)

        response = 'A'
        return response

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_video(message, dataset)

    def OryxMetaForCausalLM_prepare_inputs_labels_for_multimodal(
        wrapper,
        self, input_ids, alpha,
        modalities, images, images_highres=None
    ):
        device = input_ids[0].device

        video_idx_in_batch = []
        for modal in range(len(modalities)):
            if 'video' in modalities[modal]:
                video_idx_in_batch.append(modal)

        aimg = images[-1]
        lowres_img = []
        for idx, img_feat in enumerate(images):
            if idx in video_idx_in_batch:
                img_feat = aimg.new(1, 3, 128, 128).fill_(0)
            lowres_img.append(img_feat)
        lowres_img_features, lowres_img_sizes = self.get_model().get_vision_tower()(lowres_img)

        highres_img_features = []
        highres_img_sizes = []
        for idx, img_feat in enumerate(images_highres):
            if img_feat.ndim == 5:
                img_feat = img_feat.squeeze(1)
            highres_img_feature, highres_img_size = self.get_model().get_vision_tower()(img_feat)
            highres_img_features.append(highres_img_feature)
            highres_img_sizes.append(highres_img_size)

        image_features = []
        for idx in range(len(modalities)):
            img_feat_highres, img_size_highres = self.get_model().vision_resampler(
                highres_img_features[idx],
                modalities[idx],
                highres_img_sizes[idx]
            )
            img_feat_lowres, img_size_lowres = self.get_model().vision_resampler(
                lowres_img_features[idx],
                modalities[idx],
                lowres_img_sizes[idx]
            )
            img_feat = self.get_model().mm_projector(
                img_feat_lowres,
                img_size_lowres,
                img_feat_highres,
                img_size_highres,
                modalities[idx]
            )
            image_features.append(img_feat.flatten(0, 1))

        new_input_embeds = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                cur_image_idx += 1
                continue

            num_image_tokens = sum(len(image_features[cur_image_idx + i]) for i in range(num_images))
            num_selection = int(num_image_tokens * alpha[batch_idx])
            selection_mask = torch.randperm(num_image_tokens, device=device) < num_selection
            mask_begin = 0

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_input_ids_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    mask_end = mask_begin + len(cur_image_features)
                    cur_image_features = cur_image_features[selection_mask[mask_begin:mask_end]]
                    cur_new_input_embeds.append(cur_image_features)
                    cur_image_idx += 1
                    mask_begin = mask_end
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)

            new_input_embeds.append(cur_new_input_embeds)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        modality_max_length = getattr(self.config, 'modality_max_length', None)

        if modality_max_length is None or modality_max_length == "None":
            if tokenizer_model_max_length is not None:
                new_input_embeds =[x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        else:
            modality_max_length = ast.literal_eval(modality_max_length)
            modality_max_length_dict = {"image": modality_max_length[0], "text": modality_max_length[1], "video": modality_max_length[2]}
            new_input_embeds =[x[: modality_max_length_dict[modality]] for x, modality in zip(new_input_embeds, modalities)]

        device = new_input_embeds[0].device
        packed_inputs = pack_embeds_with_position_ids_for_flash_attn(new_input_embeds, device)
        return packed_inputs
