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
from thop import profile
import json

from ..base import BaseModel
from ...smp import *


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


class ForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, position_ids, attention_mask, inputs_embeds):
        # print(inputs_embeds.shape)
        # breakpoint()
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            cache_position=position_ids,
        )
        return outputs


class BaselineOryxLogitsTimeit(BaseModel):
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

    def generate_inner_video(self, message, dataset=None):
        video_file = None
        for item in message:
            if item['type'] == 'video':
                video_file = item['value']
                break
        assert video_file is not None
    
        nframe = self.model_config.nframe
        video_processed = []
        video = uniform_sample_video(video_file, nframe)
        nframe = len(video)
        modalities = ["video_long"] * nframe
        for idx, frame in enumerate(video):
            self.image_processor.do_resize = False
            self.image_processor.do_center_crop = False
            frame = process_anyres_video_genli(frame, self.image_processor)
            video_processed.append(frame.unsqueeze(0))
        video_processed = torch.cat(video_processed, dim=0).bfloat16().cuda()

        question = ''
        system_prompt = "You are a helpful assistant."
        for item in message:
            if item['type'] == 'video':
                question += '<image>' * nframe
            else:
                question += item['value']
        assert video_file is not None
        qwen_message = []
        qwen_message.append({'from': 'human','value': question})
        qwen_message.append({'from': 'gpt','value': None})

        input_ids = preprocess_qwen(
            qwen_message, self.tokenizer, has_image=True, system_message=system_prompt
        ).cuda()

        _, all_position_ids, all_attention_mask, _, all_inputs_embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, 
            video_processed, modalities,
            image_sizes=None,
            images_highres=video_processed
        )

        inputs = (
            all_position_ids,
            all_attention_mask,
            all_inputs_embeds
        )

        if not hasattr(self, "perf_record"):
            self.perf_record = []
            self.forward_wrapper = ForwardWrapper(self.model)

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

        outputs = 'A'
        return outputs

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_video(message, dataset)

    @torch.no_grad()
    def OryxQwenForCausalLM_generate(
        wrapper,
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        images_highres: Optional[List[torch.FloatTensor]] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, 
             position_ids, 
             attention_mask, 
             _, 
             inputs_embeds, 
             _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, 
                images, modalities,
                image_sizes=image_sizes,
                images_highres=images_highres
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        outputs = self.__call__(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            cache_position=position_ids,
        )

        return outputs
