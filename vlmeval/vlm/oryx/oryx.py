import re
import os
import argparse
import torch
import math
import numpy as np
from typing import Dict, Optional, Sequence, List
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


class Oryx1_5(BaseModel):
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
        video_file = None
        system_prompt = "You are a helpful assistant."
        for item in message:
            # if item['role'] == 'system':
            #     system_prompt = item['value']
            if item['type'] == 'video':
                video_file = item['value']
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

        output_ids = self.model.generate(
            inputs=input_ids,
            images=video_processed,
            images_highres=video_processed,
            modalities=modalities,
            do_sample=True,
            temperature=0.2,
            num_beams=1,
            max_new_tokens=16,
            use_cache=True,
        )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        return outputs

        # if dataset == "LongVideoBench":
        #     num_options = 5
        # else:
        #     num_options = 4
        # option_ids = self.tokenizer(
        #     [chr(ord('A') + i) for i in range(num_options)] + [f" {chr(ord('A') + i)}" for i in range(num_options)], 
        #     return_tensors="pt"
        # ).input_ids.flatten().to(input_ids.device)

        # return text_outputs, extra

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_video(message, dataset)
