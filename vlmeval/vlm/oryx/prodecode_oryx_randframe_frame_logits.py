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


class ProdecodeOryxRandframeFrameLogits(BaseModel):
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

        all_alpha = [
            self.model_config.alpha1,
            self.model_config.alpha2,
        ]

        nframe = self.model_config.nframe
        all_video_processed = []
        all_input_ids = []

        for i in range(len(all_alpha)):
            alpha = all_alpha[i]
            nframe_i = int(nframe * alpha)
            video = random_sample_video(video_file, nframe_i)
            nframe_i = len(video)

            for idx, frame in enumerate(video):
                self.image_processor.do_resize = False
                self.image_processor.do_center_crop = False
                frame = process_anyres_video_genli(frame, self.image_processor)
                all_video_processed.append(frame.unsqueeze(0))

            question = ''
            system_prompt = "You are a helpful assistant."
            for item in message:
                if item['type'] == 'video':
                    question += '<image>' * nframe_i
                else:
                    question += item['value']
            qwen_message = []
            qwen_message.append({'from': 'human','value': question})
            qwen_message.append({'from': 'gpt','value': None})

            input_ids = preprocess_qwen(
                qwen_message, self.tokenizer, has_image=True, system_message=system_prompt
            ).cuda()
            all_input_ids.append(input_ids.squeeze(0))

        all_video_processed = torch.cat(all_video_processed, dim=0).bfloat16().cuda()
        all_modalities = ["video_long"] * len(all_video_processed)

        _, position_ids, attention_mask, _, inputs_embeds = self.OryxMetaForCausalLM_prepare_inputs_labels_for_multimodal(
            self.model,
            all_input_ids, all_modalities, all_video_processed, all_video_processed
        )

        if dataset == "LongVideoBench":
            num_options = 5
        else:
            num_options = 4
        option_ids = self.tokenizer(
            [chr(ord('A') + i) for i in range(num_options)] + [f" {chr(ord('A') + i)}" for i in range(num_options)],
            return_tensors="pt"
        ).input_ids.flatten().to(input_ids.device)

        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        start_time = time.time()

        all_outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            cache_position=position_ids,
        )

        end_time = time.time()
        torch.cuda.synchronize(0)
        peak_mem_stats = torch.cuda.max_memory_allocated(0)
        extra = {
            "time": end_time - start_time,
            "peak_mem": peak_mem_stats
        }
        last_logits = all_outputs.logits[:, -1, :]
        bsz = last_logits.shape[0]

        option_logits = last_logits[:, option_ids].view(bsz, -1, num_options).sum(1)

        first_option_logits = option_logits[0]
        second_option_logits = option_logits[1]
        top2_options = torch.argsort(first_option_logits, descending=True)[:2]
        if second_option_logits[top2_options[0]] >= second_option_logits[top2_options[1]]:
            generated_ids = option_ids[top2_options[0]].unsqueeze(0)
        else:
            generated_ids = option_ids[top2_options[1]].unsqueeze(0)
        extra.update({
            "first_option_logits": first_option_logits.to(torch.float16).detach().cpu().numpy(),
            "second_option_logits": second_option_logits.to(torch.float16).detach().cpu().numpy(),
        })

        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        return outputs, extra

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_video(message, dataset)

    @torch.no_grad()
    def OryxQwenForCausalLM_generate(
        wrapper, self,
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
            ) = wrapper.OryxMetaForCausalLM_prepare_inputs_labels_for_multimodal(
                self,
                inputs, 
                modalities, images, images_highres
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

    def OryxMetaForCausalLM_prepare_inputs_labels_for_multimodal(
        wrapper,
        self, input_ids,
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
                    cur_new_input_embeds.append(cur_image_features)
                    cur_image_idx += 1
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

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for i, cur_new_embed in enumerate(new_input_embeds):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))
            if cur_len > 0:
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        return None, position_ids, attention_mask, None, new_input_embeds
