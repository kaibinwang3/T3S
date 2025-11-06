from __future__ import annotations

import os

import qwen_vl_utils.vision_process

import decord
import torch
import numpy as np
import os
import time
from typing import Tuple, Optional, Union, List
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    BaseModelOutputWithPast,
    DynamicCache,
    Cache,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
    _flash_attention_forward,
)
from flash_attn_kv_score import flash_attn_kv_score


def sample_random_frames_decord(ele: dict) -> Tuple[torch.Tensor, float]:
    """
    使用 decord 从视频中随机采样指定数量的帧。

    Args:
        ele (dict): 一个包含视频配置的字典。
        支持的键:
            - "video" (str): 视频的路径。
            - "nframes" (int): 需要随机采样的帧数。

    Returns:
        Tuple[torch.Tensor, float]:
            - torch.Tensor: 形状为 (T, C, H, W) 的视频张量，其中 T 是 nframes。
            - float: 基于采样密度的名义上的采样率。
    
    Raises:
        FileNotFoundError: 如果视频文件不存在。
        ValueError: 如果请求的 nframes 无效 (非正数或大于视频总帧数)。
        KeyError: 如果字典 ele 中缺少 "video" 或 "nframes" 键。
    """
    video_path = ele["video"].replace("file://", "")
    nframes = ele["nframes"]
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件未找到: {video_path}")
    if not isinstance(nframes, int) or nframes <= 0:
        raise ValueError(f"要采样的帧数 'nframes' 必须是一个正整数，但得到的是: {nframes}")
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        if nframes > total_frames:
            print(f"警告：请求采样 {nframes} 帧, 但视频总共只有 {total_frames} 帧。")
            nframes = total_frames
        random_indices = np.sort(np.random.choice(total_frames, nframes, replace=False))
        idx = random_indices.tolist()
        video = vr.get_batch(idx).asnumpy()
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        sample_fps = nframes / max(total_frames, 1) * video_fps
        return video, sample_fps
    except decord.DECORDError as e:
        raise RuntimeError(f"使用 decord 读取视频失败: {video_path}") from e


import sys
import warnings
import math
import logging
import time
from functools import partial

import torch

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, listinstr
from ...dataset import DATASET_MODALITY

VLLM_MAX_IMAGE_INPUT_NUM = 24


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images


def setup_visible_devices_per_rank():
    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    assert world_size == 1, "Only support world_size == 1 for vLLM inference"
    num_gpus = total_gpus // world_size
    start_idx = rank * num_gpus
    assigned_devices = list(range(start_idx, start_idx + num_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in assigned_devices)
    logging.info(f"[Rank {rank}] Visible GPUs: {assigned_devices}")
    return num_gpus


def get_topk_mask(attn_score, k, video_mask):
    """
    Generates a binary mask indicating the top-k values in an attention score tensor.

    Args:
    attn_score (torch.Tensor): A 1D tensor of attention scores.
    k (int): The number of top scores to identify.
    video_mask (torch.Tensor, optional): A 1D tensor of the same size as
                                            attn_score, with 1s for elements to
                                            consider and 0s for elements to ignore.
                                            If None, all scores are considered.

    Returns:
    torch.Tensor: A 1D binary tensor (0s and 1s) of the same size as attn_score,
                    with 1s at the positions of the top-k scores.
    """
    if not isinstance(video_mask, bool):
        video_mask = video_mask.bool()
    masked_scores = torch.full_like(attn_score, -torch.inf)
    masked_scores[video_mask] = attn_score[video_mask]
    _, top_indices = torch.topk(masked_scores, k)
    topk_mask = torch.zeros_like(attn_score, dtype=torch.bool)
    topk_mask[top_indices] = True
    return topk_mask


class ProdecodeQwen2VLRandframeAttnLogits(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str = "/mnt/afs/wangkaibin/models/Qwen2.5-VL-7B-Instruct",
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        use_audio_in_video: bool = False,
        model_config = None,
        **kwargs,
    ):
        os.environ["FORCE_QWENVL_VIDEO_READER"] = "random_sample"
        qwen_vl_utils.vision_process.VIDEO_READER_BACKENDS["random_sample"] = sample_random_frames_decord
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.model_config = model_config
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        if self.total_pixels and self.total_pixels > 24576 * 28 * 28:
            print('The total number of video tokens might become too large, resulting in an overly long input sequence. We recommend lowering **total_pixels** to below **24576 × 28 × 28**.')  # noqa: E501
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        if self.fps is None and self.nframe is None:
            print("Warning: fps and nframe are both None, \
                  using default nframe/fps setting in qwen-vl-utils/qwen-omni-utils, \
                  the fps/nframe setting in video dataset is omitted")
        self.use_audio_in_video = use_audio_in_video
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356")  # noqa: E501
                raise err
            MODEL_CLS = Qwen2_5OmniForConditionalGeneration
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        elif listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
            from transformers import AutoProcessor
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0
        self.use_vllm = kwargs.get('use_vllm', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM

        self.model = MODEL_CLS.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map=0,
            attn_implementation='flash_attention_2',
        )
        self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value'])
                }
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'audio':
                item = {'type':'audio','audio':s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner_transformers(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        self.generate_kwargs["max_new_tokens"] = 1
        self.generate_kwargs["do_sample"] = True
        self.generate_kwargs["top_k"] = None
        self.generate_kwargs["top_p"] = None

        all_alpha = self.model_config.alpha
        all_position_ids = []
        all_attention_mask = []
        all_inputs_embeds = []
        bsz = len(all_alpha)

        for i in range(bsz):
            # Process inputs and move to GPU
            text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info([messages])
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
            inputs = inputs.to(0)

            _, position_ids, attention_mask, _, inputs_embeds, video_mask = \
                self.prepare_multimodal_inputs_single(
                    self.model,
                    input_ids=inputs.input_ids,
                    pixel_values_videos=inputs.pixel_values_videos,
                    video_grid_thw=inputs.video_grid_thw,
                    second_per_grid_ts=inputs.second_per_grid_ts
                )
            
            # [OPTIMIZATION] Delete the initial large inputs tensor now that it's processed
            del inputs
            torch.cuda.empty_cache()

            outputs = self.Qwen2_5_VLModel_forward(
                self.model.model,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                return_qkv=True,
            )

            select_attn_layer = self.model_config.select_attn_layer
            attn_score = torch.zeros(outputs.qkv[0][0].shape[1], dtype=outputs.qkv[0][0].dtype, device=outputs.qkv[0][0].device)
            
            # [OPTIMIZATION] Isolate the large QKV tensor list and delete the rest of `outputs`
            qkv_all_layers = outputs.qkv
            del outputs
            torch.cuda.empty_cache()

            for layer_idx in range(len(qkv_all_layers)):
                if layer_idx != select_attn_layer and select_attn_layer != -1:
                    continue
                
                query_states, key_states, _ = qkv_all_layers[layer_idx]
                query_states = query_states[:, -1:, :, :]
                
                attn_score += flash_attn_kv_score(query_states, key_states).mean((0, 1))
                
                # [OPTIMIZATION] Delete layer-specific tensors immediately after use
                del query_states, key_states
                torch.cuda.empty_cache()

            # [OPTIMIZATION] Delete the list of QKVs
            del qkv_all_layers
            torch.cuda.empty_cache()
            
            # --- Original logic for selection ---
            k = int(video_mask.sum() * all_alpha[i])
            select_mask = get_topk_mask(attn_score, k, video_mask)
            select_mask |= ~video_mask

            position_ids = position_ids[:, :, select_mask]
            attention_mask = attention_mask[:, select_mask]
            inputs_embeds = inputs_embeds[:, select_mask, :]
            
            # [OPTIMIZATION] Move results to CPU to free up GPU VRAM for the next loop iteration
            all_position_ids.append(position_ids)
            all_attention_mask.append(attention_mask)
            all_inputs_embeds.append(inputs_embeds)
            
            # [OPTIMIZATION] Clean up other loop-specific GPU tensors
            del position_ids, attention_mask, inputs_embeds, video_mask, attn_score, select_mask
            torch.cuda.empty_cache()

        # [OPTIMIZATION] Pad and concatenate on the CPU
        max_len = max(it.shape[1] for it in all_inputs_embeds)
        # The device for padding tensors will be CPU by default
        for i in range(bsz):
            pad_len = max_len - all_inputs_embeds[i].shape[1]
            if pad_len == 0:
                continue
            
            # Create padding on CPU
            all_position_ids[i] = torch.cat([
                torch.zeros([3, 1, pad_len], dtype=torch.long, device=0), all_position_ids[i]
            ], dim=2)
            all_attention_mask[i] = torch.cat([
                torch.zeros([1, pad_len], dtype=torch.long, device=0), all_attention_mask[i]
            ], dim=1)
            all_inputs_embeds[i] = torch.cat([
                torch.zeros([1, pad_len, all_inputs_embeds[i].shape[2]], dtype=all_inputs_embeds[i].dtype, device=0), all_inputs_embeds[i]
            ], dim=1)

        # Concatenate all tensors on the CPU
        all_position_ids = torch.cat(all_position_ids, dim=1)
        all_attention_mask = torch.cat(all_attention_mask, dim=0)
        all_inputs_embeds = torch.cat(all_inputs_embeds, dim=0)

        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        start_time = time.time()

        outputs = self.model.model(
            position_ids=all_position_ids,
            attention_mask=all_attention_mask,
            inputs_embeds=all_inputs_embeds
        )
        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)

        end_time = time.time()
        torch.cuda.synchronize(0)
        peak_mem_stats = torch.cuda.max_memory_allocated(0)
        extra = {
            "time": end_time - start_time,
            "peak_mem": peak_mem_stats
        }
        
        if dataset == "LongVideoBench":
            num_options = 5
        else:
            num_options = 4
        option_ids = self.processor.tokenizer(
            [chr(ord('A') + i) for i in range(num_options)] + [f" {chr(ord('A') + i)}" for i in range(num_options)], 
            return_tensors="pt"
        ).input_ids.flatten().to(0)

        last_logits = logits[:, -1, :]
        first_option_logits = last_logits[0, option_ids].view(-1, num_options).sum(0)
        top2_options = torch.argsort(first_option_logits, descending=True)[:2]
        second_option_logits = last_logits[1, option_ids].view(-1, num_options).sum(0)
        if second_option_logits[top2_options[0]] >= second_option_logits[top2_options[1]]:
            generated_tokens = option_ids[top2_options[0]].unsqueeze(0)
        else:
            generated_tokens = option_ids[top2_options[1]].unsqueeze(0)

        out = self.processor.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]

        extra.update({
            "first_option_logits": first_option_logits.to(torch.float16).detach().cpu().numpy(),
            "second_option_logits": second_option_logits.to(torch.float16).detach().cpu().numpy(),
        })

        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response, extra

    def generate_inner(self, message, dataset=None):
        return self.generate_inner_transformers(message, dataset=dataset)

    def prepare_multimodal_inputs_single(
        wrapper,
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        video_mask = video_mask[0, :, 0]
        attention_mask = torch.ones_like(input_ids)

        return None, position_ids, attention_mask, None, inputs_embeds, video_mask

    def prepare_multimodal_inputs(
        self,
        input_ids_list: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos_list: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw_list: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts_list: Optional[torch.Tensor] = None,
        alpha_list: Optional[torch.Tensor] = None
    ):
        bsz = len(alpha_list)
        all_position_ids = []
        all_attention_mask = []
        all_inputs_embeds = []

        for i in range(bsz):
            _, position_ids, attention_mask, _, inputs_embeds, video_mask = \
                self.prepare_multimodal_inputs_single(
                    self.model,
                    input_ids=input_ids_list[i],
                    pixel_values_videos=pixel_values_videos_list[i],
                    video_grid_thw=video_grid_thw_list[i],
                    second_per_grid_ts=second_per_grid_ts_list[i]
                )

            alpha = alpha_list[i]
            drop_indices = torch.where(video_mask)[0]
            drop_indices = drop_indices[torch.randperm(len(drop_indices), device=drop_indices.device)]
            drop_indices = drop_indices[int(len(drop_indices) * alpha):]
            select_mask = torch.ones_like(video_mask)
            select_mask[drop_indices] = False

            position_ids = position_ids[:, :, select_mask]
            attention_mask = attention_mask[:, select_mask]
            inputs_embeds = inputs_embeds[:, select_mask, :]

            all_position_ids.append(position_ids)
            all_attention_mask.append(attention_mask)
            all_inputs_embeds.append(inputs_embeds)

        max_len = max(it.shape[1] for it in all_inputs_embeds)
        device = all_inputs_embeds[0].device
        for i in range(bsz):
            pad_len = max_len - all_inputs_embeds[i].shape[1]
            if pad_len == 0:
                continue
            all_position_ids[i] = torch.cat([
                torch.zeros([3, 1, pad_len], dtype=torch.long, device=device), all_position_ids[i]
            ], dim=2)
            all_attention_mask[i] = torch.cat([
                torch.zeros([1, pad_len], dtype=torch.long, device=device), all_attention_mask[i]
            ], dim=1)
            all_inputs_embeds[i] = torch.cat([
                torch.zeros([1, pad_len, all_inputs_embeds[i].shape[2]], dtype=all_inputs_embeds[i].dtype, device=device), all_inputs_embeds[i]
            ], dim=1)

        all_position_ids = torch.cat(all_position_ids, dim=1)
        all_attention_mask = torch.cat(all_attention_mask, dim=0)
        all_inputs_embeds = torch.cat(all_inputs_embeds, dim=0)
        return all_position_ids, all_attention_mask, all_inputs_embeds

    def Qwen2_5_VLForConditionalGeneration_forward(
        wrapper,
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        video_mask = video_mask[0, :, 0]
        drop_indices = torch.where(video_mask)[0]
        drop_indices = drop_indices[torch.randperm(len(drop_indices), device=drop_indices.device)]
        drop_indices = drop_indices[int(len(drop_indices) * alpha):]
        select_mask = torch.ones_like(video_mask)
        select_mask[drop_indices] = False

        position_ids = position_ids[:, :, select_mask]
        attention_mask = attention_mask[:, select_mask]
        inputs_embeds = inputs_embeds[:, select_mask, :]

        outputs = wrapper.Qwen2_5_VLModel_forward(
            self.model,
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,  # None
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,  # None
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def Qwen2_5_VLModel_forward(
        wrapper,
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_qkv: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_qkv = () if return_qkv else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = wrapper.Qwen2_5_VLDecoderLayer_forward(
                decoder_layer,
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                return_qkv=return_qkv,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if return_qkv:
                all_qkv += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        if return_qkv:
            outputs.qkv = all_qkv
        return outputs

    def Qwen2_5_VLDecoderLayer_forward(
        wrapper,
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        return_qkv: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, qkv = wrapper.Qwen2_5_VLFlashAttention2_forward(
            self.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            return_qkv=return_qkv,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if return_qkv:
            outputs += (qkv,)

        return outputs

    def Qwen2_5_VLFlashAttention2_forward(
        wrapper,
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        return_qkv: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if return_qkv:
            qkv = (query_states, key_states, value_states)
        else:
            qkv = None

        return attn_output, attn_weights, past_key_value, qkv
