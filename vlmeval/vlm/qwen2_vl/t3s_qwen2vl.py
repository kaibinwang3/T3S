from __future__ import annotations

import logging
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import qwen_vl_utils.vision_process
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)

from ...dataset import DATASET_MODALITY
from ...smp import get_gpu_memory, get_rank_and_world_size, listinstr
from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin

VLLM_MAX_IMAGE_INPUT_NUM = 24


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
        video_metadata = dict(
            fps=video_fps,
            frames_indices=idx,
            total_num_frames=total_frames,
            video_backend="decord",
        )
        return video, video_metadata, sample_fps
    except decord.DECORDError as e:
        raise RuntimeError(f"使用 decord 读取视频失败: {video_path}") from e


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


class T3S_Qwen2VL(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
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
        """
        Generates a sequence of tokens autoregressively using a custom sampling strategy, KV cache,
        and a repetition penalty to mitigate repetition.
        """
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        # --- 1. Initialization ---
        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        
        # Update the user prompt to encourage step-by-step reasoning
        # Note: This assumes message is a list of dicts with a 'value' key.
        # Adjust if your data structure is different.
        if isinstance(message, list) and message:
            message[-1]['value'] = message[-1]['value'].replace(
                "Answer with the option's letter from the given choices directly.",
                "First, provide a step-by-step reasoning for your choice, and then conclude with the final answer starting with 'So the answer is'."
            )
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        # The batch size is now dynamically controlled by the number of alpha values.
        alpha_list = self.model_config.alpha
        batch_size = len(alpha_list)

        # --- Configuration for generation ---
        self.generate_kwargs["max_new_tokens"] = 1024
        top_k = self.model_config.topk
        # Get repetition penalty from config, default to 1.0 (no penalty)
        repetition_penalty = getattr(self.model_config, 'repetition_penalty', 1.0)
        eos_token_id = self.processor.tokenizer.eos_token_id
        device = 'cuda'

        # --- 2. Prompt Processing and Initial KV Cache Generation ---
        # This part seems inefficient as it processes the same message multiple times.
        # For this change, we'll keep it as-is to focus on the repetition penalty.
        input_ids_list = []
        pixel_values_videos_list = []
        video_grid_thw_list = []
        second_per_grid_ts_list = []

        for _ in range(batch_size):
            text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info([messages])
            initial_inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
            initial_inputs = initial_inputs.to(device)
            input_ids_list.append(initial_inputs.input_ids)
            pixel_values_videos_list.append(initial_inputs.pixel_values_videos)
            video_grid_thw_list.append(initial_inputs.video_grid_thw)
            second_per_grid_ts_list.append(initial_inputs.second_per_grid_ts)

        all_position_ids, all_attention_mask, all_inputs_embeds = \
            self.prepare_multimodal_inputs(
                input_ids_list=input_ids_list,
                pixel_values_videos_list=pixel_values_videos_list,
                video_grid_thw_list=video_grid_thw_list,
                second_per_grid_ts_list=second_per_grid_ts_list,
                alpha_list=alpha_list
            )

        outputs = self.model.model(
            position_ids=all_position_ids,
            attention_mask=all_attention_mask,
            inputs_embeds=all_inputs_embeds,
            use_cache=True
        )
        
        past_key_values = outputs.past_key_values
        logits = self.model.lm_head(outputs.last_hidden_state)

        # --- 3. Assertion and Token Selection for the FIRST token ---
        assert logits.shape[0] == 2, f"Token selection logic requires a batch size of 2, but got {logits.shape[0]} from len(self.model_config.alpha)."

        last_logits1 = logits[0, -1, :]
        last_logits2 = logits[1, -1, :]
        _, topk_indices = torch.topk(last_logits1, top_k)
        second_logits_for_topk = last_logits2[topk_indices]
        best_token_in_topk = torch.argmax(second_logits_for_topk)
        next_token_id = topk_indices[best_token_in_topk].unsqueeze(0)

        generated_token_ids = [next_token_id]

        # --- 4. Autoregressive Generation Loop with KV Cache ---
        for i in range(1, self.generate_kwargs["max_new_tokens"]):
            if next_token_id.item() == eos_token_id:
                break

            next_token_embeds = self.model.model.get_input_embeddings()(next_token_id).unsqueeze(0).expand(2, -1, -1)
            all_attention_mask = torch.cat(
                [all_attention_mask, torch.ones((2, 1), dtype=torch.long, device=device)], dim=1
            )
            next_position_ids = all_position_ids[:, :, -1:] + i

            outputs = self.model.language_model(
                inputs_embeds=next_token_embeds,
                attention_mask=all_attention_mask,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            past_key_values = outputs.past_key_values
            logits = self.model.lm_head(outputs.last_hidden_state)

            # --- APPLY REPETITION PENALTY ---
            if repetition_penalty > 1.0 and len(generated_token_ids) > 0:
                generated_ids_tensor = torch.cat(generated_token_ids, dim=-1).squeeze()
                if generated_ids_tensor.dim() == 0:
                    generated_ids_tensor = generated_ids_tensor.unsqueeze(0)
                
                unique_generated_ids = torch.unique(generated_ids_tensor)

                # Apply penalty to both sets of logits
                for logit_set in [logits[0, -1, :], logits[1, -1, :]]:
                    scores_to_penalize = torch.gather(logit_set, 0, unique_generated_ids)
                    penalized_scores = torch.where(
                        scores_to_penalize > 0,
                        scores_to_penalize / repetition_penalty,
                        scores_to_penalize * repetition_penalty
                    )
                    logit_set.scatter_(0, unique_generated_ids, penalized_scores)
            # --- End of Repetition Penalty ---

            last_logits1 = logits[0, -1, :]
            last_logits2 = logits[1, -1, :]
            _, topk_indices = torch.topk(last_logits1, top_k)
            second_logits_for_topk = last_logits2[topk_indices]
            best_token_in_topk = torch.argmax(second_logits_for_topk)
            next_token_id = topk_indices[best_token_in_topk].unsqueeze(0)

            generated_token_ids.append(next_token_id)

        # --- 5. Finalization ---
        generated_ids = torch.cat(generated_token_ids, dim=-1).unsqueeze(0)
        response = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')

        return response

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
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)
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
                or self.model.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
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
                or self.model.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
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

        outputs = self.model(
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
            rope_deltas=self.model.rope_deltas,
        )
