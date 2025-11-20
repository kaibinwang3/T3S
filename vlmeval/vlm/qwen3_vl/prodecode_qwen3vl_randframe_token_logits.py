from __future__ import annotations

import logging
import os
import warnings
import time
import torch
import decord
from copy import deepcopy
import numpy as np

from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from ...smp import get_gpu_memory, listinstr

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Optional,
    Unpack,
    Cache,
    TransformersKwargs,
    Union,
    Qwen3VLModelOutputWithPast,
    is_torchdynamo_compiling,
)


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


def pack_processed_inputs_for_flash_attn(all_position_ids, all_inputs_embeds, device):
    """
    将已经处理好的inputs pack成一个序列，用于Flash Attention
    
    Args:
        all_position_ids: List[torch.Tensor] - 每个tensor是一个序列的position_ids
        all_inputs_embeds: List[torch.Tensor] - 每个tensor是一个序列的inputs_embeds
        device: 设备

        # (Pdb) pp all_position_ids[0].shape
        # torch.Size([3, 1, 36987])
        # (Pdb) pp all_attention_mask[0].shape
        # torch.Size([1, 36987])
        # (Pdb) pp all_inputs_embeds[0].shape
        # torch.Size([1, 36987, 3584])
    
    Returns:
        dict: 包含packed inputs和Flash Attention所需参数的字典
    """
    # 获取每个序列的长度（从inputs_embeds获取，因为它包含实际的序列长度信息）
    seq_lengths = [embeds.shape[1] for embeds in all_inputs_embeds]  # 假设embeds是[seq_len, hidden_size]
    original_lengths = seq_lengths.copy()
    
    # 计算cumulative lengths (从0开始)
    cu_seq_lens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device=device)
    cu_seq_lens[1:] = torch.cumsum(torch.tensor(seq_lengths, device=device), dim=0)
    
    # Pack inputs_embeds - 直接连接
    packed_inputs_embeds = torch.cat(all_inputs_embeds, dim=1)  # [total_length, hidden_size]
    
    # Pack position_ids - 连接所有position_ids
    packed_position_ids = torch.cat(all_position_ids, dim=2)  # [total_length]
    
    # Pack attention_mask - 连接所有attention_mask
    
    # 最大序列长度
    max_length = max(seq_lengths)
    
    return {
        'inputs_embeds': packed_inputs_embeds,      # [1, total_length, hidden_size]
        'position_ids': packed_position_ids,       # [1, total_length]
        'cu_seq_lens_q': cu_seq_lens,                           # [num_seqs + 1]
        'cu_seq_lens_k': cu_seq_lens,                           # [num_seqs + 1] 
        'max_length_q': max_length,                             # int
        'max_length_k': max_length,                             # int
        'original_lengths': original_lengths                    # List[int]
    }

def unpack_outputs(hidden_states, original_lengths):
    """
    将packed的输出解包回原始的list格式
    
    Args:
        hidden_states: packed的隐藏状态 [1, total_length, hidden_size]
        original_lengths: 原始序列长度列表
    
    Returns:
        List[torch.Tensor]: 解包后的隐藏状态列表，每个是[seq_len, hidden_size]
    """
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


def is_moe_model(model_path: str) -> bool:
    """Check if the model is a Mixture of Experts model."""
    path_parts = model_path.split('/')
    non_moe_patterns = ['2B','4B','8B','32B']
    for part in path_parts:
        if any(pattern in part for pattern in non_moe_patterns):
            return False
    return True


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class ProdecodeQwen3VLRandFrameTokenLogits(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str = '/mnt/afs/share_data/models_weights/external/Qwen/Qwen3/Qwen3-VL-8B-Instruct',
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        model_config = None,
        **kwargs,
    ) -> None:
        os.environ["FORCE_QWENVL_VIDEO_READER"] = "random_sample"
        import qwen_vl_utils.vision_process
        qwen_vl_utils.vision_process.VIDEO_READER_BACKENDS["random_sample"] = sample_random_frames_decord
        
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print('The total number of video tokens might too large, resulting in an overly long input sequence.')
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
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video

        self.model_config = model_config

        assert model_path is not None
        self.model_path = model_path
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Use official Qwen3-Omni classes when model_path indicates omni
        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers")
                raise err
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag `use_vllm` to True"
        if self.use_vllm:
            if listinstr(['omni'], self.model_path.lower()):
                os.environ['VLLM_USE_V1'] = '0'
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'"
                )
            enable_expert_parallel = is_moe_model(self.model_path)
            # For Qwen3-Omni, vLLM engine v1 is not supported yet
            if listinstr(['omni'], self.model_path.lower()):
                limit_mm = {"image": 3, "video": 3, "audio": 3}
            else:
                limit_mm = {"image": self.limit_mm_per_prompt}
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=8,
                limit_mm_per_prompt=limit_mm,
                tensor_parallel_size=tp_size,
                enable_expert_parallel=enable_expert_parallel,
                seed=0,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                trust_remote_code=True,
            )
        else:
            if listinstr(['omni'], model_path.lower()):
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path, dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
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
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
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
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner_transformers(self, message, dataset=None):
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        # for conv in messages:
        #     for span in conv['content']:
        #         if span['type'] == 'video':
        #             span['nframes'] = self.model_config.nframe

        alpha_list = self.model_config.alpha

        if is_omni:
            # For Qwen3-Omni, messages is a list of dicts
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors='pt',
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:

            all_input_ids = []
            all_pixel_values_videos = []
            all_video_grid_thw = []
            all_attention_mask = []

            for _ in range(len(alpha_list)):
                text = self.processor.apply_chat_template(deepcopy(messages), tokenize=False, add_generation_prompt=True)
                images, videos, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )

                video_metadatas = None
                if videos is not None:
                    videos, video_metadatas = zip(*videos)
                    videos, video_metadatas = list(videos), list(video_metadatas)

                inputs = self.processor(
                    text=text,
                    images=images,
                    videos=videos,
                    video_metadata=video_metadatas,
                    do_resize=False,
                    return_tensors='pt',
                    **(video_kwargs or {}),
                )

                try:
                    inputs = inputs.to(self.model.device)
                    if hasattr(self.model, 'dtype'):
                        inputs = inputs.to(self.model.dtype)
                except Exception:
                    inputs = inputs.to('cuda')

                all_input_ids.append(inputs.input_ids)
                all_pixel_values_videos.append(inputs.pixel_values_videos)
                all_video_grid_thw.append(inputs.video_grid_thw)
                all_attention_mask.append(inputs.attention_mask)

        if is_omni:
            try:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video,
                )
            except TypeError:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=self.use_audio_in_video,
                )
            response = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            all_position_ids, all_inputs_embeds = self.prepare_multimodal_inputs(
                input_ids_list=all_input_ids,
                pixel_values_videos_list=all_pixel_values_videos,
                video_grid_thw_list=all_video_grid_thw,
                attention_mask_list=all_attention_mask,
                alpha_list=alpha_list
            )
            # (Pdb) pp processed_inputs['inputs_embeds'].shape
            # torch.Size([1, 93367, 4096])

            device = all_position_ids[0].device
            packed_inputs = pack_processed_inputs_for_flash_attn(all_position_ids, all_inputs_embeds, device)

            torch.cuda.synchronize(0)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            start_time = time.time()

            # outputs = self.model.model.language_model(**packed_inputs)
            outputs = self.model.language_model(
                position_ids=packed_inputs['position_ids'],
                inputs_embeds=packed_inputs['inputs_embeds'],
                cu_seq_lens_q=packed_inputs['cu_seq_lens_q'],
                cu_seq_lens_k=packed_inputs['cu_seq_lens_k'],
                max_length_q=packed_inputs['max_length_q'],
                max_length_k=packed_inputs['max_length_k']
            )
            last_hidden_state = outputs.last_hidden_state
            logits = self.model.lm_head(last_hidden_state)

            end_time = time.time()
            torch.cuda.synchronize(0)
            peak_mem_stats = torch.cuda.max_memory_allocated(0)
            extra = {
                "time": end_time - start_time,
                "peak_mem": peak_mem_stats
            }

            logits = unpack_outputs(logits, packed_inputs['original_lengths'])

            if dataset == "LongVideoBench":
                num_options = 5
            else:
                num_options = 4
            option_ids = self.processor.tokenizer(
                [chr(ord('A') + i) for i in range(num_options)] + [f" {chr(ord('A') + i)}" for i in range(num_options)], 
                return_tensors="pt"
            ).input_ids.flatten().to(logits[0].device)

            first_option_logits = logits[0][-1, option_ids].view(-1, num_options).sum(0)
            top2_options = torch.argsort(first_option_logits, descending=True)[:2]
            second_option_logits = logits[1][-1, option_ids].view(-1, num_options).sum(0)
            if second_option_logits[top2_options[0]] >= second_option_logits[top2_options[1]]:
                generated_ids = option_ids[top2_options[0]].unsqueeze(0)
            else:
                generated_ids = option_ids[top2_options[1]].unsqueeze(0)

            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]

            extra.update({
                "first_option_logits": first_option_logits.to(torch.float16).detach().cpu().numpy(),
                "second_option_logits": second_option_logits.to(torch.float16).detach().cpu().numpy(),
            })

            return response, extra

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
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if is_omni:
            audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if is_omni and 'audios' in locals() and audios is not None:
            mm_data['audio'] = audios

        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if is_omni:
            req['mm_processor_kwargs'] = {"use_audio_in_video": self.use_audio_in_video}
        elif video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs

        outputs = self.llm.generate([req], sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
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
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)

    def prepare_multimodal_inputs_single(
        wrapper,
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
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

        inputs = dict(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return inputs

    def prepare_multimodal_inputs(
        self,
        input_ids_list,
        pixel_values_videos_list,
        video_grid_thw_list,
        attention_mask_list,
        alpha_list
    ):
        bsz = len(alpha_list)
        all_position_ids = []
        all_inputs_embeds = []

        for i in range(bsz):
            inputs = self.prepare_multimodal_inputs_single(
                self.model,
                input_ids=input_ids_list[i],
                pixel_values_videos=pixel_values_videos_list[i],
                video_grid_thw=video_grid_thw_list[i]
            )

            alpha = alpha_list[i]
            video_mask = inputs['visual_pos_masks'].squeeze(0)
            drop_indices = torch.where(video_mask)[0]
            drop_indices = drop_indices[torch.randperm(len(drop_indices), device=drop_indices.device)]
            drop_indices = drop_indices[int(len(drop_indices) * alpha):]
            select_mask = torch.ones_like(video_mask)
            select_mask[drop_indices] = False

            position_ids = inputs['position_ids'][:, :, select_mask]
            inputs_embeds = inputs['inputs_embeds'][:, select_mask, :]

            all_position_ids.append(position_ids)
            all_inputs_embeds.append(inputs_embeds)

        return all_position_ids, all_inputs_embeds
