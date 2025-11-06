import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp

from typing import List, Optional, Tuple, Union, Dict, Callable
from transformers.generation.utils import GenerateOutput
from transformers.models.qwen2.modeling_qwen2 import (
    CausalLMOutputWithPast,
    Cache, DynamicCache,
    Unpack,
    FlashAttentionKwargs,
    BaseModelOutputWithPast,
    KwargsForCausalLM,
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS
)

from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE, DATASET_MODALITY
import copy
import requests


class ProdecodeLlavaVideoRandframeTokenLogits(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(
            self,
            model_path="/mnt/afs/wangkaibin/models/LLaVA-Video-7B-Qwen2",
            model_config=None,
            **kwargs
    ):
        self.model_config = model_config
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import (
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
                KeywordsStoppingCriteria,
            )  # noqa: E501
        except Exception as err:
            logging.critical(
                "Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`"
            )
            raise err

        video_kwargs_default = dict(
            overwrite=True, mm_spatial_pool_mode="average", force_sample=True
        )
        video_kwargs_default.update(kwargs)
        self.video_kwargs = video_kwargs_default

        overwrite_config = None
        if "video" in model_path.lower():
            if self.video_kwargs["overwrite"]:
                overwrite_config = {}
                overwrite_config["mm_spatial_pool_mode"] = self.video_kwargs[
                    "mm_spatial_pool_mode"
                ]

        rank, world_size = get_rank_and_world_size()
        model_name = get_model_name_from_path(model_path)
        if model_name == "LLaVA-Video-7B-Qwen2":
            model_name = "llava_qwen"
        import warnings
        # filter warning align with official code
        warnings.filterwarnings("ignore")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map=0,
            overwrite_config=overwrite_config,
            # model_config=model_config
        )
        model.eval()
        model.tie_weights()

        if "llava" in model_path.lower():
            conv_mode = "qwen_1_5"
        if 'llava-video' in model_path.lower():
            self.nframe = 64
        else:
            self.nframe = 16
            if "72b" in model_path.lower():
                self.nframe = 32

        if "video" in model_path.lower():
            self.force_sample = self.video_kwargs["force_sample"]
        else:
            self.force_sample = False

        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = (
            process_images  # Store process_images as a class attribute
        )
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        image_sizes = []  # Store image sizes

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)  # Store the size of each image
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        # Process images using the class attribute self.process_images
        image_tensor = self.process_images(
            images, self.image_processor, self.model.config
        )
        image_tensor = [
            _image.to(dtype=torch.float16, device="cuda") for _image in image_tensor
        ]

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def generate_inner_video(self, message, dataset=None):
        content, text_content, visual_content, videos = "", "", "", []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            else:
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        if len(videos) > 1:
            raise ValueError(
                "LLaVA-OneVision does not support multiple videos as input."
            )

        # Prepare common inputs
        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], visual_content + text_content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).cuda().unsqueeze(0)

        # Load two different sets of frames from the same video
        all_video_frames = []
        for _ in range(2):
            video_frames, _, _ = self.load_video_random(
                videos[0], self.nframe, 1, self.force_sample
            )
            all_video_frames.append(video_frames)

        image_sizes = [frame.size for frame in all_video_frames[0]] # image sizes are the same
        modalities = ["video"] * len(image_sizes)

        # Prepare embeddings for both inputs
        embeds_list = []
        alphas = self.model_config.alpha
        
        for i in range(2):
            frames_processed = self.image_processor.preprocess(all_video_frames[i], return_tensors="pt")["pixel_values"].half().cuda()
            
            # Get multimodal embeddings
            _, _, _, _, current_embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, None,
                [frames_processed],
                modalities,
                image_sizes=image_sizes
            )

            # Apply alpha-based sampling
            image_token_pos = torch.where(input_ids[0] == self.IMAGE_TOKEN_INDEX)[0].item()
            prefix_len = image_token_pos
            suffix_len = len(input_ids[0]) - image_token_pos - 1
            seq_len = current_embeds.shape[1]
            image_len = seq_len - prefix_len - suffix_len

            prefix_embeds = current_embeds[:, :prefix_len, :]
            image_embeds = current_embeds[:, prefix_len:prefix_len+image_len, :]
            suffix_embeds = current_embeds[:, prefix_len+image_len:, :]
            
            random_mask = torch.randperm(image_len, device=image_embeds.device) < image_len * alphas[i]
            sampled_embeds = torch.cat([prefix_embeds, image_embeds[:, random_mask, :], suffix_embeds], dim=1)
            embeds_list.append(sampled_embeds)

        # Pad to max length and create attention mask
        max_len = max(e.shape[1] for e in embeds_list)
        batched_embeds = torch.zeros(len(embeds_list), max_len, embeds_list[0].shape[2], device=embeds_list[0].device, dtype=embeds_list[0].dtype)
        attention_mask = torch.zeros(len(embeds_list), max_len, device=embeds_list[0].device, dtype=torch.long)

        for i, e in enumerate(embeds_list):
            current_len = e.shape[1]
            pad_len = max_len - current_len
            batched_embeds[i, pad_len:] = e
            attention_mask[i, pad_len:] = 1

        # Batched forward pass
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        start_time = time.time()

        # Use the model's forward method directly
        outputs = self.model(
            inputs_embeds=batched_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        torch.cuda.synchronize(0)
        peak_mem_stats = torch.cuda.max_memory_allocated(0)
        end_time = time.time()
        extra = {
            "time": end_time - start_time,
            "peak_mem": peak_mem_stats
        }

        # Process logits as before
        if dataset == "LongVideoBench":
            num_options = 5
        else:
            num_options = 4
        option_ids = self.tokenizer(
            [chr(ord('A') + i) for i in range(num_options)] + [f" {chr(ord('A') + i)}" for i in range(num_options)],
            # [chr(ord('A') + i) for i in range(num_options)],
            return_tensors="pt"
        ).input_ids.flatten().to(input_ids.device)

        logits = outputs.logits
        last_token_logits = logits[:, -1, :]
        logits1 = last_token_logits[0]
        logits2 = last_token_logits[1]

        first_option_logits = logits1[option_ids].view(-1, num_options).sum(0)
        top2_options = torch.argsort(first_option_logits, descending=True)[:2]
        second_option_logits = logits2[option_ids].view(-1, num_options).sum(0)
        if second_option_logits[top2_options[0]] >= second_option_logits[top2_options[1]]:
            generated_tokens = option_ids[top2_options[0]].unsqueeze(0)
        else:
            generated_tokens = option_ids[top2_options[1]].unsqueeze(0)
        text_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        extra.update({
            "first_option_logits": first_option_logits.detach().cpu().numpy(),
            "second_option_logits": second_option_logits.detach().cpu().numpy(),
        })

        return text_outputs, extra


    @torch.no_grad()
    def LlavaQwenForCausalLM_generate(
        wrapper,
        self,
        alpha: float = 0.5,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pad_token_id is None:
            pad_token_id = getattr(self.config, 'pad_token_id', 0)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', 2)

        max_new_tokens = 1  # btnkij

        # 获取图像token位置信息
        image_token_pos = torch.where(inputs[0] == wrapper.IMAGE_TOKEN_INDEX)[0].item()
        prefix_len = image_token_pos
        suffix_len = len(inputs[0]) - image_token_pos - 1

        # 准备多模态输入
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes
            )
            # only inputs_embeds is not None
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        seq_len = inputs_embeds.shape[1]
        image_len = seq_len - prefix_len - suffix_len
        total_frame_num = wrapper.model_config.nframe
        token_per_frame = image_len // total_frame_num

        prefix_embeds = inputs_embeds[:, :prefix_len, :]
        image_embeds = inputs_embeds[:, prefix_len:prefix_len+image_len, :]
        suffix_embeds = inputs_embeds[:, prefix_len+image_len:, :]

        random_half = torch.randperm(image_len, device=image_embeds.device) < image_len * alpha
        inputs_embeds = torch.cat([prefix_embeds, image_embeds[:, random_half, :], suffix_embeds], dim=1)
        cache_position = torch.arange(inputs_embeds.shape[1], device=image_embeds.device)
        last_logits = wrapper._prefill_stage(self, inputs_embeds, cache_position)

        return last_logits

    def _prefill_stage(wrapper, self, inputs_embeds, cache_position=None):
        """
        Prefill阶段：处理输入序列，生成初始的KV cache和状态
        """
        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device
        
        # Prefill前向传播 - 处理完整的输入序列
        if cache_position is None:
            cache_position = torch.arange(seq_len, device=device)
        
        outputs = wrapper.Qwen2ForCausalLM_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,  # 初始时没有cache
            inputs_embeds=inputs_embeds,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,  # 只保留最后一个位置的logits
            return_qkv=False
        )
        last_logits = outputs.logits[:, -1, :]
        
        return last_logits

    # def _inference_stage(
    #     wrapper, self, kv, last_logits,
    #     max_new_tokens,
    #     cache_position,
    #     temperature, top_p, top_k, repetition_penalty,
    #     do_sample, pad_token_id, eos_token_id
    # ):
    #     """
    #     Inference阶段：基于prefill的状态进行逐步生成
    #     """        
    #     batch_size = last_logits.shape[0]
    #     device = last_logits.device

    #     generated_tokens = torch.zeros((batch_size, 0), dtype=torch.long, device=device)

    #     past_key_values = DynamicCache()
    #     past_key_values.key_cache, past_key_values.value_cache = kv
    #     past_key_values._seen_tokens = current_seq_len = kv[0][0].shape[2]
        
    #     # 逐步生成循环
    #     for step in range(max_new_tokens):
            
    #         # ========== Token采样 ==========
    #         next_token = _sample_next_token(
    #             last_logits, generated_tokens, temperature,
    #             top_p, top_k, repetition_penalty, do_sample
    #         )
            
    #         # 添加到生成序列
    #         generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            
    #         # 检查是否遇到结束token
    #         if torch.all(next_token.squeeze(-1) == eos_token_id):
    #             break
            
    #         # ========== 准备下一步的输入 ==========
    #         if step < max_new_tokens - 1:  # 不是最后一步
                
    #             # 准备下一步的输入embeddings（只需要最新的token）
    #             current_inputs_embeds = self.get_model().embed_tokens(next_token)
                
    #             # 下一步前向传播
    #             outputs = wrapper.Qwen2ForCausalLM_forward(
    #                 self,
    #                 input_ids=None,
    #                 attention_mask=None,
    #                 position_ids=None,
    #                 past_key_values=past_key_values,
    #                 inputs_embeds=current_inputs_embeds,
    #                 use_cache=True,
    #                 cache_position=cache_position,
    #                 logits_to_keep=1,
    #             )
                
    #             # 更新状态
    #             past_key_values = outputs.past_key_values
    #             last_logits = outputs.logits[:, -1, :]
    #             cache_position = cache_position + 1
        
    #     return generated_tokens

    def Qwen2ForCausalLM_forward(
        wrapper,
        self,   
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_qkv: bool = False,  # btnkij
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = wrapper.Qwen2Model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_qkv=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        causal_lm_output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        if return_qkv:
            causal_lm_output.qkv = outputs.qkv
        return causal_lm_output

    def Qwen2Model_forward(
        wrapper,
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_qkv: bool = False,  # btnkij
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_qkv = () if return_qkv else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = wrapper.Qwen2DecoderLayer_forward(
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
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if return_qkv:
                qkv = layer_outputs[-1]
                all_qkv += (qkv,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        if return_qkv:
            outputs.qkv = all_qkv
        return outputs

    def Qwen2DecoderLayer_forward(
        wrapper,
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        return_qkv: bool = False,  # btnkij
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        outputs = wrapper.Qwen2Attention_forward(
            self.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            return_qkv=return_qkv,  # btnkij
            **kwargs,
        )
        if return_qkv:
            qkv, hidden_states, self_attn_weights = outputs
        else:
            hidden_states, self_attn_weights = outputs
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if return_qkv:
            outputs += (qkv,)

        return outputs

    def Qwen2Attention_forward(
        wrapper,
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_qkv: bool = False,  # btnkij
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                # logger.warning_once(
                #     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                #     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                # )
                pass
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if return_qkv:  # btnkij
            qkv = (query_states, key_states, value_states)
            return qkv, attn_output, attn_weights
        else:
            return attn_output, attn_weights

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        from decord import VideoReader, cpu
        import numpy as np

        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, sample_fps, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb;pdb.set_trace()
        return spare_frames, frame_time, video_time
    
    def load_video_random(self, video_path, max_frames_num, fps=1, force_sample=False):
        from decord import VideoReader, cpu
        import numpy as np

        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        
        num_to_sample = min(max_frames_num, total_frame_num)
        frame_idx = np.random.choice(total_frame_num, num_to_sample, replace=False)
        frame_idx.sort()
        frame_idx = frame_idx.tolist()
            
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        
        return spare_frames, frame_time_str, video_time

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == 'VIDEO':
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)

"""
 {'type': 'text',
  'value': 'Question: What color is the main male character in the video?\n'
           'Options:\n'
           '(A) Yellow\n'
           '(B) Red\n'
           '(C) Green\n'
           '(D) Blue'},
"""
