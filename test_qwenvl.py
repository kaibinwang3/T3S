import torch
import numpy as np
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.models.qwen2_5_vl.modeling_my_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration
)
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
#     Qwen2_5_VLForConditionalGeneration
# )


class DictObject(dict):
    """Dictionary that can be accessed with dot notation and serialized."""
    
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            # Handle initialization with a dictionary
            super().__init__(**args[0])
        else:
            super().__init__(*args, **kwargs)
        self.__dict__ = self
        
        # Convert nested dictionaries recursively
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DictObject(value)
            elif isinstance(value, list):
                # Handle lists of dictionaries
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        value[i] = DictObject(item)
    
    def to_dict(self):
        """Convert back to regular dict for serialization."""
        result = {}
        for key, value in self.items():
            if isinstance(value, DictObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                # Handle lists of DictObjects
                result[key] = [item.to_dict() if isinstance(item, DictObject) else item 
                              for item in value]
            else:
                result[key] = value
        return result


model_config = DictObject(
    use_fused_attention=True,
    full_attention_layers=2,
    alpha=0.9,
    # cpu_attention_type="random_projection_attention",
    cpu_attention_type="kernelized_attention",
    # cpu_attention_type="svd_lowrank_attention",
    # cpu_attention_type="lsh_attention",
    proj_dim=16,
    n_hashes=150,
    head_dim=128,
    top_p=0.1,
    n_bits=10
)

# default: Load the model on the available device(s)
pretrained_path = "/mnt/afs/wangkaibin/models/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    model_config=model_config
)
processor = AutoProcessor.from_pretrained(pretrained_path)

# Messages containing a local video path and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///mnt/afs/share_data/opencompass/.cache/VideoMME/video/5KlS-p5eYH8.mp4",
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
video_inputs[0] = video_inputs[0][:300]  # btnkij
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)


# # Messages containing a local video path and a text query
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "video": "file:///mnt/afs/share_data/opencompass/.cache/MVBench/video/star/Charades_v1_480/W7CR5.mp4",
#                 "fps": 1.0,
#             },
#             {"type": "text", "text": "Describe this video."},
#         ],
#     }
# ]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
#     **video_kwargs,
# )
# inputs = inputs.to("cuda")

# # Inference
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)

