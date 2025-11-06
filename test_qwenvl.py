from vlmeval.config import supported_VLM
from omegaconf import OmegaConf


model_config = OmegaConf.create({
    "nframe": 64,
    "num_forward": 4
})


model = supported_VLM['refine_qwen2vl'](model_config=model_config)

messages1 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"},
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
messages = [messages1, messages2]

response = model.generate(message=messages, dataset="Video-MME")
print(response)
