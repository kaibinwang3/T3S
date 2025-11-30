from vlmeval.config import supported_VLM
from omegaconf import OmegaConf


model_config = OmegaConf.create({
    "alpha": [0.5, 0.3],
    "topk": 10
})


MODEL_CLASS = supported_VLM['t3s_qwen2vl']
model = MODEL_CLASS(
    model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    model_config=model_config
)

message = [
    {
        "type": "video",
        "value": "/your/video/path"
    },
    {
        "type": "text",
        "value": "Describe this video."
    }
]

response = model.generate(message=message)
print(response)
