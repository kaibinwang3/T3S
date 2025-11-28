from vlmeval.config import supported_VLM
from omegaconf import OmegaConf


model_config = OmegaConf.create({
    "alpha": [0.5, 0.3],
    "topk": 10
})


MODEL_CLASS = supported_VLM['t3s_qwen2vl']
model = MODEL_CLASS(
    model_path="/mnt/afs/wangkaibin/models/Qwen2.5-VL-7B-Instruct",
    model_config=model_config
)

message = [
    {
        "type": "video",
        "value": "/mnt/afs/share_data/opencompass/.cache/VideoMME/video/_8lBR0E_Tx8.mp4"
    },
    {
        "type": "text",
        "value": "Describe this video."
    }
]

response = model.generate(message=message)
print(response)
