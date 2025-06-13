from vlmeval.config import supported_VLM
from omegaconf import OmegaConf


model_config = OmegaConf.create({
    "nframe": 64,
    "num_forward": 4
})


model = supported_VLM['refine_llava'](model_config=model_config)

message = [
    {
        'type': 'video',
        'value': '/mnt/afs/share_data/opencompass/.cache/VideoMME/video/0ay2Qy3wBe8.mp4'
    },
    {
        'type': 'text',
        'value': 'describe this video.'
        # 'value': '2+3=?'
    }
]

response = model.generate(message=message, dataset="Video-MME")
print(response)
