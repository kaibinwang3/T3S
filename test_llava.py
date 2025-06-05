from vlmeval.config import supported_VLM


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
    # use_sketch=True,
    # sketch_type="frequent_directions",
    # sketch_type="newest",
    # num_sketch_tokens=16*13*(13+1),
    use_sketch=False,
    use_merge=True,
    merge_layer=27,
    chunk_size=16*13*(13+1)
)


model = supported_VLM['llava_video_qwen2_7b'](model_config=model_config)

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
