import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE, DATASET_MODALITY
import copy
import requests


class BaselineLlavaVideo(BaseModel):
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

        breakpoint()
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

        video_frames, frame_time, video_time = self.load_video(
            videos[0], self.nframe, 1, self.force_sample
        )

        time_instruciton = (
            f"The video lasts for {video_time:.2f} seconds,"
            f"and {len(video_frames[0])} frames are uniformly sampled from it."
            f"These frames are located at {frame_time}."
            f"Please answer the following questions related to this video.\n"
        )

        if self.force_sample:
            content = visual_content + time_instruciton + text_content
        else:
            content = visual_content + text_content

        image_tensors = []
        frames = (
            self.image_processor.preprocess(video_frames, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )
        image_tensors.append(frames)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()
        image_sizes = [frame.size for frame in video_frames]
        modalities = ["video"] * len(video_frames)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        if dataset == "LongVideoBench":
            num_options = 5
        else:
            num_options = 4
        option_ids = self.tokenizer(
            [chr(ord('A') + i) for i in range(num_options)] + [f" {chr(ord('A') + i)}" for i in range(num_options)], 
            return_tensors="pt"
        ).input_ids.flatten().to(input_ids.device)

        # Pass image sizes along with other parameters
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        start_time = time.time()
        outputs = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=1,
            modalities=modalities,
            stopping_criteria=[stopping_criteria],
            output_scores=True,
            return_dict_in_generate=True
        )
        end_time = time.time()
        torch.cuda.synchronize(0)
        peak_mem_stats = torch.cuda.max_memory_allocated(0)
        extra = {
            "time": end_time - start_time,
            "peak_mem": peak_mem_stats
        }

        scores = outputs.scores[0]
        option_scores = scores[:, option_ids].view(-1, num_options).sum(0)
        generated_ids = option_ids[torch.argmax(option_scores)].unsqueeze(0)

        text_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text_outputs, extra

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
