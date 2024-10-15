import torch
import pathlib
import urllib
import os
import torchvision
from contextlib import contextmanager
from collections import OrderedDict

from lavila.lavila.models import models
from lavila.lavila.models.utils import inflate_positional_embeds
from lavila.lavila.data.video_transforms import Permute
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video


def download_and_load_model_weight(download_path, url):
    model_name = url.split('/')[-1].split('.')[0]
    model_path = os.path.join(download_path, f'{model_name}.pth')
    if not os.path.exists(model_path):
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Failed to download the file. Error: {e}")
    else:
        print("Model already exists. Skipping download.")

    ckpt_path = pathlib.Path(model_path)
    with set_posix_windows():
        ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
    return ckpt, state_dict


def initialize_model_from_ckpt(ckpt, state_dict, n_frames):
    old_args = ckpt['args']
    print('=> creating model: {}'.format(old_args.model))
    model = getattr(models, old_args.model)(
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
        timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
        timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
        freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
        freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
        num_frames=n_frames,
        drop_path_rate=0,
    )

    if 'TIMESFORMER' in old_args.model:
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=n_frames,
            load_temporal_fix='bilinear',
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def transform_input(ckpt, input):
    old_args = ckpt['args']
    crop_size = 224 if '336PX' not in old_args.model else 336
    val_transform = transforms.Compose([
        Permute([1, 0, 2, 3]), 
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if ('OPENAI' not in old_args.model) else
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
    ])
    transformed_frames = val_transform(input)
    output = transformed_frames.unsqueeze(0)
    return output


def get_start_end_description(video_dict: dict) -> list[tuple]:
    segments = video_dict['segments']
    labeled_intervals = [(step['start_time'], step['end_time'], step['step_description']) for step in segments]
    return labeled_intervals


def read_frames_between_intervals(video_dict, video_path):
    video_object = torchvision.io.VideoReader(video_path)
    video_object.set_current_stream("video")
    segments = video_dict['segments']
    labeled_intervals = [(step['start_time'], step['end_time'], step['step_description']) for step in segments]
    interval_frames = []
    frame_iterator = iter(video_object)  
    current_frame = next(frame_iterator, None)
    for start_time, end_time, _ in labeled_intervals:
        frames_in_interval = []
        
        while current_frame is not None:
            timestamp = current_frame['pts']
            if start_time <= timestamp <= end_time:
                frames_in_interval.append(current_frame['data'].to(torch.float32))
            if timestamp > end_time:
                break
            current_frame = next(frame_iterator, None)

        interval_frames.append(torch.stack(frames_in_interval))

    return labeled_intervals, interval_frames

@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup
