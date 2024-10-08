import itertools
import pathlib
import torch
import torchvision
import sys
from collections import OrderedDict
sys.path.insert(0, './lavila')
from lavila.data.video_transforms import Permute
from contextlib import contextmanager
import matplotlib.pyplot as plt
from collections import OrderedDict
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

def process_segments(segments: list[dict]) -> dict:
    labeled_time = {}
    for step in segments:
        start = step['start_time']
        end = step['end_time']
        step_description = step['step_description']
        key = (start, end)
        if key in labeled_time:
            print(f"Key {key} already exists, skipping update.")
        else:
            labeled_time[key] = step_description
    return labeled_time

def calculate_gap(time_intervals: list[tuple]) -> list[int]:
    gaps = []
    for i in range(len(time_intervals) - 1):
        current_end = time_intervals[i][1]
        next_start = time_intervals[i + 1][0]
        if next_start > current_end:
            gap = next_start - current_end
            gaps.append(gap)
    return gaps

def get_labeled_time_interval(video: dict, substep: bool = False) -> dict:
    segments = video['segments']
    if substep:
        subsegments = [segment['segments'] for segment in segments]
        subsegments = list(itertools.chain(*subsegments))
        labeled_time = process_segments(subsegments)

    else:
        labeled_time = process_segments(segments)
    return labeled_time


def get_total_gap(video_dict: dict, substep: bool = False) -> list[tuple]:

    labeled_time = get_labeled_time_interval(video_dict, substep)

    time_intervals = list(labeled_time.keys())
    gaps = calculate_gap(time_intervals)

    video_start = video_dict['start_time']
    video_end = video_dict['end_time']

    if len(time_intervals) > 0:
        initial_start, _ = time_intervals[0]
        _, last_end = time_intervals[-1]

        if video_start < initial_start:
            gaps = [initial_start - video_start] + gaps
        if video_end > last_end:
            gaps.append(video_end - last_end)
    

    
    return gaps

def read_frames_between_intervals(video_object, intervals: list[tuple[int, int]]):
    interval_frames = {}
    frame_position = 0

    for start_time, end_time in intervals:
        frames_in_interval = []
        for frame in video_object:
            timestamp = frame['pts'] 
            frame_position += 1       
            if start_time <= timestamp <= end_time:
                frames_in_interval.append((frame_position, timestamp, frame['data']))  
            if timestamp > end_time:
                break

        interval_frames[(start_time, end_time)] = frames_in_interval

    return interval_frames

def visualize_frames(frames: list[tuple], n = 5):
    for frame_idx, pts, frame in frames[:n]:
        frame_np = frame.permute(1, 2, 0).numpy()
        plt.imshow(frame_np)
        plt.title(f'Frame {frame_idx} PTS: {pts}')
        plt.axis('off')  
        plt.show()

@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup

def visualize_transformed_frames(transformed_frames, num_samples = 3):
    random_indices = torch.randint(0, transformed_frames.size(1), (num_samples,))
    for i in random_indices: 
        frame = transformed_frames[:, i, :, :]    # Extract frame of shape [C, H, W]
        frame_np = frame.permute(1, 2, 0).numpy()  # Convert to (H, W, C)

        plt.imshow(frame_np)
        plt.title(f'Frame {i}')
        plt.axis('off')  
        plt.show()

def decode_one(generated_ids, tokenizer):
    # get the index of <EOS>
    if tokenizer.eos_token_id == tokenizer.bos_token_id:
        if tokenizer.eos_token_id in generated_ids[1:].tolist():
            eos_id = generated_ids[1:].tolist().index(tokenizer.eos_token_id) + 1
        else:
            eos_id = len(generated_ids.tolist()) - 1
    elif tokenizer.eos_token_id in generated_ids.tolist():
        eos_id = generated_ids.tolist().index(tokenizer.eos_token_id)
    else:
        eos_id = len(generated_ids.tolist()) - 1
    generated_text_str = tokenizer.tokenizer.decode(generated_ids[1:eos_id].tolist())
    return generated_text_str

def load_model(model, model_weight_path):
    ckpt_path = pathlib.Path(model_weight_path)
    with set_posix_windows():
        ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
        
    constructed_model = model(
        text_use_cls_token=False,
        project_embed_dim=256,
        gated_xattn=True,
        timesformer_gated_xattn=False,
        freeze_lm_vclm=False,
        freeze_visual_vclm=False,
        freeze_visual_vclm_temporal=False,
        num_frames=4,
        drop_path_rate=0.
    )
    constructed_model.load_state_dict(state_dict, strict=True)
    constructed_model.eval()
    return constructed_model

def transform_frames(crop_size, original_frames):
    val_transform = transforms.Compose([
        Permute([1, 0, 2, 3]), # convert to [C, T, H, W]
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])

    ])
    transformed_frames = val_transform(original_frames)
    return transformed_frames

def generate_caption(model, tokenizer, num_input_frames, input_frames, k_candidate):
    descriptions = []
    input = input_frames.unsqueeze(0) 
    for i in range(0, input.size(2),num_input_frames):
        frame_chunk = input[:, :, i: i + num_input_frames, :, :]
        frame_des = []
        with torch.no_grad():
            image_features = model.encode_image(frame_chunk)  
            generated_text_ids,_ = model.generate(
                image_features,
                tokenizer,
                target=None,
                max_text_length=77,
                top_k=None,
                top_p=0.95,
                num_return_sequences=k_candidate,  
                temperature=0.5,
                early_stopping=True,
            )
        
        for i in range(k_candidate):
            generated_text_str = decode_one(generated_text_ids[i], tokenizer)
            frame_des.append(generated_text_str)

        descriptions.append(frame_des)

    for i, description in enumerate(descriptions):
        print(f"Frame chunk {i}: {description}")
    
    return descriptions


# def example_read_video(video_object, start=0, end=None):
#     if end is None:
#         end = float("inf")
#     if end < start:
#         raise ValueError(
#             "end time should be larger than start time, got "
#             "start time={} and end time={}".format(start, end)
#         )

#     video_frames = torch.empty(0)
#     video_pts = []
#     video_object.set_current_stream("video")
#     frames = []
#     for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start)):
#         frames.append(frame['data'])
#         video_pts.append(frame['pts'])
#     if len(frames) > 0:
#         video_frames = torch.stack(frames, 0)

#     return video_frames, video_pts