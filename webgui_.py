#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
webui
'''
import argparse
import os
import random
import uuid
from pathlib import Path

import cv2
import dotenv
import gradio as gr
import numpy as np
import requests
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from facenet_pytorch import MTCNN
from moviepy.editor import VideoFileClip, AudioFileClip
from omegaconf import OmegaConf
from openai import OpenAI

from src.models.face_locator import FaceLocator
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad

import asyncio
import sys
import base64

dotenv.load_dotenv()

default_values = {
    "width": 512,
    "height": 512,
    "length": 1200,
    "seed": 420,
    "facemask_dilation_ratio": 0.1,
    "facecrop_dilation_ratio": 0.5,
    "context_frames": 12,
    "context_overlap": 3,
    "cfg": 2.5,
    "steps": 30,
    "sample_rate": 16000,
    "fps": 24,
    "device": "cuda"
}

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print(
        "please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

config_path = "./configs/prompts/animation.yaml"
config = OmegaConf.load(config_path)
if config.weight_dtype == "fp16":
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

inference_config_path = config.inference_config
infer_config = OmegaConf.load(inference_config_path)

############# model_init started #############
## vae init
vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to("cuda", dtype=weight_dtype)

## reference net init
reference_unet = UNet2DConditionModel.from_pretrained(
    config.pretrained_base_model_path,
    subfolder="unet",
).to(dtype=weight_dtype, device=device)
reference_unet.load_state_dict(torch.load(config.reference_unet_path, map_location="cpu"))

## denoising net init
if os.path.exists(config.motion_module_path):
    ### stage1 + stage2
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)
else:
    ### only stage1
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
            "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
        }
    ).to(dtype=weight_dtype, device=device)

denoising_unet.load_state_dict(torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)

## face locator init
face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(dtype=weight_dtype,
                                                                                                  device="cuda")
face_locator.load_state_dict(torch.load(config.face_locator_path))

## load audio processor params
audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

## load face detector params
face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
                      post_process=True, device=device)

############# model_init finished #############

sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
scheduler = DDIMScheduler(**sched_kwargs)

pipe = Audio2VideoPipeline(
    vae=vae,
    reference_unet=reference_unet,
    denoising_unet=denoising_unet,
    audio_guider=audio_processor,
    face_locator=face_locator,
    scheduler=scheduler,
).to("cuda", dtype=weight_dtype)


def get_video_bytes(video_path):
    """将视频文件转换为Base64字符串，用于跨机器传输"""
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            return f"错误：视频文件不存在 {video_path}"

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        return base64.b64encode(video_bytes).decode("utf-8")
    except Exception as e:
        return f"错误：{str(e)}"

def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None
    sorted_bboxes = sorted(filtered_bboxes, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]


def process_video(uploaded_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio,
                  facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, device):
    if seed is not None and seed > -1:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(random.randint(100, 1000000))

    #### face musk prepare
    face_img = cv2.imread(uploaded_img)
    face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
    det_bboxes, probs = face_detector.detect(face_img)
    select_bbox = select_face(det_bboxes, probs)
    if select_bbox is None:
        face_mask[:, :] = 255
    else:
        xyxy = select_bbox[:4]
        xyxy = np.round(xyxy).astype('int')
        rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
        r_pad = int((re - rb) * facemask_dilation_ratio)
        c_pad = int((ce - cb) * facemask_dilation_ratio)
        face_mask[rb - r_pad: re + r_pad, cb - c_pad: ce + c_pad] = 255

        #### face crop
        r_pad_crop = int((re - rb) * facecrop_dilation_ratio)
        c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)
        crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]),
                     min(re + r_pad_crop, face_img.shape[0])]
        face_img, _ = crop_and_pad(face_img, crop_rect)
        face_mask, _ = crop_and_pad(face_mask, crop_rect)
        face_img = cv2.resize(face_img, (width, height))
        face_mask = cv2.resize(face_mask, (width, height))

    ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
    face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device="cuda").unsqueeze(0).unsqueeze(
        0).unsqueeze(0) / 255.0

    video = pipe(
        ref_image_pil,
        uploaded_audio,
        face_mask_tensor,
        width,
        height,
        length,
        steps,
        cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=context_frames,
        fps=fps,
        context_overlap=context_overlap
    ).videos

    save_dir = Path("output/tmp")
    save_dir.mkdir(exist_ok=True, parents=True)
    output_video_path = save_dir / "output_video.mp4"
    save_videos_grid(video, str(output_video_path), n_rows=1, fps=fps)

    video_clip = VideoFileClip(str(output_video_path))
    audio_clip = AudioFileClip(uploaded_audio)
    final_output_path = save_dir / "output_video_with_audio.mp4"
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(str(final_output_path), codec="libx264", audio_codec="aac")

    return str(final_output_path), "视频生成成功"


def fetch_data(query: str):
    """调用开放API获取对应Agent生成的文案内容"""
    # 1.从环境变量中获取数据
    url = os.getenv("LLMOPS_API_BASE")
    api_key = os.getenv("LLMOPS_API_KEY")
    app_id = os.getenv("LLMOPS_APP_ID")

    # 2.组装请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 3.组装请求数据
    request_body = {
        "app_id": app_id,
        "end_user_id": "",
        "conversation_id": "",
        "stream": False,
        "query": query,
        "image_urls": [],
    }

    try:
        # 4.使用requests包发起post请求
        response = requests.post(url, json=request_body, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data["data"]["answer"]
        else:
            return f"请求失败: {response.status_code}"
    except Exception as e:
        return f"请求异常: {str(e)}"


def get_video_bytes(video_path):
    with open(video_path, "rb") as f:
        return f.read()

def text_to_speech(text: str):
    """将传递的文本转换成音频数据"""
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            response_format="mp3",
            input=text,
        )
        audio_file = Path(__file__).parent / f"storage/audio/{str(uuid.uuid4())}.mp3"
        response.stream_to_file(audio_file)
        return audio_file
    except Exception as e:
        return f"语音生成失败: {str(e)}"


with gr.Blocks() as demo:
    gr.Markdown('# EchoMimic')
    gr.Markdown('![]()')
    with gr.Row():
        with gr.Column():
            # 数字人主题输入框&点击按钮
            input_box = gr.Textbox(label="文案主题", placeholder="请输入数字人口播文案主题")
            fetch_button = gr.Button("获取文案")
            output_box = gr.Textbox(
                label="文案内容",
                placeholder="请输入数字人口播文案内容或通过AI生成",
                lines=5,
                interactive=True
            )

            # 将文案装换成音频
            tts_button = gr.Button("转换为语音")
            uploaded_audio = gr.Audio(type="filepath", label="Input Audio")
            uploaded_img = gr.Image(type="filepath", label="Reference Image")

            # 绑定事件
            fetch_button.click(fetch_data, inputs=input_box, outputs=output_box)
            tts_button.click(text_to_speech, inputs=output_box, outputs=uploaded_audio)
        with gr.Column():
            output_video = gr.Video()

    with gr.Accordion("Configuration", open=False):
        width = gr.Slider(label="Width", minimum=128, maximum=1024, value=default_values["width"])
        height = gr.Slider(label="Height", minimum=128, maximum=1024, value=default_values["height"])
        length = gr.Slider(label="Length", minimum=100, maximum=5000, value=default_values["length"])
        seed = gr.Slider(label="Seed", minimum=0, maximum=10000, value=default_values["seed"])
        facemask_dilation_ratio = gr.Slider(label="Facemask Dilation Ratio", minimum=0.0, maximum=1.0, step=0.01,
                                            value=default_values["facemask_dilation_ratio"])
        facecrop_dilation_ratio = gr.Slider(label="Facecrop Dilation Ratio", minimum=0.0, maximum=1.0, step=0.01,
                                            value=default_values["facecrop_dilation_ratio"])
        context_frames = gr.Slider(label="Context Frames", minimum=0, maximum=50, step=1,
                                   value=default_values["context_frames"])
        context_overlap = gr.Slider(label="Context Overlap", minimum=0, maximum=10, step=1,
                                    value=default_values["context_overlap"])
        cfg = gr.Slider(label="CFG", minimum=0.0, maximum=10.0, step=0.1, value=default_values["cfg"])
        steps = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=default_values["steps"])
        sample_rate = gr.Slider(label="Sample Rate", minimum=8000, maximum=48000, step=1000,
                                value=default_values["sample_rate"])
        fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=default_values["fps"])
        device = gr.Radio(label="Device", choices=["cuda", "cpu"], value=default_values["device"])

    generate_button = gr.Button("Generate Video")


    def generate_video(uploaded_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio,
                       facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, device):
        try:
            final_output_path, _ = process_video(
                uploaded_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio,
                facecrop_dilation_ratio,
                context_frames, context_overlap, cfg, steps, sample_rate, fps, device
            )
            # 返回视频文件和路径文本
            return final_output_path, str(final_output_path)
        except Exception as e:
            # 返回错误信息
            error_msg = f"生成视频失败: {str(e)}"
            return None, error_msg

    with gr.Row(visible=False):  # 隐藏在Web界面中，仅通过API调用
        video_path_input = gr.Textbox()  # 接收视频路径
        video_bytes_output = gr.Textbox()  # 输出字节流（Gradio会自动处理）
        video_path_input.change(
            fn=get_video_bytes,
            inputs=video_path_input,
            outputs=video_bytes_output
        )

    generate_button.click(
        generate_video,
        inputs=[
            uploaded_img,
            uploaded_audio,
            width,
            height,
            length,
            seed,
            facemask_dilation_ratio,
            facecrop_dilation_ratio,
            context_frames,
            context_overlap,
            cfg,
            steps,
            sample_rate,
            fps,
            device
        ],
        outputs=[output_video, video_path_input]
    )
parser = argparse.ArgumentParser(description='EchoMimic')
parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
parser.add_argument('--server_port', type=int, default=7860, help='Server port')
args = parser.parse_args()


# demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=True)


if __name__ == '__main__':
    # demo.launch(server_name='0.0.0.0')
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    demo.launch(
        server_name=args.server_name,
        server_port=7860,
        inbrowser=True,
        share=True,
        app_kwargs={"timeout_keep_alive": 6000}
    )
