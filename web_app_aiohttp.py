import aiohttp
from aiohttp import web
import ssl
import json
import uuid
import os
import logging
import asyncio
import shutil
import torch
import OpenSSL
from pathlib import Path
from datetime import datetime
import time
from werkzeug.utils import secure_filename
from typing import Dict, List, Any, Optional

# Импорт модулей ML из оригинального кода
from torch import nn
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
import argparse

from hallo.animate.face_animate import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.utils.config import filter_non_none
from hallo.utils.util import tensor_to_video

### hallo.utils.util
def tensor_to_video(tensor, output_video_file, audio_source, img_size_orig, fps=25):
    """
    Simplified version that automatically handles dimension issues.
    Fixed scaling for wide format images.
    """
    import numpy as np
    from moviepy.editor import VideoClip, AudioFileClip
    import cv2
    
    # Convert tensor to numpy array [f, h, w, c]
    tensor = tensor.permute(1, 2, 3, 0).cpu().numpy()
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    
    # Ensure original image size is divisible by 2
    target_width = (img_size_orig[0] // 2) * 2
    target_height = (img_size_orig[1] // 2) * 2
    
    def make_frame(t):
        frame_index = min(int(t * fps), tensor.shape[0] - 1)
        frame = tensor[frame_index]
        
        # Resize frame to exact target dimensions WITHOUT preserving aspect ratio
        # This will stretch the image to fill the entire frame
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    duration = tensor.shape[0] / fps
    video_clip = VideoClip(make_frame, duration=duration)
    
    try:
        audio_clip = AudioFileClip(audio_source).subclip(0, duration)
        video_clip = video_clip.set_audio(audio_clip)
    except Exception as e:
        print(f"Warning: Could not add audio: {e}")
    
    # Critical: Use these specific settings for H.264 compatibility
    video_clip.write_videofile(
        output_video_file,
        fps=fps,
        codec='libx264',
        audio_codec='aac',
        bitrate='5000k',
        preset='medium',
        ffmpeg_params=[
            '-pix_fmt', 'yuv420p',  # This is crucial
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-movflags', '+faststart'
        ]
    )
    
    video_clip.close()
    if 'audio_clip' in locals():
        audio_clip.close()


# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Конфигурация
API_KEY = os.getenv("API_KEY", "default-api-key-change-me")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
TASK_TIMEOUT = 3600  # 1 hour

def process_audio_emb(audio_emb):
    """
    Process the audio embedding to concatenate with other tensors.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb

class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        imageproj,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(self,):
        """
        empty function to override abstract function of nn Module
        """

    def get_modules(self):
        """
        Simple method to avoid too-few-public-methods pylint error
        """
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "imageproj": self.imageproj,
            "audioproj": self.audioproj,
        }

#class TaskManager:
#    def __init__(self):
#        self.tasks: Dict[str, Dict] = {}
#    
#    def create_task(self, task_data: Dict) -> str:
#        task_id = str(uuid.uuid4())
#        task_data.update({
#            "task_id": task_id,
#            "status": "queued",
#            "progress": 0.0,
#            "created_at": datetime.now().isoformat(),
#            "completed_at": None,
#            "error": None
#        })
#        self.tasks[task_id] = task_data
#        return task_id

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.task_queue = asyncio.Queue()
        self.is_processing = False
    
    def create_task(self, task_data: Dict) -> str:
        task_id = str(uuid.uuid4())
        task_data.update({
            "task_id": task_id,
            "status": "queued",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None
        })
        self.tasks[task_id] = task_data
        
        # Add task to the queue
        asyncio.create_task(self.task_queue.put(task_id))
        
        # Start queue processing if not already running
        if not self.is_processing:
            asyncio.create_task(self._process_queue())
            
        return task_id
    
    async def _process_queue(self):
        """Process tasks in the queue one after another"""
        self.is_processing = True
        
        while True:
            try:
                # Get the next task from the queue
                task_id = await self.task_queue.get()
                
                if task_id not in self.tasks:
                    self.task_queue.task_done()
                    continue
                
                # Get the application context
                app = self.tasks[task_id].get('app')
                if not app:
                    self.tasks[task_id]['status'] = 'failed'
                    self.tasks[task_id]['error'] = 'Application context not found'
                    self.task_queue.task_done()
                    continue
                
                # Process the task
                try:
                    self.tasks[task_id]['status'] = 'processing'
                    await process_video_task(app, task_id)
                    self.tasks[task_id]['status'] = 'completed'
                    self.tasks[task_id]['completed_at'] = datetime.now().isoformat()
                except Exception as e:
                    self.tasks[task_id]['status'] = 'failed'
                    self.tasks[task_id]['error'] = str(e)
                    self.tasks[task_id]['completed_at'] = datetime.now().isoformat()
                
                # Mark task as done in the queue
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Unexpected error in queue processing: {e}")
                await asyncio.sleep(1)  # Avoid tight loop on persistent errors
        
        self.is_processing = False


async def verify_client_cert(request: web.Request) -> bool:
    ssl_info = request.transport.get_extra_info('ssl_object')
    
    if not ssl_info or not ssl_info.getpeercert():
        raise web.HTTPForbidden(reason="Client certificate required")
        
    # Пример проверки отпечатка
    client_cert = ssl_info.getpeercert(binary_form=True)
    cert_dn = dict(x[0] for x in ssl_info.getpeercert()['subject'])
    client_cn = cert_dn.get('commonName', 'Unknown')    
    # Конвертируем DER в PEM и вычисляем SHA1
    pem_cert = ssl.DER_cert_to_PEM_cert(client_cert)
    cert = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, pem_cert)
    cert_fingerprint = cert.digest("sha1").decode().replace(":", "").lower()    
    
    print ("------------>", cert_fingerprint)    
    
    expected_fingerprint = '2A420DA5F9E08CF6475EBD1C395E2456AC638193'
    
#    if cert_fingerprint != expected_fingerprint.lower():
#        raise web.HTTPForbidden(reason="Invalid client certificate")
    
    return True

async def verify_api_key(request: web.Request) -> bool:
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        raise web.HTTPUnauthorized(reason="Invalid API key")
    return True

async def init_models(app: web.Application):
    """Инициализация моделей ML"""
    logger.info("Initializing ML models...")
    
    # Парсинг аргументов
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/inference/default.yaml")
    args = parser.parse_args(["--config", "configs/inference/default.yaml"])
    
    # Загрузка конфигурации
    config = OmegaConf.load(args.config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    # Инициализация моделей (аналогично оригинальному коду)
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    vae = AutoencoderKL.from_pretrained(config.vae.model_path)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.base_model_path, subfolder="unet")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs),
        use_landmark=False,
    )
    
    face_locator = FaceLocator(conditioning_embedding_channels=320)
    image_proj = ImageProjModel(
        cross_attention_dim=denoising_unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    )

    audio_proj = AudioProjModel(
        seq_len=5,
        blocks=12,  # use 12 layers' hidden states of wav2vec
        channels=768,  # audio embedding channel
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ).to(device=device, dtype=weight_dtype)

    audio_ckpt_dir = config.audio_ckpt_dir
    
    # Freeze models
    vae.requires_grad_(False)
    image_proj.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    face_locator.requires_grad_(False)
    audio_proj.requires_grad_(False)

    reference_unet.enable_gradient_checkpointing()
    denoising_unet.enable_gradient_checkpointing()

    # Create the Net model
    net = Net(
        reference_unet,
        denoising_unet,
        face_locator,
        image_proj,
        audio_proj,
    )

    # Load weights
    m, u = net.load_state_dict(
        torch.load(
            os.path.join(audio_ckpt_dir, "net.pth"),
            map_location="cpu",
        ),
    )
    assert len(m) == 0 and len(u) == 0, "Failed to load correct checkpoint."
    logger.info(f"Loaded weights from {os.path.join(audio_ckpt_dir, 'net.pth')}")
    
    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        face_locator=FaceLocator(conditioning_embedding_channels=320),
        scheduler=val_noise_scheduler,
        image_proj=ImageProjModel(
            cross_attention_dim=denoising_unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=4,
        ),
    ).to(device)
    
    app['models'] = {
        'pipeline': pipeline,
        'config': config,
        'device': device,
        'net': net
    }
    logger.info("ML models initialized")

async def cleanup_models(app: web.Application):
    """Очистка ресурсов"""
    if 'models' in app:
        del app['models']
    torch.cuda.empty_cache()

async def handle_generate_video(request: web.Request) -> web.Response:
    """Обработка запроса на генерацию видео"""
    await verify_client_cert(request)
    #await verify_api_key(request)
    
    data = await request.post()
    task_mgr = request.app['task_manager']
    models = request.app['models']
    
    # Сохранение файлов
    file_paths = {}
    for field in ['audio', 'image']:
        file = data.get(field)
        if file and isinstance(file, web.FileField):
            filename = secure_filename(file.filename)
            save_path = UPLOAD_DIR / filename
            save_path.write_bytes(file.file.read())
            file_paths[field] = save_path
    print ("HANDLE_GENERATE_VIDEO", data.get('callback_url'))
    # Создание задачи
    task_id = task_mgr.create_task({
        **file_paths,
        "params": json.loads(data.get('video_params', '{}')),
        "callback_url": data.get('callback_url'),
        "app": request.app  # Передаем ссылку на приложение
    })
    
    # Запуск фоновой задачи
    #asyncio.create_task(process_video_task(request.app, task_id))
    
    return web.json_response({
        "task_id": task_id,
        "status": "queued",
        "created_at": task_mgr.tasks[task_id]['created_at']
    })

## Работает
#async def process_video_task(app: web.Application, task_id: str):
#    """Фоновая задача обработки видео с полной логикой генерации"""
#    task = app['task_manager'].tasks[task_id]
#    save_path = OUTPUT_DIR / f"temp_{task_id}"
#    output_path = OUTPUT_DIR / f"{task_id}.mp4"
#    # Create a temporary save path for intermediate files
#    os.makedirs(save_path, exist_ok=True)
#    
#    
#    models = app['models']
#    config = models['config']
#    pipeline = models['pipeline']
#    device = models['device']
#    net = models['net']
#        
#    
#    try:
#        # 1. Prepare source image, face mask, face embeddings
#        img_size = (config.data.source_image.width, config.data.source_image.height)
#        clip_length = config.data.n_sample_frames
#        face_analysis_model_path = config.face_analysis.model_path
#        
#        with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
#            source_image_pixels, \
#            source_image_face_region, \
#            source_image_face_emb, \
#            source_image_full_mask, \
#            source_image_face_mask, \
#            source_image_lip_mask = image_processor.preprocess(str(task['image']), str(save_path), config.face_expand_ratio)

#        # 2. Prepare audio embeddings
#        sample_rate = config.data.driving_audio.sample_rate
#        assert sample_rate == 16000, "audio sample rate must be 16000"
#        fps = config.data.export_video.fps
#        wav2vec_model_path = config.wav2vec.model_path
#        wav2vec_only_last_features = config.wav2vec.features == "last"
#        audio_separator_model_file = config.audio_separator.model_path
#        
#        with AudioProcessor(
#            sample_rate,
#            fps,
#            wav2vec_model_path,
#            wav2vec_only_last_features,
#            os.path.dirname(audio_separator_model_file),
#            os.path.basename(audio_separator_model_file),
#            os.path.join(save_path, "audio_preprocess")
#        ) as audio_processor:
#            audio_emb, audio_length = audio_processor.preprocess(str(task['audio']), clip_length)

#        # 3. Process audio embeddings
#        audio_emb = process_audio_emb(audio_emb)
#        # 4. Prepare tensors for inference
#        source_image_pixels = source_image_pixels.unsqueeze(0)
#        source_image_face_region = source_image_face_region.unsqueeze(0)
#        source_image_face_emb = source_image_face_emb.reshape(1, -1)
#        source_image_face_emb = torch.tensor(source_image_face_emb)

#        source_image_full_mask = [
#            (mask.repeat(clip_length, 1))
#            for mask in source_image_full_mask
#        ]
#        source_image_face_mask = [
#            (mask.repeat(clip_length, 1))
#            for mask in source_image_face_mask
#        ]
#        source_image_lip_mask = [
#            (mask.repeat(clip_length, 1))
#            for mask in source_image_lip_mask
#        ]

#        times = audio_emb.shape[0] // clip_length
#        tensor_result = []
#        generator = torch.manual_seed(42)
#        motion_scale = [
#            task['params'].get('pose_weight', 1.0),
#            task['params'].get('face_weight', 1.0),
#            task['params'].get('lip_weight', 1.0)
#        ]
#        for t in range(times):
#            print(f"[{t+1}/{times}]")

#            if len(tensor_result) == 0:
#                # The first iteration
#                motion_zeros = source_image_pixels.repeat(
#                    config.data.n_motion_frames, 1, 1, 1)
#                motion_zeros = motion_zeros.to(
#                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
#                pixel_values_ref_img = torch.cat(
#                    [source_image_pixels, motion_zeros], dim=0)  # concat the ref image and the first motion frames
#            else:
#                motion_frames = tensor_result[-1][0]
#                motion_frames = motion_frames.permute(1, 0, 2, 3)
#                motion_frames = motion_frames[0-config.data.n_motion_frames:]
#                motion_frames = motion_frames * 2.0 - 1.0
#                motion_frames = motion_frames.to(
#                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
#                pixel_values_ref_img = torch.cat(
#                    [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames

#            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

#            audio_tensor = audio_emb[
#                t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])
#            ]
#            audio_tensor = audio_tensor.unsqueeze(0)
#            audio_tensor = audio_tensor.to(
#                device=net.audioproj.device, dtype=net.audioproj.dtype)
#            audio_tensor = net.audioproj(audio_tensor)

#            print (".......................>>>>>>", img_size, audio_emb.shape, 
#                   pixel_values_ref_img.shape, config.data.n_sample_frames, len(motion_scale), motion_scale, source_image_face_region.shape) 
#            # (512, 512) torch.Size([192, 5, 12, 768]) torch.Size([1, 3, 3, 512, 512]) 16 <class 'list'> <class 'torch.Tensor'>
#            # (512, 512) torch.Size([192, 5, 12, 768]) torch.Size([1, 3, 3, 512, 512]) 16 3 [1.0, 1.0, 1.0] torch.Size([1, 3, 512, 512])

#            # (512, 512) torch.Size([192, 5, 12, 768]) torch.Size([1, 3, 3, 512, 512]) 16 3 [1.0, 1.0, 1.0] torch.Size([1, 3, 512, 512])

###             Запуск пайплайна

#            pipeline_output = pipeline(
#                ref_image=pixel_values_ref_img,
#                audio_tensor=audio_tensor,
#                face_emb=source_image_face_emb,
#                face_mask=source_image_face_region,
#                pixel_values_full_mask=source_image_full_mask,
#                pixel_values_face_mask=source_image_face_mask,
#                pixel_values_lip_mask=source_image_lip_mask,
#                width=img_size[0],
#                height=img_size[1],
#                video_length=clip_length,
#                num_inference_steps=config.inference_steps,
#                guidance_scale=config.cfg_scale,
#                generator=generator,
#                motion_scale=motion_scale,
#            )

#            tensor_result.append(pipeline_output.videos)

#        tensor_result = torch.cat(tensor_result, dim=2)
#        tensor_result = tensor_result.squeeze(0)
#        tensor_result = tensor_result[:, :audio_length]

#        # 5. Save the result
#        tensor_to_video(tensor_result, output_path, str(task['audio']))
#        
#        return True
#    except Exception as e:
#        print(f"Error in video generation: {str(e)}")
#        return False
#    finally:
#        # Clean up temporary files
#        shutil.rmtree(save_path, ignore_errors=True)


async def process_video_task(app: web.Application, task_id: str):
    """Фоновая задача обработки видео с полной логикой генерации"""
    task = app['task_manager'].tasks[task_id]
    save_path = OUTPUT_DIR / f"temp_{task_id}"
    output_path = OUTPUT_DIR / f"{task_id}.mp4"
    # Create a temporary save path for intermediate files
    os.makedirs(save_path, exist_ok=True)
    
    
    models = app['models']
    config = models['config']
    pipeline = models['pipeline']
    device = models['device']
    net = models['net']
  
    # 1. Подготовка исходного изображения
    task['status'] = "processing"
    task['progress'] = 0.1        
    
    user_id = str(task['image']).split("_")[2]
    try:
        # 1. Prepare source image, face mask, face embeddings
        img_size = (config.data.source_image.width, config.data.source_image.height)
        clip_length = config.data.n_sample_frames
        face_analysis_model_path = config.face_analysis.model_path
        print ("CLIP_LENGTH", clip_length)
        try:
            with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
                source_image_pixels, \
                source_image_face_region, \
                source_image_face_emb, \
                source_image_full_mask, \
                source_image_face_mask, \
                source_image_lip_mask, \
                img_size_orig = image_processor.preprocess(str(task['image']), str(save_path), config.face_expand_ratio)
        except IndexError:
            print ("НЕТ ИЗОБРАЖЕНИЯ")
            task.update({
                "status": "error",
                "progress": 0.0,
                "completed_at": datetime.now().isoformat(),
            })
            await send_callback(task)
            return
                

        # 2. Prepare audio embeddings
        sample_rate = config.data.driving_audio.sample_rate
        assert sample_rate == 16000, "audio sample rate must be 16000"
        fps = config.data.export_video.fps
        wav2vec_model_path = config.wav2vec.model_path
        wav2vec_only_last_features = config.wav2vec.features == "last"
        audio_separator_model_file = config.audio_separator.model_path
        
        # 2. Подготовка аудио
        task['progress'] = 0.25
        with AudioProcessor(
            config.data.driving_audio.sample_rate,
            config.data.export_video.fps,
            config.wav2vec.model_path,
            config.wav2vec.features == "last",
            os.path.dirname(config.audio_separator.model_path),
            os.path.basename(config.audio_separator.model_path),
            str(save_path / "audio_preprocess")
        ) as audio_processor:
            audio_emb, audio_length = await asyncio.to_thread(
                audio_processor.preprocess,
                str(task['audio']),
                clip_length
            )


        # 3. Process audio embeddings
        audio_emb = process_audio_emb(audio_emb)
        # 4. Prepare tensors for inference
        source_image_pixels = source_image_pixels.unsqueeze(0)
        source_image_face_region = source_image_face_region.unsqueeze(0)
        source_image_face_emb = source_image_face_emb.reshape(1, -1)
        source_image_face_emb = torch.tensor(source_image_face_emb)

        source_image_full_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_full_mask
        ]
        source_image_face_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_face_mask
        ]
        source_image_lip_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_lip_mask
        ]

        times = audio_emb.shape[0] // clip_length
        tensor_result = []
        generator = torch.manual_seed(42)
        motion_scale = [
            task['params'].get('pose_weight', 1.0),
            task['params'].get('face_weight', 1.0),
            task['params'].get('lip_weight', 1.0)
        ]
        
        #await send_callback(task)
        print ("SEND_CALLBACK---", task, user_id)
        if user_id == "naturalkind":
            config.inference_steps = 10
        else:
            times_max = 14 #10 #4
            times = min([times_max, times])
            config.inference_steps = 5
            
            
        _test_time_start = time.time()
        for t in range(times):
            if len(tensor_result) == 0:
                # The first iteration
                motion_zeros = source_image_pixels.repeat(
                    config.data.n_motion_frames, 1, 1, 1)
                motion_zeros = motion_zeros.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_zeros], dim=0)  # concat the ref image and the first motion frames
            else:
                motion_frames = tensor_result[-1][0]
                motion_frames = motion_frames.permute(1, 0, 2, 3)
                motion_frames = motion_frames[0-config.data.n_motion_frames:]
                motion_frames = motion_frames * 2.0 - 1.0
                motion_frames = motion_frames.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames

            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

            audio_tensor = audio_emb[
                t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])
            ]
            audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(
                device=net.audioproj.device, dtype=net.audioproj.dtype)
            audio_tensor = net.audioproj(audio_tensor)

#            clip_length = 3
#            config.inference_steps = 10
##             Запуск пайплайна
            pipeline_output = await asyncio.to_thread(
                pipeline,
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                face_emb=source_image_face_emb,
                face_mask=source_image_face_region,
                pixel_values_full_mask=source_image_full_mask,
                pixel_values_face_mask=source_image_face_mask,
                pixel_values_lip_mask=source_image_lip_mask,
                width=img_size[0],
                height=img_size[1],
                video_length=clip_length,
                num_inference_steps=config.inference_steps,
                guidance_scale=config.cfg_scale,
                generator=generator,
                motion_scale=motion_scale,
            )
            task['progress'] = 0.3 + 0.6 * (t / times)
            print (f"[{t+1}/{times}] OUT --------->", task['progress'], 
                   clip_length, config.inference_steps)
            tensor_result.append(pipeline_output.videos)
            torch.cuda.empty_cache()
            
        _test_time_end = time.time() - _test_time_start
        tensor_result = torch.cat(tensor_result, dim=2)
        tensor_result = tensor_result.squeeze(0)
        tensor_result = tensor_result[:, :audio_length]

        # 5. Save the result
        print (f"IMG SIZE: {img_size_orig} SAVE THE RESULT----------->", output_path, str(task['audio']), _test_time_end) # 415.62347054481506
        tensor_to_video(tensor_result, str(output_path), str(task['audio']), img_size_orig)
        torch.cuda.empty_cache() 
        # Обновление статуса
        task.update({
            "status": "completed",
            "progress": 1.0,
            "completed_at": datetime.now().isoformat(),
            "output_path": str(output_path)
        })
        print ("CALLBACK_URL----------->", task.get('callback_url'))
        # Отправка webhook
        if task.get('callback_url'):
            await send_callback(task)
        
        
#        await send_callback({"task_id":"603789567", "status": "completed"})


    except asyncio.CancelledError:
        logger.info(f"Task {task_id} cancelled")
        task.update({
            "status": "cancelled",
            "completed_at": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        task.update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
    finally:
        # Очистка временных файлов
        await asyncio.to_thread(shutil.rmtree, save_path, ignore_errors=True)
        torch.cuda.empty_cache()


async def send_callback(task: Dict):
    """Отправка webhook уведомления
       Создаёмo client_trust.pem:
       penssl x509 -in YOURPUBLIC.pem -out client_trust.pem"""
    ssl_context = ssl.create_default_context(cafile='ssl/client_trust.pem') 
    ssl_context.check_hostname = False # не проверяем имея сервера, иначе ошибка
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        if task["status"] == "completed":
            await session.post(
                #task['callback_url'],
                "https://178.158.131.41:8443/video_callback",
                json={
                    "task_id": task['task_id'],
                    "status": task['status'],
                    "download_url": f"/download/{task['task_id']}",  # Добавляем URL для скачивания 
                    #"download_url": task.get('output_path'),
                    #"error": task.get('error')
                }
            )    
        else:
            await session.get("https://178.158.131.41:8443/video_callback",
                                json={
                                    "task_id": task['task_id'],
                                    "status": task['status'],
                                    "progress": task['progress']
                                })
async def handle_download(request: web.Request) -> web.FileResponse:
    """Обработчик скачивания готового видео"""
    await verify_api_key(request)
    task_id = request.match_info['task_id']
    task = request.app['task_manager'].tasks.get(task_id)
    print ("--------HANDLE_DOWNLOAD--------", task_id)
    # Проверка существования задачи
    if not task:
        raise web.HTTPNotFound(text=json.dumps({"error": "Task not found"}), 
                             content_type="application/json")

    # Проверка статуса задачи
    if task['status'] != 'completed':
        raise web.HTTPBadRequest(text=json.dumps({"error": "Video not ready"}), 
                               content_type="application/json")

    # Проверка существования файла
    output_path = Path(task.get('output_path', ''))
    if not output_path.exists():
        raise web.HTTPNotFound(text=json.dumps({"error": "File not found"}), 
                             content_type="application/json")

    # Отправка файла
    return web.FileResponse(
        path=output_path,
        headers={
            "Content-Disposition": f'attachment; filename="video_{task_id}.mp4"'
        }
    )

async def handle_status(request):
    await verify_api_key(request)
    task_id = request.match_info['task_id']
    task = request.app['task_manager'].tasks.get(task_id)
    if not task:
        raise web.HTTPNotFound(reason="Task not found")
    
    return web.json_response({
        "status": task["status"],
        "progress": task["progress"],
        "created_at": task["created_at"]
    })


async def handle_cancel(request):
    await verify_api_key(request)
    task_id = request.match_info['task_id']
    task = request.app['task_manager'].tasks.get(task_id)
    
    if not task:
        raise web.HTTPNotFound(reason="Task not found")
    
    if task['status'] in ['completed', 'failed']:
        raise web.HTTPBadRequest(reason="Cannot cancel finished task")
    
    task['status'] = 'cancelled'
    return web.json_response({"status": "cancelled"})

async def health_check(request):
    return web.json_response({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available()
    })

def init_app() -> web.Application:
    app = web.Application(client_max_size=1024*1024*400)
    app['task_manager'] = TaskManager()
    
    # Инициализация моделей при старте
    app.on_startup.append(init_models)
    app.on_cleanup.append(cleanup_models)
    
    # Регистрация роутов
    app.router.add_post('/generate_video', handle_generate_video)
    app.router.add_get('/download/{task_id}', handle_download)
    app.router.add_get('/status/{task_id}', handle_status)
    app.router.add_delete('/cancel/{task_id}', handle_cancel)
    app.router.add_get('/health', health_check)
    
    return app

if __name__ == '__main__':
    # Создание директорий
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # SSL конфигурация
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain('ssl/server.crt', 'ssl/server.key')
    ssl_context.load_verify_locations('ssl/ca.crt')
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    # Запуск приложения
    web.run_app(init_app(), port=5000, ssl_context=ssl_context)
