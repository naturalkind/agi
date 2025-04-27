import tornado.httpserver
import tornado.ioloop
import tornado.web
import ssl
import json
import requests
import logging
import zlib
import pickle
import zmq
import re
from zmq.asyncio import Context
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import redis
import sqlite3
from io import StringIO
import aiohttp
import asyncio
from datetime import datetime
from datetime import timedelta
import calendar
import time
import os
from multiprocessing import Process, Queue
import multiprocessing as mp
from pydub import AudioSegment
from io import BytesIO
import soundfile as sf
import numpy as np
import librosa
import cairosvg
import intel_extension_for_pytorch as ipex
from contextlib import contextmanager
from urllib.parse import urljoin
import gc
import base64

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Инициализация устройства
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
logger.info(f"Using device: {device} | {torch.xpu.get_device_name(0)} | {torch.xpu.is_available()}")

# Конфигурация моделей
WHISPER_MODEL_ID = "whisper-large-v3"

# Загрузка конфигурации
with open('config.bot', 'r') as json_file:
    data = json.load(json_file)
    
# Убедимся, что директория пользователя существует
if not os.path.exists('data_users'):
    os.makedirs('data_users')

BOT_TOKEN = data['BOT_TOKEN']
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
SQLITE_DB = 'dialogs.db'
ZMQ_PIPELINE_ADDRESS = "tcp://127.0.0.1:5555"
ZMQ_RESULT_ADDRESS = "tcp://127.0.0.1:5556"

# Конфигурация сервера модели LLM
PHI3_SERVER_URL = "https://192.168.1.60:5000/generate"
# Конфигурация сервера модели LCM_Dreamshaper_v7
DREAMSHAPER_SERVER_URL = "https://192.168.1.60:5000/generate_image"
SSL_VERIFY = False  # Для самоподписанных сертификатов

# Настройки лимита генирации
DAILY_WORD_LIMIT = 15
MONTHLY_WORD_LIMIT = DAILY_WORD_LIMIT*10
REDIS_DAILY_PREFIX = "daily_words:"
REDIS_MONTHLY_PREFIX = "monthly_words:"

DAILY_GEN_LIMIT = 2
MONTHLY_GEN_LIMIT = DAILY_GEN_LIMIT*4
REDIS_DAILY_PREFIX_GEN = "daily_gen:"
REDIS_MONTHLY_PREFIX_GEN = "monthly_gen:"

# Redis для хранения состояния меню
REDIS_MENU_PREFIX = "menu:"

# Инициализация компонентов
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
conn = sqlite3.connect(SQLITE_DB)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS dialogs
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
     chat_id INTEGER,
     message TEXT,
     role TEXT,
     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
''')
conn.commit()

# Хранилище диалогов в памяти
dialogs = {}

# Функции для работы с лимитами использования
def get_user_gen_counts(user_id: int) -> tuple[int, int]:
    daily = int(redis_client.get(f"{REDIS_DAILY_PREFIX_GEN}{user_id}") or 0)
    monthly = int(redis_client.get(f"{REDIS_MONTHLY_PREFIX_GEN}{user_id}") or 0)
    return daily, monthly

def update_gen_counts(user_id: int):
    now = datetime.now()
    
    # Daily counter with TTL
    daily_key = f"{REDIS_DAILY_PREFIX_GEN}{user_id}"
    redis_client.incr(daily_key, 1)
    if redis_client.ttl(daily_key) == -1:
        redis_client.expireat(daily_key, int((now + timedelta(days=1)).replace(hour=0, minute=0, second=0).timestamp()))
    
    # Monthly counter
    monthly_key = f"{REDIS_MONTHLY_PREFIX_GEN}{user_id}"
    redis_client.incr(monthly_key, 1)
    if redis_client.ttl(monthly_key) == -1:
        last_day = calendar.monthrange(now.year, now.month)[1]
        redis_client.expireat(monthly_key, int(now.replace(day=last_day, hour=23, minute=59, second=59).timestamp()))

def check_gen_limits(user_id: int) -> str | None:
    daily, monthly = get_user_gen_counts(user_id)
    
    if daily >= DAILY_GEN_LIMIT:
        reset_time = datetime.fromtimestamp(redis_client.ttl(f"{REDIS_DAILY_PREFIX_GEN}{user_id}") + time.time())
        return f"⚠️ Дневной лимит генераций исчерпан ({DAILY_GEN_LIMIT} в день). Сброс: {reset_time:%H:%M}"
    
    if monthly >= MONTHLY_GEN_LIMIT:
        return "⚠️ Месячный лимит генераций исчерпан. Доступ откроется в следующем месяце"
    
    return None

#----------------------->

def get_user_word_counts(user_id: int) -> tuple[int, int]:
    daily = int(redis_client.get(f"{REDIS_DAILY_PREFIX}{user_id}") or 0)
    monthly = int(redis_client.get(f"{REDIS_MONTHLY_PREFIX}{user_id}") or 0)
    return daily, monthly

def update_word_counts(user_id: int, words: int):
    now = datetime.now()
    
    # Daily counter with TTL
    daily_key = f"{REDIS_DAILY_PREFIX}{user_id}"
    redis_client.incrby(daily_key, words)
    if redis_client.ttl(daily_key) == -1:  # Set TTL only once
        redis_client.expireat(daily_key, int((now + timedelta(days=1)).replace(hour=0, minute=0, second=0).timestamp()))
    
    # Monthly counter
    monthly_key = f"{REDIS_MONTHLY_PREFIX}{user_id}"
    redis_client.incrby(monthly_key, words)
    if redis_client.ttl(monthly_key) == -1:
        last_day = calendar.monthrange(now.year, now.month)[1]
        redis_client.expireat(monthly_key, int(now.replace(day=last_day, hour=23, minute=59, second=59).timestamp()))

def check_word_limits(user_id: int) -> str | None:
    daily, monthly = get_user_word_counts(user_id)
    
    if daily >= DAILY_WORD_LIMIT:
        reset_time = datetime.fromtimestamp(redis_client.ttl(f"{REDIS_DAILY_PREFIX}{user_id}") + time.time())
        return f"⚠️ Дневной лимит генерации аудио исчерпан ({DAILY_WORD_LIMIT} слов). Сброс: {reset_time:%H:%M}"
    
    if monthly >= MONTHLY_WORD_LIMIT:
        return "⚠️ Месячный лимит генерации аудио исчерпан. Доступ откроется в следующем месяце"
    
    return None

def reset_word_limits(user_id: int, reset_daily: bool = True, reset_monthly: bool = True) -> str:
    """
    Reset daily and/or monthly word usage limits for a specific user.
    
    Args:
        user_id: The user ID whose limits will be reset
        reset_daily: Whether to reset the daily limit
        reset_monthly: Whether to reset the monthly limit
        
    Returns:
        A message indicating which limits were reset
    """
    results = []
    
    if reset_daily:
        daily_key = f"{REDIS_DAILY_PREFIX}{user_id}"
        redis_client.delete(daily_key)
        redis_client.delete(f"{REDIS_DAILY_PREFIX_GEN}{user_id}")
        results.append("daily")
        
    
    if reset_monthly:
        monthly_key = f"{REDIS_MONTHLY_PREFIX}{user_id}"
        redis_client.delete(monthly_key)
        redis_client.delete(f"{REDIS_MONTHLY_PREFIX_GEN}{user_id}")
        results.append("monthly")
    
    if not results:
        return "Нет лимитов для сброса"
    
    return f"Cброшен {' и '.join(results)} лимит слов для {user_id}"

# Функция подсчета слов
def count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

@contextmanager
def xpu_memory_scope(device="xpu:0"):
    try:
        yield
    finally:
        torch.xpu.synchronize(device)
        torch.xpu.empty_cache()
        gc.collect()

# Функции сжатия/распаковки
def compress(obj):
    return zlib.compress(pickle.dumps(obj))

def decompress(pickled):
    return pickle.loads(zlib.decompress(pickled))

def split_text(text, max_length=100):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def synthesize_speech(text, model, config, user_id):
    audio_path = f"data_users/speaker_reference_{user_id}.wav"
    chunks = split_text(text)
    wav_chunks = []
    #with torch.xpu.amp.autocast():
    for chunk in chunks:
        outputs = model.synthesize(
            chunk,
            config,
            speaker_wav=audio_path,
            language="ru",
        )
        wav_chunks.append(outputs["wav"])
    wav_path = f"data_users/{user_id}_clon_out.wav"
    sf.write(wav_path, np.concatenate(wav_chunks), samplerate=config.audio.output_sample_rate)
    return wav_path

async def query_phi3_server(messages, generation_args={"max_new_tokens":250, "temperature":0.0}):
    request_data = {
        "messages": messages,
        "max_new_tokens": generation_args.get("max_new_tokens", 250),
        "temperature": generation_args.get("temperature", 0.0)
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                PHI3_SERVER_URL,
                json=request_data,
                ssl=SSL_VERIFY,
                timeout=30*4
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Phi-3 server error: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Failed to query Phi-3 server: {str(e)}")
        return None

async def query_image_server(prompt, chat_id, message_id):
    request_data = {
        "prompt": prompt,
        "chat_id": str(chat_id),
        "message_id": str(message_id)
    }
    print (request_data)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                DREAMSHAPER_SERVER_URL,
                json=request_data,
                ssl=SSL_VERIFY,
                timeout=30*4
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Dreamshaper-7 server error: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Failed to query Dreamshaper-7 server: {str(e)}")
        return None

# Новый обработчик для callback
class VideoCallbackHandler(tornado.web.RequestHandler):
    def initialize(self, sender):
        self.sender = sender   
         
    async def get(self):
        try:
            data = json.loads(self.request.body)
            print ("GET---", data)
            task_id = data['task_id']
            status = data['status']
            # Получаем связанные данные из Redis
            task_data = redis_client.get(f"video_task:{task_id}")
            if not task_data:
                return
                
            task_data = json.loads(task_data)
            chat_id = task_data['chat_id']
            message_id = task_data['message_id']
            user_id = task_data['user_id']
            if status == "error":
                #await send_message(chat_id, message_id, "Ошибка, лицо не подходит", task_data)
                await update_status_message(chat_id, message_id, "Ошибка, лицо не подходит")
                # Удаляем временные данные
                redis_client.delete(f"video_task:{task_id}")
                unique_key = f"{chat_id}:{message_id}"
                ## завершение отображение печати
                if unique_key in typing_tasks:
                    typing_task = typing_tasks[unique_key]
                    del typing_tasks[unique_key]
                    if not typing_task.done():
                        typing_task.cancel()
                        try:
                            await typing_task
                        except asyncio.CancelledError:
                            pass
            else:
                await update_status_message(chat_id, message_id, "🎥 Генерация видео...")
#                await send_message(chat_id, message_id, "🎥 Генерация видео...", task_data)
        except Exception as e:
            logger.error(f"Video GET callback error: {str(e)}")
            
    async def post(self):
        try:
            data = json.loads(self.request.body)
            task_id = data['task_id']
            status = data['status']
            # Получаем связанные данные из Redis
            task_data = redis_client.get(f"video_task:{task_id}")
            if not task_data:
                return
                
            task_data = json.loads(task_data)
            chat_id = task_data['chat_id']
            message_id = task_data['message_id']
            user_id = task_data['user_id']
            
#            chat_id = "603789567"
#            message_id = "873"
            
            print ("WORK VIDEOCALLBACKHANDLER", data, chat_id, message_id)
            if status == 'completed':
                video_url = urljoin('https://192.168.1.50:5000/', data['download_url'])
                await self.send_video_to_telegram(chat_id, message_id, video_url)
                
                # Удаляем временные данные
                redis_client.delete(f"video_task:{task_id}")
                unique_key = f"{chat_id}:{message_id}"
                ## завершение отображение печати
                if unique_key in typing_tasks:
                    typing_task = typing_tasks[unique_key]
                    del typing_tasks[unique_key]
                    if not typing_task.done():
                        typing_task.cancel()
                        try:
                            await typing_task
                        except asyncio.CancelledError:
                            pass
                update_gen_counts(user_id)
                await delete_status_message(chat_id, message_id)
                await self.sender.send(compress({
                    'chat_id': chat_id,
                    'message_id': message_id,
                    'type': 'video_gen_done'
                }))

        except Exception as e:
            logger.error(f"Video callback error: {str(e)}")

    async def download_video(self, video_url: str) -> None:
        headers = {"X-API-Key": "default-api-key-change-me"}
        ssl_context = ssl.create_default_context(cafile='ssl/ca.crt')
        ssl_context.load_cert_chain('ssl/client.crt', 'ssl/client.key')
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED       
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_context)
        ) as session:
            async with session.get(video_url, headers=headers) as response:
                response.raise_for_status()
                return await response.read()
                    
    async def send_video_to_telegram(self, chat_id, message_id, video_url):
        async with aiohttp.ClientSession() as session:
            # Скачиваем видео с сервера генерации
            video_data = await self.download_video(video_url)

            # Отправляем видео в Telegram
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
            data = aiohttp.FormData()
            data.add_field('chat_id', str(chat_id))
            data.add_field('reply_to_message_id', str(message_id))
            data.add_field('video', video_data, 
                         filename='video.mp4',
                         content_type='video/mp4')
            
            async with session.post(url, data=data) as tg_resp:
                if tg_resp.status != 200:
                    logger.error(f"Failed to send video: {await tg_resp.text()}")

def get_user_image_path(user_id: int, file_id: str) -> str:
    """Возвращает путь к изображению пользователя с проверкой расширений"""
    base_path = f"data_users/speaker_reference_{user_id}_{file_id}"
    
    # Проверяем существование файлов с разными расширениями
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate_path = f"{base_path}{ext}"
        if os.path.exists(candidate_path):
            return candidate_path
    
    # Если файл не найден, возвращаем дефолтное изображение
    return "/home/npu/agi/media/4.jpg"

async def send_gen_image(chat_id: int, message_id: int, image_base64: str) -> bool:
    """Отправка изображения в формате base64 через Telegram Bot API"""
    try:
        # Декодируем base64 в бинарные данные
        image_data = base64.b64decode(image_base64)
    except (base64.binascii.Error, TypeError) as e:
        logging.error(f"Base64 decoding error: {str(e)}")
        return False

    # Создаем форму данных
    data = aiohttp.FormData()
    data.add_field('chat_id', str(chat_id))
    data.add_field('reply_to_message_id', str(message_id))
    
    try:
        # Добавляем изображение как файл в память
        data.add_field(
            name='photo',
            value=image_data,
            filename='generated_image.jpg',
            content_type='image/jpeg'
        )

        # Отправляем запрос
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post(url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logging.error(f"Telegram API error: {error_text}")
                    return False

                # Обрабатываем успешный ответ
                result = await response.json()
                sent_message_id = result['result']['message_id']
                await save_menu_state(chat_id, sent_message_id, 'main')
                return True

    except aiohttp.ClientError as e:
        logging.error(f"Network error: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return False

async def query_synthesize_video_server(user_id, chat_id, message_id, file_id):
    start_time = time.time()
    timeout = 600  # 10 minutes in seconds
    retry_delay = 5  # Start with 5 seconds between retry attempts
    max_retry_delay = 30  # Maximum delay between retries
    
    while time.time() - start_time < timeout:
        try:
            # Получаем текущий режим голоса
            voice_mode = redis_client.get(f"voice_mode:{user_id}") or b"neural"
            voice_mode = voice_mode.decode()
            print("QUERY_SYNTHESIZE_VIDEO_SERVER!!!!!!!!!!!!------------", voice_mode)
            
            # Выбираем соответствующий аудиофайл
            if voice_mode == "user":
                audio_path = f"data_users/speaker_reference_{user_id}.wav"
            else:
                audio_path = f"data_users/{user_id}_clon_out.wav"
            
            image_path = get_user_image_path(user_id, file_id)
            
            callback_url = "https://192.168.1.50:8443/video_callback"  # Ваш внешний URL
            
            ssl_context = ssl.create_default_context(cafile='ssl/ca.crt')
            ssl_context.load_cert_chain('ssl/client.crt', 'ssl/client.key')
            
            data = aiohttp.FormData()
            data.add_field('audio', open(audio_path, 'rb'), filename='audio.wav')
            data.add_field('image', open(image_path, 'rb'), filename=f'speaker_reference_{user_id}_{file_id}.jpg')
            data.add_field('video_params', json.dumps({"pose_weight": 1.0}))
            data.add_field('callback_url', callback_url)
            
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                async with session.post('https://192.168.1.50:5000/generate_video', data=data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        task_id = result.get('task_id')
                        # Сохраняем связь задачи с чатом
                        redis_client.setex(
                            f"video_task:{task_id}",
                            3600*3,  # 3 часа
                            json.dumps({
                                'chat_id': chat_id,
                                'message_id': message_id,
                                'user_id': user_id
                            })
                        )
                        return task_id
                    else:
                        logger.warning(f"Video server responded with status: {resp.status}, retrying in {retry_delay} seconds...")
                        
        except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error: {str(e)}, retrying in {retry_delay} seconds...")
        except Exception as e:
            logger.error(f"Video task creation failed: {str(e)}")
            # For non-connection errors, we don't retry
            return None
            
        # Wait before retrying
        await asyncio.sleep(retry_delay)
        # Implement exponential backoff (increasing the delay between retries)
        retry_delay = min(retry_delay * 1.5, max_retry_delay)
    
    # If we've exhausted our retry attempts
    logger.error(f"Failed to connect to video server after trying for {timeout} seconds")
    return None

#async def query_synthesize_video_server(user_id, chat_id, message_id, file_id):
#    try:
#        # Получаем текущий режим голоса
#        voice_mode = redis_client.get(f"voice_mode:{user_id}") or b"neural"
#        voice_mode = voice_mode.decode()
#        print ("QUERY_SYNTHESIZE_VIDEO_SERVER!!!!!!!!!!!!------------", voice_mode)
#        # Выбираем соответствующий аудиофайл
#        if voice_mode == "user":
#            audio_path = f"data_users/speaker_reference_{user_id}.wav"
#        else:
#            audio_path = f"data_users/{user_id}_clon_out.wav"
#        
#        #image_path = f"data_users/speaker_reference_{user_id}_{file_id}.jpg"  # Предполагаем наличие изображения
#        image_path = get_user_image_path(user_id, file_id)
#        
#        callback_url = "https://192.168.1.50:8443/video_callback"  # Ваш внешний URL
#        
#        ssl_context = ssl.create_default_context(cafile='ssl/ca.crt')
#        ssl_context.load_cert_chain('ssl/client.crt', 'ssl/client.key')
#        
#        
#        data = aiohttp.FormData()
#        data.add_field('audio', open(audio_path, 'rb'), filename='audio.wav')
#        data.add_field('image', open(image_path, 'rb'), filename=f'speaker_reference_{user_id}_{file_id}.jpg')
#        data.add_field('video_params', json.dumps({"pose_weight": 1.0}))
#        data.add_field('callback_url', callback_url)
#        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
#            async with session.post('https://192.168.1.50:5000/generate_video', data=data) as resp:
#                if resp.status == 200:
#                    result = await resp.json()
#                    task_id = result.get('task_id')
#                    # Сохраняем связь задачи с чатом
#                    redis_client.setex(
#                        f"video_task:{task_id}",
#                        3600*3,  # 1 час
#                        json.dumps({
#                            'chat_id': chat_id,
#                            'message_id': message_id,
#                            'user_id': user_id
#                        })
#                    )
#                    return task_id
#    except Exception as e:
#        logger.error(f"Video task creation failed: {str(e)}")
#    return None
    

async def pipeline_worker():
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    context = zmq.asyncio.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(ZMQ_PIPELINE_ADDRESS)
    sender = context.socket(zmq.PUSH)
    sender.bind(ZMQ_RESULT_ADDRESS)

    # Инициализация моделей с IPEX
    with xpu_memory_scope():
        # Инициализация Whisper
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_ID,
            torch_dtype=torch.bfloat16,
            #low_cpu_mem_usage=True,
            use_safetensors=True
        )
        whisper_model.to("xpu:0")
        whisper_model = ipex.optimize(whisper_model, dtype=torch.bfloat16)
        whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=whisper_processor.tokenizer,
            feature_extractor=whisper_processor.feature_extractor,
            torch_dtype=torch.bfloat16,
            device="xpu:0",
            chunk_length_s=30,
            batch_size=16, 
        )
        
        # Инициализация XTTS
        xtts_config = XttsConfig()
        xtts_config.load_json("./XTTS-v2/config.json")
        xtts_model = Xtts.init_from_config(xtts_config)
        xtts_model.load_checkpoint(xtts_config, checkpoint_dir="./XTTS-v2/", eval=True)
        xtts_model.to("xpu:0")
        # В конфигурации XTTS установите
        #xtts_config.batch_size = 8  # Увеличить размер батча
        #xtts_config.use_low_precision = True  # Использовать низкую точность
        
        
        async def send_status_update(chat_id, message_id, status):
            await sender.send(compress({
                'chat_id': chat_id,
                'message_id': message_id,
                'status': status,
                'type': 'status_update'
            }))
            
        # Функция для обработки голосового ответа (вынесена для устранения дублирования)
        async def process_voice_response(response_text, message_type, is_voice_input=False):
            nonlocal chat_id, message_id, message
            user_id = message['user_id']
            print ("PROCESS_VOICE_RESPONSE", user_id)
            if user_id == "naturalkind":
                DAILY_WORD_LIMIT = 2000
            else:
                DAILY_WORD_LIMIT = 15
            MONTHLY_WORD_LIMIT = DAILY_WORD_LIMIT*10
            # Получаем текущие счетчики пользователя
            daily, monthly = get_user_word_counts(user_id)
            remaining_daily = DAILY_WORD_LIMIT - daily
            remaining_monthly = MONTHLY_WORD_LIMIT - monthly

                
            # Определяем максимально допустимое количество слов
            max_allowed = min(remaining_daily, remaining_monthly)            
            # Формируем текст для отображения
            if is_voice_input:
                display_text = f"""*Стенограмма голоса:* 
                               ```{text}```"""
                await sender.send(compress({
                    'chat_id': chat_id,
                    'text': display_text,
                    'message_id': message_id,
                    'type': 'process_voice'
                }))  
                display_text = f"""{response_text}"""
                await sender.send(compress({
                    'chat_id': chat_id,
                    'text': display_text,
                    'message_id': message_id,
                    'type': 'process_voice_response'
                }))                  
#                await send_status_update(chat_id, message_id, f"{display_text}\n\n 🔊 Синтез речи...")

#                # Проверяем наличие доступных лимитов
#                if remaining_daily <= 0 or remaining_monthly <= 0:
#                    limit_msg = check_word_limits(user_id)
#                    await send_status_update(chat_id, message_id, limit_msg)
#                    await sender.send(compress({
#                        'chat_id': chat_id,
#                        'text': display_text,
#                        'message_id': message_id,
#                        'type': 'stop_typing_action'
#                    }))
#                    return
#                word_count = count_words(response_text)
#                
#                # Обрезаем текст, если превышает лимит
#                if word_count > max_allowed:
#                    words = response_text.split()[:max_allowed]
#                    response_text = ' '.join(words)
#                    word_count = max_allowed

#                # Обновляем счетчики
#                update_word_counts(user_id, word_count)
#                
#                # Синтезируем речь
#                output_path = synthesize_speech(response_text, xtts_model, xtts_config, user_id)
#                with open(output_path, 'rb') as audio_file:
#                    audio_content = audio_file.read()
#                # Отправляем голосовое сообщение
#                await sender.send(compress({
#                    'chat_id': chat_id,
#                    'audio': audio_content,
#                    'text': display_text,
#                    'message_id': message_id,
#                    'type': 'voice'
#                }))                               
            else:
                await send_status_update(chat_id, message_id, f"🔊 Синтез речи...")
                
                # Проверяем наличие доступных лимитов
                if remaining_daily <= 0 or remaining_monthly <= 0:
                    limit_msg = check_word_limits(user_id)
                    await send_status_update(chat_id, message_id, limit_msg)
                    await sender.send(compress({
                        'chat_id': chat_id,
                        'message_id': message_id,
                        'type': 'stop_typing_action_'
                    }))
                    return
                word_count = count_words(response_text)
                
                # Обрезаем текст, если превышает лимит
                if word_count > max_allowed:
                    words = response_text.split()[:max_allowed]
                    response_text = ' '.join(words)
                    word_count = max_allowed

                # Обновляем счетчики
                update_word_counts(user_id, word_count)                
                # Синтезируем речь
                output_path = synthesize_speech(response_text, xtts_model, xtts_config, user_id)
                with open(output_path, 'rb') as audio_file:
                    audio_content = audio_file.read()
                # Отправляем голосовое сообщение
                await sender.send(compress({
                    'chat_id': chat_id,
                    'audio': audio_content,
                    'message_id': message_id,
                    'type': message_type, 
                    'user_id': user_id
                }))  
        
        while True:
            try:
                # Получение и разбор сообщения
                message = decompress(await receiver.recv())
                chat_id = message['chat_id']
                message_id = message['message_id']
                message_type = message['type']
                torch.xpu.synchronize()
                logger.info(f"PIPELINE_WORKER-------------->{message_type}")
                
                # Инициализация переменных
                text = None
                response = None
                _response = None
                
                # Обработка входящих данных на основе типа сообщения
                if message_type == 'text':
                    text = message['text']
                    await send_status_update(chat_id, message_id, "🔤 Анализ текста...")
                elif message_type == 'voice':
                    audio_content = message['audio_content']
                    await send_status_update(chat_id, message_id, "🎙️ Обработка голоса...")
                    result = whisper_pipe(audio_content)
                    torch.xpu.empty_cache()
                    text = result["text"]
                    await send_status_update(chat_id, message_id, "🔤 Анализ текста...")
                elif message_type == 'file':
                    text = message['text']
                elif message_type == 'status_update_video':
                    await send_status_update(chat_id, message_id, message["status"])
                    continue  # Переходим к следующей итерации цикла
#                elif message_type == 'gen_voice':
                elif message_type.startswith(('gen_voice')):
                    # Обрабатываем gen_voice отдельно - здесь уже есть текст для синтеза
                    text = message['text']
                    response = message['text']
                elif message_type.startswith(('gen_image')):
                    # Обрабатываем gen_voice отдельно - здесь уже есть текст для синтеза
                    text = message['text'] 
                    await send_status_update(chat_id, message_id, "🖼️ Генерация изображения...")  
                    output_image = await query_image_server(text, chat_id, message_id)
                    await send_gen_image(chat_id, message_id, output_image["response"]["image"])
                    await sender.send(compress({
                        'chat_id': chat_id,
                        'message_id': message_id,
                        'type': 'stop_typing_action',
                        'text': text
                    }))
                else:
                    logger.warning(f"Неизвестный тип сообщения: {message_type}")
                    continue
                    
                # Логируем информацию только если есть текст
                if text is not None:
                    logger.info(f"PIPELINE_WORKER--------------2>{message_type}, {text}")
                
                # Генерация текстового ответа если нужно
                if message_type not in ['status_update_video', 'gen_voice', 'gen_image'] and text is not None:
                    # Генерация текста
                    cursor.execute('SELECT message, role FROM dialogs WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 5', (chat_id,))
                    history = cursor.fetchall()
                    history.reverse()
                    messages = [{"role": role, "content": msg} for msg, role in history]
                    messages.append({"role": "user", "content": text})
                    await send_status_update(chat_id, message_id, "🧠 Генерация ответа...")
                    
                    output = await query_phi3_server(messages) 
                    if output is None:
                        response = "⚠️ Ошибка при обработке запроса. Попробуйте позже."
                    else:
                        response = output.get("response", "Не удалось получить ответ")
                
                # Обработка ответа в зависимости от типа сообщения
                if message_type == 'voice':
                    if response:
                        await process_voice_response(response, message_type, is_voice_input=True)
#                elif message_type == 'gen_voice':
                elif message_type.startswith(('gen_voice')):
                    if response:
                        await process_voice_response(response, message_type, is_voice_input=False)
                elif message_type == 'text':
                    if response:
                        await sender.send(compress({
                            'chat_id': chat_id,
                            'text': response,
                            'message_id': message_id,
                            'type': 'text'
                        }))
                
                # Очистка памяти GPU
                torch.xpu.empty_cache()
                
            except Exception as e:
                torch.xpu.empty_cache()
                logger.error(f"Error in pipeline worker: {e}")

## Функция для запуска pipeline_worker
def start_pipeline_worker():
    asyncio.run(pipeline_worker())

def save_message_to_db(chat_id, message, role):
    cursor.execute('''
        INSERT INTO dialogs (chat_id, message, role)
        VALUES (?, ?, ?)
    ''', (chat_id, message, role))
    conn.commit()

def get_cached_response(question):
    return redis_client.get(question)

def cache_response(question, answer):
    redis_client.setex(question, 3600, answer)  ## Кэшируем ответ на 1 час

def reset_dialog(chat_id):
    if chat_id in dialogs:
        del dialogs[chat_id]
    cursor.execute('DELETE FROM dialogs WHERE chat_id = ?', (chat_id,))
    conn.commit()
    return "Диалог сброшен. Начнем сначала!"

def get_code_block(generated_text):
    code_start = generated_text.find('```python')
    code_end = generated_text.find('```', code_start + 1)

    if code_start != -1 and code_end != -1:
        code_block = generated_text[code_start+9:code_end].strip()
        return code_block, code_start
    else:
        return None, -1

async def delete_previous_menu(chat_id):
    menu_data = redis_client.hgetall(f"{REDIS_MENU_PREFIX}{chat_id}")
    if menu_data:
        message_id = menu_data.get(b'message_id')
        if message_id:
            try:
                await delete_message(chat_id, message_id.decode())
            except Exception as e:
                logger.error(f"Error deleting menu: {e}")
        redis_client.delete(f"{REDIS_MENU_PREFIX}{chat_id}")

async def save_menu_state(chat_id, message_id, menu_type):
    redis_client.hset(f"{REDIS_MENU_PREFIX}{chat_id}", mapping={
        "message_id": message_id,
        "type": menu_type
    })


class MessageHandler(tornado.web.RequestHandler):
    def initialize(self, sender, send_message_func, typing_tasks):
        self.sender = sender
        self.send_message_func = send_message_func
        self.typing_tasks = typing_tasks
        self.task_monitor = asyncio.create_task(self.monitor_tasks())
        
        
    async def post(self):
        try:
            data = json.loads(self.request.body)
            if 'message' in data:
                message = data['message']
                chat_id = message['chat']['id']
                message_id = message["message_id"]
                
                if 'username' in message['from']:
                    user_id =  message['from']['username']
                else:
                    user_id = message['from']['id'] 
                ## Create a unique key for each user in each chat
                unique_key = f"{chat_id}:{message_id}"
                         
                ## Обработка сообщений...
                if 'forward_from' in message:
                    print ("FORWARD_FROM", message)
                    if 'voice' in message:
                        await self.handle_voice_message(chat_id, user_id, message_id, message['voice']['file_id'])
                    elif 'text' in message:
                        await self.handle_text_message(chat_id, user_id, message_id, message['text'])
                    elif 'photo' in message:
                        if user_id == "naturalkind":
                            DAILY_GEN_LIMIT = 2000
                            telegram_photo = message['photo'][-1]
                        else:
                            DAILY_GEN_LIMIT = 2
                            telegram_photo = message['photo'][-2]
                        MONTHLY_GEN_LIMIT = DAILY_GEN_LIMIT*4
                        # Получаем текущие счетчики пользователя
                        daily, monthly = get_user_gen_counts(user_id)
                        remaining_daily = DAILY_GEN_LIMIT - daily
                        remaining_monthly = MONTHLY_GEN_LIMIT - monthly
                        # Проверяем наличие доступных лимитов
                        if remaining_daily <= 0 or remaining_monthly <= 0:
                            limit_msg = check_gen_limits(user_id)                    
                            await self.send_message_func(chat_id, message_id, limit_msg, menu_mod=False)
                        else:
                            await self.download_image(telegram_photo['file_id'], user_id)
                            print ("ИЗОБРАЖЕНИЕ!!!!", message['photo'], telegram_photo, telegram_photo['file_id'])
                            #await send_status_update(chat_id, message_id, "🎥 Генерация видео...")
                            await sender.send(compress({
                                'chat_id': chat_id,
                                'message_id': message_id,
                                'user_id':user_id, 
                                'status': "🎥 Генерация видео...",
                                'type': 'status_update_video'
                            }))
                            await self.start_typing_action(message_id, chat_id)
                            await query_synthesize_video_server(user_id, chat_id, message_id, telegram_photo['file_id'])                    
                    else:
                        await self.send_message_func(chat_id, message_id, "Пожалуйста, перешлите текстовое или голосовое сообщение")
                    return

                if 'text' in message:
                    await self.handle_text_message(chat_id, user_id, message_id, message['text'])
                elif 'voice' in message:
                    await self.handle_voice_message(chat_id, user_id, message_id, message['voice']['file_id'])
                elif 'document' in message:
                    await self.handle_document_message(chat_id, user_id, message_id, message['document'])
                elif 'photo' in message:
                    if user_id == "naturalkind":
                        DAILY_GEN_LIMIT = 2000
                        telegram_photo = message['photo'][-1]
                    else:
                        DAILY_GEN_LIMIT = 2
                        telegram_photo = message['photo'][-2]
                    MONTHLY_GEN_LIMIT = DAILY_GEN_LIMIT*4                        
                    # Получаем текущие счетчики пользователя
                    daily, monthly = get_user_gen_counts(user_id)
                    remaining_daily = DAILY_GEN_LIMIT - daily
                    remaining_monthly = MONTHLY_GEN_LIMIT - monthly
                    # Проверяем наличие доступных лимитов
                    if remaining_daily <= 0 or remaining_monthly <= 0:
                        limit_msg = check_gen_limits(user_id)                    
                        await self.send_message_func(chat_id, message_id, limit_msg, menu_mod=False)
                    else:
                        await self.download_image(telegram_photo['file_id'], user_id)
                        print ("ИЗОБРАЖЕНИЕ!!!!", message['photo'], telegram_photo, telegram_photo['file_id'])
                        #await send_status_update(chat_id, message_id, "🎥 Генерация видео...")
                        await sender.send(compress({
                            'chat_id': chat_id,
                            'message_id': message_id,
                            'user_id':user_id, 
                            'status': "🎥 Генерация видео...",
                            'type': 'status_update_video'
                        }))
                        await self.start_typing_action(message_id, chat_id)
                        await query_synthesize_video_server(user_id, chat_id, message_id, telegram_photo['file_id'])
                elif 'audio' in message:
                    await self.handle_audio_message(chat_id, user_id, message_id, message['audio']['file_id'])
                    
                else:
                    
                    await self.send_message_func(chat_id, message_id, "Пожалуйста, отправьте текстовое сообщение, голосовое сообщение или текстовый файл", menu_mod=False)         
                               
#            else:
            elif 'callback_query' in data:
                callback_query = data.get('callback_query', {})
                chat_id = callback_query.get('message', {}).get('chat', {}).get('id')
                message_id = callback_query.get('message', {}).get('message_id')
                user_id = callback_query.get('from', {}).get('username')
                data = callback_query.get('data')
                unique_key = f"{chat_id}:{message_id}"
                print ("CALLBACK !!!!!!!!!>", data, user_id)
                if data == 'about':
                    await self.send_about_message(chat_id)
                elif data == 'help':
                    await self.send_help_message(chat_id, user_id)
                elif data == 'main_menu':
                    await self.send_start_menu(chat_id)
                elif data == 'reset':
                    #await self.stop_typing_action(unique_key)
                    await self.send_reset_message(chat_id, message_id)                
                elif data == 'gen_video':
                    await self.send_gen_video_menu(chat_id, message_id)
                elif data == 'menu_close':
                    await menu_close(chat_id, message_id)
#                elif data == 'settings':
                elif data.lower().startswith(('settings')):
                    print ("SETTTTT->>>>>")
                    # использльзавать голос пользователя
                    parts = data.split()
                    if len(parts) > 1:
        #                #голос пользователя 
                        if parts[1].lower() == "user":
                            await self.handle_voice_selection(user_id, chat_id, message_id, "user")
                        elif parts[1].lower() == "neural": 
                            await self.handle_voice_selection(user_id, chat_id, message_id, "neural")                
                    else:        
                        await self.send_settings_menu(chat_id, message_id, user_id)
#                elif data == 'gen_voice':
                elif data.startswith(('gen_voice')):
                    await self.start_typing_action(message_id, chat_id)
                    parts = data.split('_')[-1]
                    if int(parts) == 0:
                        text = callback_query.get('message', {}).get('reply_to_message', {}).get('text', {})
                    elif int(parts) == 1:
                        text = callback_query.get('message', {}).get('text', {})
                    # Отправить
                    #await self.gen_voice_selection(text, chat_id, message_id, user_id)
                    await sender.send(compress({
                        'chat_id': chat_id,
                        'user_id': user_id,
                        'text': text,
                        'message_id': message_id,
                        'type': 'gen_voice'
                    }))
                elif data.startswith(('gen_image')):
                    await self.start_typing_action(message_id, chat_id) 
                    parts = data.split('_')[-1]
                    if int(parts) == 0:
                        text = callback_query.get('message', {}).get('reply_to_message', {}).get('text', {})
                    elif int(parts) == 1:
                        text = callback_query.get('message', {}).get('text', {})                   
                    
                    # Отправить на сервер генерации LCM_Dreamshaper_v7-int8-ov
                    
                    await sender.send(compress({
                        'chat_id': chat_id,
                        'user_id': user_id,
                        'text': text,
                        'message_id': message_id,
                        'type': 'gen_image'
                    }))                    
                    
                    
#                elif data == 'select_image':
#                    await self.send_image_selection(chat_id, message_id)
#                elif data.startswith('voice_'):
#                    voice_type = data.split('_')[1]
#                    await self.handle_voice_selection(chat_id, message_id, voice_type)
#                elif data.startswith('image_'):
#                    image_type = data.split('_')[1]
#                    await self.handle_image_selection(chat_id, message_id, image_type)                    
                    
                    
                    
                ## Обязательно отправляем ответ на callback-запрос
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery"
                async with aiohttp.ClientSession() as session:
                    await session.post(url, json={
                        "callback_query_id": callback_query.get('id')
                    })
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.stop_typing_action(unique_key)
    
#    async def gen_voice_selection(self, response, chat_id, message_id, user_id):
#        # Получаем текущие счетчики пользователя
#        daily, monthly = get_user_word_counts(user_id)
#        remaining_daily = DAILY_WORD_LIMIT - daily
#        remaining_monthly = MONTHLY_WORD_LIMIT - monthly
#        print ("----------->", daily, monthly, remaining_daily, remaining_monthly)
#        # Проверяем наличие доступных лимитов
#        if remaining_daily <= 0 or remaining_monthly <= 0:
#            limit_msg = check_word_limits(user_id)
#            await send_status_update(chat_id, message_id, limit_msg)
#            await sender.send(compress({
#                'chat_id': chat_id,
#                'text': response,
#                'message_id': message_id,
#                'type': 'stop_typing_action'
#            }))
#        else:
#            # Определяем максимально допустимое количество слов
#            max_allowed = min(remaining_daily, remaining_monthly)
#            word_count = count_words(response)
#            # Обрезаем текст, если превышает лимит
#            if word_count > max_allowed:
#                
#                words = response.split()[:max_allowed]
#                response = ' '.join(words)
#                word_count = max_allowed

#            # Обновляем счетчики
#            update_word_counts(user_id, word_count)
#            # Остальная логика обработки...
#            
#            output_path = synthesize_speech(response, xtts_model, xtts_config, user_id)
#            #------------------------
#            #подключаюсь к серверу
##                    await send_status_update(chat_id, message_id, "🎥 Генерация видео...")
##                    video_task_id = await query_synthesize_video_server(user_id, chat_id, message_id)
#                                                                
#            
#            # Сохраняем временный ответ
#            #response += "\n\n🎬 Видео обрабатывается..."                    
#            
#            #------------------------
#            #response = f"*Перевод:* `{text}`\n *Ответ:* `{response}`"
#            with open(output_path, 'rb') as audio_file:
#                audio_content = audio_file.read()
#            
#            await sender.send(compress({
#                'chat_id': chat_id,
#                'audio': audio_content,
#                'text': response,
#                'message_id': message_id,
#                'type': 'voice'
#            }))
        
        
        
    
    async def send_gen_video_menu(self, chat_id, message_id):
        """
        Написать здесь функцию
        предоставить выбор пользователю какую запись использовать
        пользователя или сгенерированную
        добавить изображение с лицом или выбрать из нескольких вариантов
        """
        buttons = []
        await edit_buttons(chat_id, message_id, buttons)
        print ("-------------------->SEND_GEN_VIDEO_MENU")


    async def send_settings_menu(self, chat_id, user_id, message_id):
        await self.handle_text_message(chat_id, user_id, message_id, "/stats")



    async def send_voice_selection(self, chat_id, message_id):
        voices = {
            "user": "Голос пользователя",
            "neural": "Голос бота",
            "custom": "Загрузить"
        }
        buttons = [[{"text": name, "callback_data": f"voice_{key}"} for key, name in voices.items()]]
        buttons.append([{"text": "🔙 Назад", "callback_data": "settings"}])
        await self.send_menu_message(chat_id, message_id, "Выберите голос:", inline_keyboard=buttons)

    async def send_image_selection(self, chat_id, message_id):
        images = {
            "avatar": "Случайное изображение",
            "custom": "Загрузить"
        }
        buttons = [[{"text": name, "callback_data": f"image_{key}"} for key, name in images.items()]]
        buttons.append([{"text": "🔙 Назад", "callback_data": "settings"}])
        text_info = "Загрузите изображение с лицом или автоматически случайное изображение из интерента"
        await self.send_menu_message(chat_id, message_id, text_info, inline_keyboard=buttons)

    async def handle_image_selection(self, chat_id, message_id, image_type):
        if image_type == "custom":
            await self.send_message_func(chat_id, message_id, "Отправьте изображение")
            redis_client.set(f"image_mode:{chat_id}", "custom")
        else:
            redis_client.set(f"image_mode:{chat_id}", image_type)
            await self.send_message_func(chat_id, message_id, f"Выбрано изображение: {image_type}")
            
            
    async def download_image(self, file_id: str, user_id: int) -> str:
        """
        Скачивает изображение из Telegram и сохраняет его для пользователя
        Возвращает путь к сохраненному файлу или None при ошибке
        """
        try:
            # 1. Получаем информацию о файле
            file_path = await self.get_file_path(file_id)
            if not file_path:
                logger.error("File path not found")
                return None

            # 2. Формируем URL для скачивания
            download_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
            
            # 3. Скачиваем файл
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as response:
                    if response.status == 200:
                        # 4. Определяем расширение файла
                        content_type = response.headers.get('Content-Type', '')
                        ext = 'jpg'  # значение по умолчанию
                        if 'png' in content_type:
                            ext = 'png'
                        elif 'jpeg' in content_type:
                            ext = 'jpg'
                        else:
                            logger.warning(f"Unsupported image type: {content_type}")

                        # 5. Сохраняем файл
                        save_path = f"data_users/speaker_reference_{user_id}_{file_id}.{ext}"
                        content = await response.read()
                        
                        with open(save_path, 'wb') as f:
                            f.write(content)
                        
                        logger.info(f"Image saved to {save_path}")
                        return save_path
                    
                    logger.error(f"Download failed. Status: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Image download error: {str(e)}")
            return None            
#------------------

    async def handle_reset_command(self, command_type, chat_id, message_id, text, user_id):
        """Handle various reset commands with permission check and validation"""
        ADMIN_ID = "naturalkind"
        
        if user_id != ADMIN_ID:
            await self.send_message_func(chat_id, message_id, "⚠️ Нет доступа администратора", menu_mod=False)
            return
            
        parts = text.split()
        if len(parts) != 2:
            await self.send_message_func(chat_id, message_id, f"⚠️ Ошибка", menu_mod=False)
        else:
            target_user = parts[1]
            # Set reset parameters based on command type
            reset_params = {
                "reset_daily": command_type in ["reset_daily", "reset_all"],
                "reset_monthly": command_type in ["reset_monthly", "reset_all"]
            }
            
            result = reset_word_limits(user_id=target_user, **reset_params)
            await self.send_message_func(chat_id, message_id, f"✅ {result}", menu_mod=False)

    async def handle_voice_selection(self, user_id, chat_id, message_id, voice_type):
        redis_client.set(f"voice_mode:{user_id}", voice_type)
        await self.send_message_func(chat_id, message_id, f"✅ Выбран режим генерации голоса: {voice_type}", menu_mod=False)
        
#    async def handle_voice_selection(self, chat_id, message_id, voice_type):
#        if voice_type == "custom":
#            await self.send_message_func(chat_id, message_id, "Отправьте аудиофайл с образцом голоса (формат WAV)")
#            redis_client.set(f"voice_mode:{chat_id}", "custom")
#        else:
#            redis_client.set(f"voice_mode:{chat_id}", voice_type)
#            await self.send_message_func(chat_id, message_id, f"Выбран голос: {voice_type}")

    async def handle_text_message(self, chat_id, user_id, message_id, text):
        logger.info(f"Received message from user {user_id} in chat {chat_id}: {text[:50]}...")

        if text.lower() == '/start':
            await self.send_start_menu(chat_id)
        elif text.lower() == '/help':
            await self.send_help_message(chat_id, user_id)
        elif text.lower() == '/reset':
            response = reset_dialog(chat_id)
            await self.send_message_func(chat_id, message_id, response, menu_mod=False)
        elif text.lower() == '/info':
            await self.send_about_message(chat_id)
        # Handle reset commands
        elif text.lower().startswith(('/reset_daily', '/reset_monthly', '/reset_all')):
            command_type = text.lower().split('/')[1].split()[0]
            await self.handle_reset_command(command_type, chat_id, message_id, text, user_id) 
        elif text.lower().startswith(('/settings')):
            # использльзавать голос пользователя
            parts = text.split()
            if len(parts) > 1:
#                #голос пользователя 
                if parts[1].lower() == "user":
                    await self.handle_voice_selection(user_id, chat_id, message_id, "user")
                elif parts[1].lower() == "neural": 
                    await self.handle_voice_selection(user_id, chat_id, message_id, "neural")
        elif text.lower().startswith(('/stats')):
            daily, monthly = get_user_gen_counts(user_id)
            daily_, monthly_ = get_user_word_counts(user_id)
            voice_mode = redis_client.get(f"voice_mode:{user_id}") or b"neural"
            voice_mode = voice_mode.decode()
            stats = (
                f"📊 Статистика использования:\n"
                f"• Генераций видео сегодня: {daily}/{DAILY_GEN_LIMIT}\n"
                f"• Генераций видео за месяц: {monthly}/{MONTHLY_GEN_LIMIT}\n"
                f"• Генераций голоса сегодня: {daily_}/{DAILY_WORD_LIMIT} слов\n"
                f"• Генераций голоса за месяц: {monthly_}/{MONTHLY_WORD_LIMIT} слов\n"
                f"• Генераций видео голосом: {voice_mode}\n"
            )
            await self.send_message_func(chat_id, message_id, stats, menu_mod=False)            
            
        else:
            cached_response = get_cached_response(text)
            if cached_response:
                await self.send_message_func(chat_id, message_id, cached_response.decode('utf-8'), menu_mod=True)
            else:
                await self.start_typing_action(message_id, chat_id)
                await self.sender.send(compress({
                    'chat_id': chat_id,
                    'user_id': user_id,
                    'text': text,
                    'message_id': message_id,
                    'type': 'text'
                }))

#    async def send_start_menu(self, chat_id, message_id):
#        ## Path to your local image file
#        image_path = 'robots-AI.jpg'  ## Replace with your actual image path
#        
#        try:
#            ## Read the entire file content first
#            with open(image_path, 'rb') as image_file:
#                image_data = image_file.read()
#            
#            ## Create form data for the request
#            data = aiohttp.FormData()
#            data.add_field('chat_id', str(chat_id))
#            data.add_field('photo', image_data, filename='logo.jpg', 
#                          content_type='image/jpeg')
##            data.add_field('caption', "Привет! Я AI-ассистент")
#            data.add_field('reply_to_message_id', str(message_id))
#            data.add_field('reply_markup', json.dumps({
##                "inline_keyboard": [
##                    [
##                        {"text": "🤖 О боте", "callback_data": "about"},
##                        {"text": "💬 Возможности", "callback_data": "features"}
##                    ],
##                    [
##                        {"text": "🔧 Сбросить диалог", "callback_data": "reset"},
##                        {"text": "❓ Справка", "callback_data": "help"}
##                    ]
##                ]
#                "inline_keyboard": [
#                    [
#                        {"text": "🤖 О боте", "callback_data": "about"},
#                        {"text": "📘 Инструкция", "callback_data": "help"}
#                    ],
#                    [
#                        {"text": "📹 Создать видео", "callback_data": "features"}
#                    ],
#                ]
#            }))
#            
#            ## Send request to Telegram API
#            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
#            async with aiohttp.ClientSession() as session:
#                async with session.post(url, data=data) as response:
#                    if response.status != 200:
#                        error_text = await response.text()
#                        logger.error(f"Error sending image: {error_text}")
#                    return await response.json()
#                    
#        except FileNotFoundError:
#            logger.error(f"Error: Image file not found at {image_path}")
#            return None
#        except Exception as e:
#            logger.error(f"Error occurred: {str(e)}")
#            return None

    async def send_switch_voice(self, chat_id):
        await delete_previous_menu(chat_id)
        
        try:

            data = aiohttp.FormData()
            data.add_field('chat_id', str(chat_id))
            data.add_field('reply_markup', json.dumps({
                "inline_keyboard": [
                    [
                        {"text": "🤖 О боте", "callback_data": "about"},
                        {"text": "📘 Инструкция", "callback_data": "help"}
                    ],
                    [
                        {"text": "📹 Настройки", "callback_data": "settings"}
                    ]
                ]
            }))

            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        message_id = result['result']['message_id']
                        await save_menu_state(chat_id, message_id, 'main')
                        return True
        except Exception as e:
            logger.error(f"Error sending start menu: {e}")
        return False



    async def send_start_menu(self, chat_id):
        await delete_previous_menu(chat_id)
        
        try:
            image_path = 'robots-AI.jpg'
            with open(image_path, 'rb') as f:
                image_data = f.read()

            data = aiohttp.FormData()
            data.add_field('chat_id', str(chat_id))
            data.add_field('photo', image_data, filename='menu.jpg', content_type='image/jpeg')
            data.add_field('reply_markup', json.dumps({
                "inline_keyboard": [
                    [
                        {"text": "🤖 О боте", "callback_data": "about"},
                        {"text": "📘 Инструкция", "callback_data": "help"}
                    ],
                    [
                        {"text": "📹 Настройки", "callback_data": "settings"}
                    ]
                ]
            }))

            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        message_id = result['result']['message_id']
                        await save_menu_state(chat_id, message_id, 'main')
                        return True
        except Exception as e:
            logger.error(f"Error sending start menu: {e}")
        return False



    async def handle_voice_message(self, chat_id, user_id, message_id, file_id):
        await self.start_typing_action(message_id, chat_id)
        audio_content = await self.get_file_content(file_id, is_voice=True)
        audio = AudioSegment.from_ogg(BytesIO(audio_content))
        wav_audio = BytesIO()
        audio.export(wav_audio, format="wav")
        wav_audio.seek(0)
        speaker_wav_data = wav_audio.getvalue()
        output_wav_path = f"data_users/speaker_reference_{user_id}.wav"
        audio.export(output_wav_path, format="wav")
                            
        await self.sender.send(compress({
            'chat_id': chat_id,
            'user_id': user_id,
            'audio_content': speaker_wav_data,
            'message_id': message_id,
            'type': 'voice'
        }))

    async def handle_audio_message(self, chat_id, user_id, message_id, file_id):
        await self.start_typing_action(message_id, chat_id)
        audio_content = await self.get_file_content(file_id, is_voice=True)
        audio = AudioSegment.from_mp3(BytesIO(audio_content))
        wav_audio = BytesIO()
        audio.export(wav_audio, format="wav")
        wav_audio.seek(0)
        speaker_wav_data = wav_audio.getvalue()
        output_wav_path = f"data_users/speaker_reference_{user_id}.wav"
        audio.export(output_wav_path, format="wav")
                            
        await self.sender.send(compress({
            'chat_id': chat_id,
            'user_id': user_id,
            'audio_content': speaker_wav_data,
            'message_id': message_id,
            'type': 'voice'
        }))

    async def handle_document_message(self, chat_id, user_id, message_id, document):
        file_id = document['file_id']
        file_name = document['file_name']
        caption = self.get_argument('caption', '')
        
        if file_name.endswith(('.txt', '.py', '.h', '.cpp')):
            await self.start_typing_action(message_id, chat_id)
            file_content = await self.get_file_content(file_id)
            full_content = f"Caption: {caption}\n\nFile Content:\n{file_content}" if caption else file_content
            await self.sender.send(compress({
                'chat_id': chat_id,
                'user_id': user_id,
                'text': full_content,
                'message_id': message_id,
                'type': 'file'
            }))
        else:
            await self.send_message_func(chat_id, message_id, "Пожалуйста, отправьте текстовый файл (.txt, .py, .h, .cpp)", menu_mod=False)

    async def monitor_tasks(self):
        while True:
            all_tasks = asyncio.all_tasks()
            active_tasks = [task for task in all_tasks if not task.done()]
            
            logging.info(f"Current active tasks: {len(active_tasks)}")
            for task in active_tasks:
                logging.info(f"Task: {task.get_name()}, State: {task._state}")
            await asyncio.sleep(5)  ## Мониторинг каждую минуту

    def on_finish(self):
        self.task_monitor.cancel()

    async def start_typing_action(self, message_id, chat_id):
        unique_key = f"{chat_id}:{message_id}"
        await self.stop_typing_action(unique_key)
        self.typing_tasks[unique_key] = asyncio.create_task(
            self.continuous_typing_action(unique_key, chat_id),
            name=f"typing_task_{unique_key}"
        )

    async def stop_typing_action(self, unique_key):
        if unique_key in self.typing_tasks:
            task = self.typing_tasks[unique_key]
            if not task.done():
                task.cancel()

    async def continuous_typing_action(self, unique_key, chat_id):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendChatAction"
        data = {
            "chat_id": chat_id,
            "action": "typing"
        }
        try:
            while True:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        if response.status != 200:
                            logger.error(f"Failed to send typing action. Status code: {response.status}")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            logger.info(f"Typing action cancelled for unique_key: {unique_key}")
        finally:
            ## Убедимся, что задача удалена из словаря typing_tasks
            if unique_key in self.typing_tasks:
                del self.typing_tasks[unique_key]

    async def get_file_content(self, file_id, is_voice=False):
        file_path = await self.get_file_path(file_id)
        url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read() if is_voice else await response.text()
                else:
                    logger.error(f"Failed to get file content. Status code: {response.status}")
                    return None

    async def get_file_path(self, file_id):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getFile"
        params = {'file_id': file_id}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    file_info = await response.json()
                    return file_info['result']['file_path']
                else:
                    logger.error(f"Failed to get file path. Status code: {response.status}")
                    return None


    async def send_about_message(self, chat_id):
        await delete_previous_menu(chat_id)
        
        about_text = """
        🤖 *О боте*:

        Привет! Это AI-ассистент, созданный 
        на основе современных технологий 
        машинного обечения. Человеческое 
        участие в [создании этого бота 10%](https://github.com/naturalkind/agi). 
        Статья в https://habr.com/ru/articles/881944/

        *Возможности*:
        - Анимация лица любым голосом            
        - Помощь в написании кода
        - Распознавание голосовых сообщений          
        - Общение
        - Генерация голоса        
        - Анализ текстовых документов
        
        *Технологии*:
        - Phi-3.5-mini языковая модель чат бот
        - Whisper распознавание речи
        - XTTS v2 синтез голоса
        - Hallo генерация видео
        """
        
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": about_text,
            "parse_mode": "Markdown",
            "reply_markup": json.dumps({
                "inline_keyboard": [
                    [{"text": "🔙 Главное меню", "callback_data": "main_menu"}]
                ]
            })
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    await save_menu_state(chat_id, result['result']['message_id'], 'about')
                    return True
        return False

    async def send_reset_message(self, chat_id, message_id):
        reset_response = reset_dialog(chat_id)
        await self.send_menu_message(chat_id, message_id, reset_response)


    async def send_help_message(self, chat_id, user_id):
        await delete_previous_menu(chat_id)
        
        voice_mode = redis_client.get(f"voice_mode:{user_id}") or b"neural"
        voice_mode = voice_mode.decode()
        
        help_text = f"""
        📘 *Инструкция*:

        - Отправьте текстовое сообщение для общения
        - Отправьте голосовое сообщение, бот ответит голосом спросившего
        - Поддерживается работа с текстовыми файлами (.txt, .py, .h, .cpp)
        - Бот может анимировать изображение с лицом

        *Команды*:
        - /start - Перезапуск бота
        - /help - Показать справку
        - /info - Информация о боте
        - /reset - Сбросить текущий диалог
        - /stats - Статистика доступной генерации

        *✅ Выбран голоса для генерации видео*: `{voice_mode}`
        """
        #- /settings neural или user - Выбор голоса для генерации видео
        
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": help_text,
            "parse_mode": "Markdown",
            "reply_markup": json.dumps({
                "inline_keyboard": [
                    [{"text": "neural", "callback_data": "settings neural"},
                     {"text": "user", "callback_data": "settings user"}],
                    [{"text": "🔙 Главное меню", "callback_data": "main_menu"}]
                ]
            })
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    await save_menu_state(chat_id, result['result']['message_id'], 'help')
                    return True
        return False


    async def send_menu_message(self, chat_id, message_id, text, inline_keyboard = [[{"text": "📘 Инструкция", "callback_data": "help"}]]):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_to_message_id": message_id,
            "reply_markup": json.dumps({
                "inline_keyboard": inline_keyboard
            })
        }
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=data)

async def send_message(chat_id, message_id, text, typing_tasks, menu_mod):
    print ("MENU_MOD =====>", menu_mod)
    code_block, code_start = get_code_block(text)
    
    ## Создаем разметку с кнопкой сброса
    data = {
            "chat_id": chat_id,
            "reply_to_message_id": message_id, 
            "parse_mode": "Markdown",   
    }
    if menu_mod: 
        reply_markup = {
            "inline_keyboard": [
                [{"text":  "👤🎤 Ваше сообщение", "callback_data": "gen_voice_0"},
                 {"text":  "🤖🎤 Сообщение бота", "callback_data": "gen_voice_1"}],
                [{"text":  "👤🖼️ Ваше сообщение", "callback_data": "gen_image_0"},
                 {"text":  "🤖🖼️ Сообщение бота", "callback_data": "gen_image_1"}],                 
                [{"text": "🔄 Сбросить диалог", "callback_data": "reset"}]
            ]
        }
        data["reply_markup"] = json.dumps(reply_markup)
    
    if menu_mod == "voice":
        reply_markup = {
            "inline_keyboard": [
                [{"text": "🔄 Сбросить диалог", "callback_data": "reset"}]
            ]
        }    
        data["reply_markup"] = json.dumps(reply_markup)
    
    if menu_mod == "stop_typing_action":
        reply_markup = {
            "inline_keyboard": [
                [{"text": "🔄 Сбросить диалог", "callback_data": "reset"}]
            ]
        }    
        data["reply_markup"] = json.dumps(reply_markup)
    
    if menu_mod == "process_voice_response":
        reply_markup = {
            "inline_keyboard": [
                [{"text":  "🤖🎤 Генерация голосом сообщение бота", "callback_data": "gen_voice_1"}],
                [{"text":  "👤🖼️ Ваше сообщение", "callback_data": "gen_image_0"},
                 {"text":  "🤖🖼️ Сообщение бота", "callback_data": "gen_image_1"}],                             
                [{"text": "🔄 Сбросить диалог", "callback_data": "reset"}]
            ]
        }    
        data["reply_markup"] = json.dumps(reply_markup)        
    
    if len(text) <= 4096:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data["text"] = text
#        data = {
#            "chat_id": chat_id,
#            "text": text,
#            "reply_to_message_id": message_id,
#            "parse_mode": "Markdown",
#            "reply_markup": json.dumps(reply_markup)
#        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    logger.error(f"Failed to send message. Status code: {response.status}, Response: {await response.text()}")
    else:
        ## Отправляем начало сообщения с кнопкой
        pre_text = text[:code_start] if 0 < code_start < 4096 else text[:50]
        await send_message(chat_id, message_id, f"Ответ слишком большой: {pre_text}...", typing_tasks)
        
        ## Отправляем файл
        file = StringIO(text)
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        data = aiohttp.FormData()
        data.add_field('chat_id', str(chat_id))
        data.add_field('document', file, filename='response.txt')
        data.add_field('reply_markup', json.dumps(reply_markup))
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status != 200:
                    logger.error(f"Failed to send document. Status code: {response.status}, Response: {await response.text()}")

    unique_key = f"{chat_id}:{message_id}"
    ## завершение отображение печати
    if unique_key in typing_tasks:
        typing_task = typing_tasks[unique_key]
        del typing_tasks[unique_key]
        if not typing_task.done():
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        
async def send_voice(chat_id, message_id, audio_content, user_id):
    ## Создаем разметку с кнопкой сброса
#    reply_markup = {
#        "inline_keyboard": [
#            [{"text": "📹 Создать видео", "callback_data": "gen_video"}]
#        ]
#    }
    daily_, monthly_ = get_user_word_counts(user_id)
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVoice"
    data = aiohttp.FormData()
    data.add_field('chat_id', str(chat_id))
    data.add_field('reply_to_message_id', str(message_id))
    data.add_field('voice', audio_content, filename='voice.ogg', content_type='audio/ogg')
    #data.add_field('reply_markup', json.dumps(reply_markup))
    data.add_field('caption', f"Сгенерировано {daily_} слов из доступных {DAILY_WORD_LIMIT} сегодня. Загрузите изображение с лицом человекоподобного существа для анимации")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            if response.status != 200:
                logger.error(f"Failed to send voice message. Status code: {response.status}")

async def edit_buttons(chat_id, message_id, buttons):
    buttons = {
        "inline_keyboard": [
#            [{"text": "✅ Видео создано", "callback_data": "video_completed"}],
#            [{"text": "🔄 Создать другое видео", "callback_data": "gen_video_again"}]
#            [{"text": "📼 Сгенерированный голос", "callback_data": "select_voice"}, 
#             {"text": "🎤 Голос пользователя", "callback_data": "select_voice"}],
#            [{"text": "🖼️ Выбрать изображение", "callback_data": "select_image"}]
            [{"text": "❌ Закрыть", "callback_data": "menu_close"}],
        ]
    }    
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageReplyMarkup"
    data = {
        "chat_id": chat_id,
        "message_id": message_id,
        "reply_markup": json.dumps(buttons),
        "caption": "Загрузите"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status != 200:
                logger.error(f"Failed to edit message. Status code: {response.status}")

async def menu_close(chat_id, message_id):
    buttons = {
        "inline_keyboard": [
            [{"text": "📹 Создать видео", "callback_data": "gen_video"}],
        ]
    }    
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageReplyMarkup"
    data = {
        "chat_id": chat_id,
        "message_id": message_id,
        "reply_markup": json.dumps(buttons),
        
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status != 200:
                logger.error(f"Failed to edit message. Status code: {response.status}")

async def edit_message(chat_id, message_id, new_text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText"
    data = {
        "chat_id": chat_id,
        "message_id": message_id,
        "parse_mode": "Markdown",
        "text": new_text
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status != 200:
                logger.error(f"Failed to edit message. Status code: {response.status}")


async def delete_message(chat_id, message_id):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage"
    data = {
        "chat_id": chat_id,
        "message_id": message_id
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status != 200:
                logger.error(f"Failed to delete message. Status code: {response.status}")

async def send_status_message(chat_id, reply_to_message_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "reply_to_message_id": reply_to_message_id
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status == 200:
                return (await response.json())['result']['message_id']
            else:
                logger.error(f"Failed to send status message. Status code: {response.status}")
                return None

status_messages = {}  # Словарь для хранения ID сообщений статуса
async def update_status_message(chat_id, message_id, status):
    if (chat_id, message_id) in status_messages:
        status_message_id = status_messages[(chat_id, message_id)]
        await edit_message(chat_id, status_message_id, status)
    else:
        status_message_id = await send_status_message(chat_id, message_id, status)
        status_messages[(chat_id, message_id)] = status_message_id

async def delete_status_message(chat_id, message_id):
    if (chat_id, message_id) in status_messages:
        status_message_id = status_messages[(chat_id, message_id)]
        await delete_message(chat_id, status_message_id)
        del status_messages[(chat_id, message_id)]

async def process_responses(receiver, send_message_func):
    while True:
        try:
            response = await receiver.recv()
            response = decompress(response)
            chat_id = response['chat_id']
            message_id = response['message_id']
            print ("PROCESS_RESPONSES------>", response['type'])
            if response['type'] == 'status_update':
                await update_status_message(chat_id, message_id, response['status'])
            elif response['type'] == 'video_gen_done':
                await delete_status_message(chat_id, message_id)
            elif response['type'] == 'stop_typing_action':
                processed_text = response['text']
                await send_message_func(chat_id, message_id, processed_text, menu_mod="stop_typing_action")
                await delete_status_message(chat_id, message_id)
                
#-------------------------------------------------                
            elif response['type'] == 'text':
                processed_text = response['text']
                input_type = response.get('type', 'text')
                
                save_message_to_db(chat_id, processed_text, "assistant")
                cache_response(processed_text, processed_text)
                
                await send_message_func(chat_id, message_id, processed_text, menu_mod=True)
                await delete_status_message(chat_id, message_id)

            elif response['type'] == 'voice':
                processed_text = response['text']
                input_type = response.get('type', 'text')
                
                save_message_to_db(chat_id, processed_text, "assistant")
                cache_response(processed_text, processed_text)
                
                await send_message_func(chat_id, message_id, processed_text, menu_mod="voice")
                await delete_status_message(chat_id, message_id)            
                audio = response['audio']
                print ("--------------VOICE!!!")
                await send_voice(chat_id, message_id, audio, response['user_id'])     
            elif response['type'] == 'process_voice':
                processed_text = response['text']
                await send_message_func(chat_id, message_id, processed_text, menu_mod=False)
                await delete_status_message(chat_id, message_id)                
            elif response['type'] == 'process_voice_response':
                processed_text = response['text']
                await send_message_func(chat_id, message_id, processed_text, menu_mod="process_voice_response")
                await delete_status_message(chat_id, message_id)                              
            elif response['type'] == 'gen_voice':
                await delete_status_message(chat_id, message_id)
                audio = response['audio']
                print ("--------------GEN VOICE!!!")
                await send_voice(chat_id, message_id, audio, response['user_id']) 
                ## завершение отображение печати
                unique_key = f"{chat_id}:{message_id}"
                if unique_key in typing_tasks:
                    typing_task = typing_tasks[unique_key]
                    del typing_tasks[unique_key]
                    if not typing_task.done():
                        typing_task.cancel()
                        try:
                            await typing_task
                        except asyncio.CancelledError:
                            pass 
            elif response['type'] == 'gen_image':    
                print ("GEN_IMAGE---------------------->>>>>>>>>>>>>")                        
                            
                            
                                
            elif response['type'] == 'stop_typing_action_':
                ## завершение отображение печати
                unique_key = f"{chat_id}:{message_id}"
                if unique_key in typing_tasks:
                    typing_task = typing_tasks[unique_key]
                    del typing_tasks[unique_key]
                    if not typing_task.done():
                        typing_task.cancel()
                        try:
                            await typing_task
                        except asyncio.CancelledError:
                            pass
#-------------------------------------------------                
                
#            else:
#                processed_text = response['text']
#                input_type = response.get('type', 'text')
#                
#                save_message_to_db(chat_id, processed_text, "assistant")
#                cache_response(processed_text, processed_text)
#                
#                await send_message_func(chat_id, message_id, processed_text, menu_mod=True)
#                await delete_status_message(chat_id, message_id)

#                if input_type == 'voice':
#                    audio = response['audio']
#                    print ("--------------VOICE!!!")
#                    await send_voice(chat_id, message_id, audio)


        except Exception as e:
            logger.error(f"Error processing response: {e}")
        
        await asyncio.sleep(0.1)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    # Запуск сервера
    Process(target=start_pipeline_worker).start()
    
    zmq_context = Context.instance()
    sender = zmq_context.socket(zmq.PUSH)
    sender.connect(ZMQ_PIPELINE_ADDRESS)
    
    receiver = zmq_context.socket(zmq.PULL)
    receiver.connect(ZMQ_RESULT_ADDRESS)
    
    typing_tasks = {}
    
    application = tornado.web.Application([
        (r'/', MessageHandler, dict(
            sender=sender, 
            send_message_func=lambda chat_id, message_id, text, menu_mod=True: send_message(chat_id, message_id, text, typing_tasks, menu_mod=menu_mod),
            typing_tasks=typing_tasks
        )),
        (r'/video_callback', VideoCallbackHandler, dict(sender=sender)),
    ])
    
    http_server = tornado.httpserver.HTTPServer(
        application,
        ssl_options={
            "certfile": "YOURPUBLIC.pem",
            "keyfile": "YOURPRIVATE.key",
            "ssl_version": ssl.PROTOCOL_TLSv1_2
        }
    )
    
    http_server.listen(8443)
    logger.info("Server started on port 8443")
    
#    tornado.ioloop.IOLoop.current().start()
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.add_callback(process_responses, receiver, lambda chat_id, message_id, text, menu_mod=True: send_message(chat_id, message_id, text, typing_tasks, menu_mod=menu_mod))
    io_loop.start()
