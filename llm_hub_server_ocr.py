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
import uuid

# Kandensky
TOPIC = 'snaptravel'
RECEIVE_PORT = 5556      # Сервер получает задачи на этом порту
CLIENT_SEND_PORT = 5555  # Сервер отправляет результаты клиентам на этом порту

# Глобальные переменные для ZMQ
work_publisher = None
context = None


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
PHI3_SERVER_URL = "https://192.168.1.50:5000/generate"
# Конфигурация сервера OCR
OCR_SERVER_URL = "https://192.168.1.50:5001/ocr"
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

# INTEL VERSION PHI
async def llm_server(messages, generation_args={"max_new_tokens":400, "temperature":0.7}):
    # Создаем SSL контекст аналогично примеру download_video
    ssl_context = ssl.create_default_context(cafile='ssl/ca.crt')
    ssl_context.load_cert_chain('ssl/client.crt', 'ssl/client.key')
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    # Добавляем заголовки аналогично примеру
    headers = {"X-API-Key": "default-api-key-change-me"}
    
    request_data = {
        "messages": messages,
        "max_new_tokens": generation_args.get("max_new_tokens", 400),
        "temperature": generation_args.get("temperature", 0.7)
    }
    
    try:
        # Создаем ClientSession с SSL контекстом аналогично примеру
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_context)
        ) as session:
            async with session.post(
                PHI3_SERVER_URL,
                json=request_data,
                headers=headers,
                timeout=60*8
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Phi-3 server error: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Failed to query Phi-3 server: {str(e)}")
        return None

###################
#### OCR Server
###################

async def query_ocr_server(user_id, chat_id, message_id, file_id):
    """
    Отправляет изображение на OCR сервер для распознавания текста
    """
    try:
        # Получаем путь к изображению пользователя
        image_path = get_user_image_path(user_id, file_id)
        
        # Настраиваем SSL-контекст для защищенного соединения
        ssl_context = ssl.create_default_context(cafile='ssl/ca.crt')
        ssl_context.load_cert_chain('ssl/client.crt', 'ssl/client.key')
        
        # Формируем данные для отправки на сервер OCR
        data = aiohttp.FormData()
        # Добавляем изображение
        data.add_field('image', open(image_path, 'rb'), filename=f'ocr_image_{user_id}_{file_id}.jpg')
        
        # Добавляем параметры для OCR (опционально)
        ocr_params = {
            'prompt': '<image>\n<|grounding|>Convert the document to markdown. ',
            'save_results': False,
            'test_compress': True
        }
        data.add_field('params', json.dumps(ocr_params))
        
#        # Создаем сессию с настроенным SSL-контекстом
#        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
#            # Отправляем POST-запрос на сервер OCR
#            async with session.post(OCR_SERVER_URL, data=data) as resp:
#                # Если запрос успешен (статус 200)
#                if resp.status == 200:
#                    # Получаем результат в формате JSON
#                    result = await resp.json()
#                    # Извлекаем распознанный текст
#                    extracted_text = result.get('extracted_text', '')
#                    return extracted_text
#                else:
#                    logger.error(f"OCR server responded with status: {resp.status}")
#                    return None
                    
        # Создаем сессию с настроенным SSL-контекстом
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            # Отправляем POST-запрос на сервер генерации видео
            async with session.post(OCR_SERVER_URL, data=data) as resp:
                # Если запрос успешен (статус 200)
                if resp.status == 200:
                    # Получаем результат в формате JSON
                    result = await resp.json()
                    # Извлекаем ID задачи
                    task_id = result.get('task_id')
                    # Сохраняем связь задачи с чатом в Redis
                    redis_client.setex(
                        f"ocr_task:{task_id}",  # Ключ для хранения в Redis
                        3600*100,  # Время жизни ключа - 100 часов
                        json.dumps({  # Сохраняем данные в формате JSON
                            'chat_id': chat_id,
                            'message_id': message_id,
                            'user_id': user_id
                        })
                    )
                    # Возвращаем ID задачи
                    return task_id
                else:
                    # Если статус не 200, логируем предупреждение и повторяем попытку
                    logger.warning(f"Video server responded with status: {resp.status}, retrying in {retry_delay} seconds...")  
  
                    
    except Exception as e:
        logger.error(f"OCR task failed: {str(e)}")
        return None

#async def query_ocr_server(user_id, chat_id, message_id, file_id):
#    # Запускаем таймер для отслеживания общего времени выполнения
#    start_time = time.time()
#    # Максимальное время ожидания - 10 минут
#    timeout = 600  # 10 минут в секундах
#    # Начальная задержка между повторными попытками
#    retry_delay = 5  # Начинаем с 5 секунд между попытками
#    # Максимальная задержка между повторными попытками
#    max_retry_delay = 30  # Максимальная задержка между попытками
#    
#    # Выполняем попытки подключения, пока не истечет время ожидания
#    while time.time() - start_time < timeout:
#        try:
#            
#            # Получаем путь к изображению пользователя
#            image_path = get_user_image_path(user_id, file_id)
#            
#            # URL для обратного вызова, куда сервер отправит результат после обработки
#            callback_url = "https://192.168.1.50:8443/video_callback"  # Внешний URL для обратного вызова
#            
#            # Настраиваем SSL-контекст для защищенного соединения
#            ssl_context = ssl.create_default_context(cafile='ssl/ca.crt')
#            ssl_context.load_cert_chain('ssl/client.crt', 'ssl/client.key')
#            
#            # Формируем данные для отправки на сервер
#            data = aiohttp.FormData()
#            # Добавляем аудиофайл
#            data.add_field('audio', open(audio_path, 'rb'), filename='audio.wav')
#            # Добавляем изображение
#            data.add_field('image', open(image_path, 'rb'), filename=f'speaker_reference_{user_id}_{file_id}.jpg')
#            # Добавляем параметры для генерации видео
#            data.add_field('video_params', json.dumps({"pose_weight": 1.0}))
#            # Добавляем URL для обратного вызова
#            data.add_field('callback_url', callback_url)
#            
#            # Создаем сессию с настроенным SSL-контекстом
#            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
#                # Отправляем POST-запрос на сервер генерации видео
#                async with session.post(OCR_SERVER_URL, data=data) as resp:
#                    # Если запрос успешен (статус 200)
#                    if resp.status == 200:
#                        # Получаем результат в формате JSON
#                        result = await resp.json()
#                        # Извлекаем ID задачи
#                        task_id = result.get('task_id')
#                        # Сохраняем связь задачи с чатом в Redis
#                        redis_client.setex(
#                            f"ocr_task:{task_id}",  # Ключ для хранения в Redis
#                            3600*100,  # Время жизни ключа - 100 часов
#                            json.dumps({  # Сохраняем данные в формате JSON
#                                'chat_id': chat_id,
#                                'message_id': message_id,
#                                'user_id': user_id
#                            })
#                        )
#                        # Возвращаем ID задачи
#                        return task_id
#                    else:
#                        # Если статус не 200, логируем предупреждение и повторяем попытку
#                        logger.warning(f"Video server responded with status: {resp.status}, retrying in {retry_delay} seconds...")
#                        
#        except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
#            # Обрабатываем ошибки соединения
#            logger.warning(f"Connection error: {str(e)}, retrying in {retry_delay} seconds...")
#        except Exception as e:
#            # Обрабатываем другие ошибки
#            logger.error(f"Video task creation failed: {str(e)}")
#            # Для не связанных с подключением ошибок не повторяем попытки
#            return None
#            
#        # Ожидаем перед повторной попыткой
#        await asyncio.sleep(retry_delay)
#        # Реализуем экспоненциальную задержку (увеличиваем время между повторными попытками)
#        retry_delay = min(retry_delay * 1.5, max_retry_delay)
#    
#    # Если исчерпали все попытки повторного подключения
#    logger.error(f"Failed to connect to video server after trying for {timeout} seconds")
#    return None


###################
#### END OCR Server
###################

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
        
        async def send_status_update(chat_id, message_id, status):
            await sender.send(compress({
                'chat_id': chat_id,
                'message_id': message_id,
                'status': status,
                'type': 'status_update'
            }))
            
        # Функция для обработки голосового ответа
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
                elif message_type.startswith(('gen_voice')):
                    # Обрабатываем gen_voice отдельно - здесь уже есть текст для синтеза
                    text = message['text']
                    response = message['text']
                elif message_type.startswith(('gen_image')):
                    # Обрабатываем gen_voice отдельно - здесь уже есть текст для синтеза
                    text = message['text'] 
                    await send_status_update(chat_id, message_id, "🖼️ Генерация изображения...")  
                    output_image = await query_image_server_simple(text, chat_id, message_id)
                    await send_gen_image(chat_id, message_id, output_image['images'])
                    await sender.send(compress({
                        'chat_id': chat_id,
                        'message_id': message_id,
                        'type': 'stop_typing_action',
                        'text': text
                    }))
                elif message_type == 'ocr':
                    # Обработка OCR запроса
                    file_id = message['file_id']
                    #await send_status_update(chat_id, message_id, "📖 Распознавание текста...")
                    extracted_text = await query_ocr_server(message['user_id'], chat_id, message_id, file_id)
                    print ("бработка OCR запроса----->", extracted_text)
                    if extracted_text:
#                        await sender.send(compress({
#                            'chat_id': chat_id,
#                            'text': f"📖 Распознанный текст:\n\n{extracted_text}",
#                            'message_id': message_id,
#                            'type': 'text'
#                        }))
                        await sender.send(compress({
                            'chat_id': chat_id,
                            'text': f"📖 Распознавание текста...\n\nID задачи: {extracted_text}",
                            'message_id': message_id,
                            'type': 'text'
                        }))
                    else:
                        await sender.send(compress({
                            'chat_id': chat_id,
                            'text': "❌ Не удалось распознать текст",
                            'message_id': message_id,
                            'type': 'text'
                        }))
                    continue
                else:
                    logger.warning(f"Неизвестный тип сообщения: {message_type}")
                    continue
                    
                # Логируем информацию только если есть текст
                if text is not None:
                    logger.info(f"PIPELINE_WORKER--------------2>{message_type}, {text}")
                
                # Генерация текстового ответа если нужно
                if message_type not in ['status_update_video', 'gen_voice', 'gen_image', 'ocr'] and text is not None:
                    # Генерация текста
                    cursor.execute('SELECT message, role FROM dialogs WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 5', (chat_id,))
                    history = cursor.fetchall()
                    history.reverse()
                    messages = [{"role": role, "content": msg} for msg, role in history]
                    messages.append({"role": "user", "content": text})
                    await send_status_update(chat_id, message_id, "🧠 Генерация ответа...")
                    
                    output = await llm_server(messages) 
                    if output is None:
                        response = "⚠️ Ошибка при обработке запроса. Попробуйте позже."
                    else:
                        response = output.get("response", "Не удалось получить ответ")
                
                # Обработка ответа в зависимости от типа сообщения
                if message_type == 'voice':
                    if response:
                        await process_voice_response(response, message_type, is_voice_input=True)
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
            print (data)
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
                        # Обработка фото для OCR
                        telegram_photo = message['photo'][-1]  # Берем самое качественное изображение
                        await self.download_image(telegram_photo['file_id'], user_id)
                        print ("ИЗОБРАЖЕНИЕ ДЛЯ OCR!!!!>", message['photo'], telegram_photo, telegram_photo['file_id'])
                        
                        # Отправляем задачу на OCR распознавание
                        await self.start_typing_action(message_id, chat_id)
                        await self.sender.send(compress({
                            'chat_id': chat_id,
                            'user_id': user_id,
                            'file_id': telegram_photo['file_id'],
                            'message_id': message_id,
                            'type': 'ocr'
                        }))
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
                    # Обработка фото для OCR
                    telegram_photo = message['photo'][-1]  # Берем самое качественное изображение
                    await self.download_image(telegram_photo['file_id'], user_id)
                    print ("ИЗОБРАЖЕНИЕ ДЛЯ OCR!!!!", message['photo'], telegram_photo, telegram_photo['file_id'])
                    
                    # Отправляем задачу на OCR распознавание
                    await self.start_typing_action(message_id, chat_id)
                    await self.sender.send(compress({
                        'chat_id': chat_id,
                        'user_id': user_id,
                        'file_id': telegram_photo['file_id'],
                        'message_id': message_id,
                        'type': 'ocr'
                    }))

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
                elif data.startswith(('gen_voice')):
                    await self.start_typing_action(message_id, chat_id)
                    parts = data.split('_')[-1]
                    if int(parts) == 0:
                        text = callback_query.get('message', {}).get('reply_to_message', {}).get('text', {})
                    elif int(parts) == 1:
                        text = callback_query.get('message', {}).get('text', {})
                    # Отправить
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
                    
                    # Отправить на сервер генерации изображений
                    await sender.send(compress({
                        'chat_id': chat_id,
                        'user_id': user_id,
                        'text': text,
                        'message_id': message_id,
                        'type': 'gen_image'
                    }))                    
                    
                ## Обязательно отправляем ответ на callback-запрос
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery"
                async with aiohttp.ClientSession() as session:
                    await session.post(url, json={
                        "callback_query_id": callback_query.get('id')
                    })
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.stop_typing_action(unique_key)
    
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
        - Распознавание текста с изображений (OCR)           
        - Помощь в написании кода
        - Распознавание голосовых сообщений          
        - Общение
        - Генерация голоса        
        - Анализ текстовых документов
        
        *Технологии*:
        - Phi-3.5-mini языковая модель чат бот
        - Whisper распознавание речи
        - XTTS v2 синтез голоса
        - DeepSeek-OCR распознавание текста
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
        - Отправьте изображение с текстом для распознавания (OCR)
        - Поддерживается работа с текстовыми файлами (.txt, .py, .h, .cpp)

        *Команды*:
        - /start - Перезапуск бота
        - /help - Показать справку
        - /info - Информация о боте
        - /reset - Сбросить текущий диалог
        - /stats - Статистика доступной генерации

        *✅ Выбран голоса для генерации видео*: `{voice_mode}`
        """
        
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

# Остальные функции (send_message, send_voice, edit_buttons, menu_close, edit_message, delete_message, 
# send_status_message, update_status_message, delete_status_message, process_responses) остаются без изменений
# Альтернативная версия с использованием str.translate (более эффективная)
# Временные маркеры для сохранения разметки
import re
def escape_markdown_v2(text: str) -> str:
    """
    Экранирует специальные символы для MarkdownV2, но сохраняет преднамеренную разметку.
    
    Args:
        text (str): Исходный текст с разметкой
        
    Returns:
        str: Текст с корректно экранированными символами
    """
    text = text.replace("**", "")
    
    # Символы, которые нужно экранировать
    escape_chars = r'_*[]()~`>#+-=|{}.!@'
    
    # Сохраняем жирный текст **text**
    bold_pattern = r'\*\*(.*?)\*\*'
    bold_matches = list(re.finditer(bold_pattern, text))
#    
#    # Сохраняем упоминания @username
#    mention_pattern = r'(@\w+)'
#    mention_matches = list(re.finditer(mention_pattern, text))
    
    # Временные замены
    temp_bold_marker = "🄱🄾🄻🄳🄼🄰🅁🄺🄴🅁"
    temp_mention_marker = "🄼🄴🄽🅃🄸🄾🄽🄼🄰🅁🄺🄴🅁"
    
    # Заменяем жирный текст на временные маркеры
    bold_replacements = []
    for i, match in enumerate(bold_matches):
        original_text = match.group(1)
        bold_replacements.append(original_text)
        text = text.replace(match.group(0), f"{temp_bold_marker}{i}{temp_bold_marker}")
    
#    # Заменяем упоминания на временные маркеры
#    mention_replacements = []
#    for i, match in enumerate(mention_matches):
#        original_text = match.group(1)
#        mention_replacements.append(original_text)
#        text = text.replace(match.group(0), f"{temp_mention_marker}{i}{temp_mention_marker}")
    
    # Экранируем весь текст
    escaped_text = ''
    for char in text:
        if char in escape_chars:
            escaped_text += '\\' + char
        else:
            escaped_text += char
    
    # Восстанавливаем жирный текст
    for i, original_bold in enumerate(bold_replacements):
        # Экранируем только внутренности жирного текста (но не звездочки)
        escaped_bold = ''
        for char in original_bold:
            if char in escape_chars:
                escaped_bold += '\\' + char
            else:
                escaped_bold += char
        escaped_text = escaped_text.replace(f"{temp_bold_marker}{i}{temp_bold_marker}", f"**{escaped_bold}**")
    
#    # Восстанавливаем упоминания
#    for i, original_mention in enumerate(mention_replacements):
#        escaped_text = escaped_text.replace(f"{temp_mention_marker}{i}{temp_mention_marker}", original_mention)
    
    return escaped_text

async def send_message(chat_id, message_id, text, typing_tasks, menu_mod):
    print ("MENU_MOD =====>", menu_mod)
    code_block, code_start = get_code_block(text)
    
    ## Создаем разметку с кнопкой сброса
    data = {
            "chat_id": chat_id,
            "reply_to_message_id": message_id, 
            "parse_mode": "Markdown",   # "MarkdownV2"
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
        print ("TEXT SEND TO TG !!!!!!!!!!!", escape_markdown_v2(text))
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        #data["text"] = text
        data = {
            "chat_id": chat_id,
            "text": escape_markdown_v2(text),
            "reply_to_message_id": message_id,
            "parse_mode": "MarkdownV2", 
            "reply_markup": json.dumps(reply_markup)
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    logger.error(f"Failed to send message. Status code: {response.status}, Response: {await response.text()}")
    else:
        ## Отправляем начало сообщения с кнопкой
        pre_text = text[:code_start] if 0 < code_start < 4096 else text[:50]
        await send_message(chat_id, message_id, f"Ответ слишком большой: {pre_text}...", typing_tasks, menu_mod=True)
        
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

class UnifiedCallbackHandler:
    """Универсальный обработчик callback-ов для всех сервисов"""
    
    def __init__(self, sender):
        self.sender = sender
        self.callback_handlers = {
            'video': self._handle_video_callback,
            'ocr': self._handle_ocr_callback
        }
    
    async def handle_callback(self, callback_type: str, data: dict):
        """Основной метод обработки callback-ов"""
        handler = self.callback_handlers.get(callback_type)
        if handler:
            await handler(data)
        else:
            logger.warning(f"Unknown callback type: {callback_type}")
    
    async def _handle_video_callback(self, data: dict):
        """Обработчик видео callback-ов"""
        try:
            task_id = data['task_id']
            status = data['status']
            
            task_data = redis_client.get(f"video_task:{task_id}")
            if not task_data:
                return
                
            task_data = json.loads(task_data)
            chat_id = task_data['chat_id']
            message_id = task_data['message_id']
            user_id = task_data['user_id']
            
            if status == 'completed':
                video_url = urljoin('https://192.168.1.50:6000/', data['download_url'])
                await self._send_video_to_telegram(chat_id, message_id, video_url)
                await self._cleanup_task(task_id, chat_id, message_id, user_id)
                update_gen_counts(user_id)
            else:
                await update_status_message(chat_id, message_id, "Ошибка генерации видео")
                await self._cleanup_task(task_id, chat_id, message_id, user_id)
                
        except Exception as e:
            logger.error(f"Video callback error: {str(e)}")

    async def _send_ocr_result(self, chat_id, message_id, extracted_text):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        inline_keyboard = [[{"text": "📘 Инструкция", "callback_data": "help"}]]
        data = {
            "chat_id": chat_id,
            "text": extracted_text,
            "parse_mode": "Markdown",
            "reply_to_message_id": message_id,
            "reply_markup": json.dumps({
                "inline_keyboard": inline_keyboard
            })
        }
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=data)    
    async def _handle_ocr_callback(self, data: dict):
        """Обработчик OCR callback-ов"""
        print ("!!!!!!!!!!!!!!!!!!!!!!Обработчик OCR callback-ов")
        try:
            task_id = data['task_id']
            status = data['status']
            
            task_data = redis_client.get(f"ocr_task:{task_id}")
            print ("!!!!!!!!!!!!!!!!!!!!!!Обработчик OCR callback-ов", status, task_data)
            if not task_data:
                return
                
            task_data = json.loads(task_data)
            chat_id = task_data['chat_id']
            message_id = task_data['message_id']
            user_id = task_data['user_id']
            
            if status == 'completed':
                extracted_text = data.get('extracted_text', '')
                print ("!------------!", task_data, extracted_text) 
                await self._send_ocr_result(chat_id, message_id, extracted_text)
                await self._cleanup_task(task_id, chat_id, message_id, user_id, task_type='ocr')
            else:
                await update_status_message(chat_id, message_id, "Ошибка распознавания текста")
                await self._cleanup_task(task_id, chat_id, message_id, user_id, task_type='ocr')
               
        except Exception as e:
            logger.error(f"OCR callback error: {str(e)}")
    
    async def _cleanup_task(self, task_id: str, chat_id: str, message_id: str, user_id: str, task_type: str = 'video'):
        """Универсальная очистка задач"""
        redis_key = f"{task_type}_task:{task_id}"
        redis_client.delete(redis_key)
        
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
        
        await delete_status_message(chat_id, message_id)


        
class UniversalCallbackHandler(tornado.web.RequestHandler):
    def initialize(self, callback_handler):
        self.callback_handler = callback_handler
    
    async def post(self):
        try:
            data = json.loads(self.request.body)
            callback_type = data.get('type', 'video')  # По умолчанию video для обратной совместимости
            
            print ("----------->", callback_type, data)
            
            await self.callback_handler.handle_callback(callback_type, data)
            self.set_status(200)
            self.write({"status": "ok"})
            
        except Exception as e:
            logger.error(f"Universal callback error: {str(e)}")
            self.set_status(500)
            self.write({"error": str(e)})



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
    
    # Создание универсального обработчика
    callback_handler = UnifiedCallbackHandler(sender)
    
    application = tornado.web.Application([
        (r'/', MessageHandler, dict(
            sender=sender, 
            send_message_func=lambda chat_id, message_id, text, menu_mod=True: send_message(chat_id, message_id, text, typing_tasks, menu_mod=menu_mod),
            typing_tasks=typing_tasks
        )),
        (r'/callback', UniversalCallbackHandler, dict(callback_handler=callback_handler)),  # Универсальный endpoint
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
    
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.add_callback(process_responses, receiver, lambda chat_id, message_id, text, menu_mod=True: send_message(chat_id, message_id, text, typing_tasks, menu_mod=menu_mod))
    io_loop.start()
