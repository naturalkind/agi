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
from datetime import datetime
from io import StringIO
import aiohttp
import asyncio
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
import gc

# Инициализация устройства
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    for chunk in chunks:
        outputs = model.synthesize(
            chunk,
            config,
            speaker_wav=audio_path,
            language="ru",
        )
        wav_chunks.append(outputs["wav"])
    wav_path = f"data_users/clon_out.wav"
    sf.write(wav_path, np.concatenate(wav_chunks), samplerate=config.audio.output_sample_rate)
    return wav_path

async def pipeline_worker():
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    context = zmq.asyncio.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(ZMQ_PIPELINE_ADDRESS)
    sender = context.socket(zmq.PUSH)
    sender.bind(ZMQ_RESULT_ADDRESS)

    # Инициализация моделей с IPEX
    model_path = "microsoft/Phi-3-mini-4k-instruct"
    with xpu_memory_scope():
        model = AutoModelForCausalLM.from_pretrained( 
            model_path,
            trust_remote_code=True,
            use_cache=True,
            attn_implementation='eager',
        )
        #model = model.to(device)
        model = model.to("xpu:0")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        #model = ipex.optimize(model, dtype=torch.bfloat16)
        #tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        pipe = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device="xpu:0"
        )

        generation_args = { 
            "max_new_tokens": 250,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        print ("---------------------->")
        # Инициализация Whisper
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_ID,
            torch_dtype=torch.bfloat16,
            #low_cpu_mem_usage=True,
            use_safetensors=True
        )
        whisper_model.to("xpu:1")
        whisper_model = ipex.optimize(whisper_model, dtype=torch.bfloat16)
        
        whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
        
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=whisper_processor.tokenizer,
            feature_extractor=whisper_processor.feature_extractor,
            torch_dtype=torch.bfloat16,
            device="xpu:1",
            chunk_length_s=30,
            batch_size=16, 
        )
        print ("----------------------> 2")
        # Инициализация XTTS
        xtts_config = XttsConfig()
        xtts_config.load_json("./XTTS-v2/config.json")
        xtts_model = Xtts.init_from_config(xtts_config)
        xtts_model.load_checkpoint(xtts_config, checkpoint_dir="./XTTS-v2/", eval=True)
        xtts_model.to("xpu:1")
        
        print ("----------------------> 3")
        async def send_status_update(chat_id, message_id, status):
            await sender.send(compress({
                'chat_id': chat_id,
                'message_id': message_id,
                'status': status,
                'type': 'status_update'
            }))

        while True:
            try:
                message = decompress(await receiver.recv())
                chat_id = message['chat_id']
                user_id = message['user_id']
                message_id = message['message_id']
                message_type = message['type']

                if message_type == 'text':
                    text = message['text']
                    await send_status_update(chat_id, message_id, "🔤 Анализ текста...")
                elif message_type == 'voice':
                    audio_content = message['audio_content']
                    await send_status_update(chat_id, message_id, "🎙️ Обработка голоса...")
                    print (type(audio_content))
                    result = whisper_pipe(audio_content)
                    torch.xpu.empty_cache()
                    text = result["text"]
                    await send_status_update(chat_id, message_id, "🔤 Анализ текста...")
                elif message_type == 'file':
                    text = message['text']
                    await send_status_update(chat_id, message_id, "🔤 Анализ текста...")
                else:
                    continue

                cursor.execute('SELECT message, role FROM dialogs WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 5', (chat_id,))
                history = cursor.fetchall()
                history.reverse()
                messages = [{"role": role, "content": msg} for msg, role in history]
                messages.append({"role": "user", "content": text})
                print ("IN-----------------------------")
                await send_status_update(chat_id, message_id, "🧠 Генерация ответа...")
                output = pipe(messages, **generation_args)
                response = output[0]['generated_text']
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
                if message_type == 'voice':
                    await send_status_update(chat_id, message_id, "🔊 Синтез речи...")
                    output_path = synthesize_speech(response, xtts_model, xtts_config, user_id)
                    response = f"Перевод: {text} Ответ: {response}"
                    with open(output_path, 'rb') as audio_file:
                        audio_content = audio_file.read()
                    
                    await sender.send(compress({
                        'chat_id': chat_id,
                        'audio': audio_content,
                        'text': response,
                        'message_id': message_id,
                        'type': 'voice'
                    }))
                else:
                    await sender.send(compress({
                        'chat_id': chat_id,
                        'text': response,
                        'message_id': message_id,
                        'type': 'text'
                    }))
                print ("OUT---------------------", response, message_type, torch.xpu.empty_cache())
#                torch.xpu.empty_cache()
#                torch.xpu.reset_accumulated_memory_stats(device="xpu:0")
#                torch.xpu.reset_peak_memory_stats(device="xpu:0")
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


class MessageHandler(tornado.web.RequestHandler):
    def initialize(self, sender, send_message_func, typing_tasks):
        self.sender = sender
        self.send_message_func = send_message_func
        self.typing_tasks = typing_tasks
        self.task_monitor = asyncio.create_task(self.monitor_tasks())
        
    async def post(self):
        try:
            data = json.loads(self.request.body)
            print ("--INNN", data)
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
                    if 'voice' in message:
                        await self.handle_voice_message(chat_id, user_id, message_id, message['voice']['file_id'])
                    elif 'text' in message:
                        await self.handle_text_message(chat_id, user_id, message_id, message['text'])
                    else:
                        await self.send_message_func(chat_id, message_id, "Пожалуйста, перешлите текстовое или голосовое сообщение")
                    return

                if 'text' in message:
                    await self.handle_text_message(chat_id, user_id, message_id, message['text'])
                elif 'voice' in message:
                    await self.handle_voice_message(chat_id, user_id, message_id, message['voice']['file_id'])
                elif 'document' in message:
                    await self.handle_document_message(chat_id, user_id, message_id, message['document'])
                else:
                    await self.send_message_func(chat_id, message_id, "Пожалуйста, отправьте текстовое сообщение, голосовое сообщение или текстовый файл")         
                               
            else:
                callback_query = data.get('callback_query', {})
                chat_id = callback_query.get('message', {}).get('chat', {}).get('id')
                message_id = callback_query.get('message', {}).get('message_id')
                user_id = callback_query.get('from', {}).get('id')
                data = callback_query.get('data')

                if data == 'about':
                    await self.send_about_message(chat_id, message_id)
                elif data == 'features':
                    await self.send_features_message(chat_id, message_id)
                elif data == 'reset':
                    await self.send_reset_message(chat_id, message_id)
                elif data == 'help' or data == 'main_menu':
                    await self.send_help_message(chat_id, message_id)
                elif data == 'main_menu':
                    await self.send_start_menu(chat_id, message_id)
                    
                ## Обязательно отправляем ответ на callback-запрос
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery"
                async with aiohttp.ClientSession() as session:
                    await session.post(url, json={
                        "callback_query_id": callback_query.get('id')
                    })
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.stop_typing_action(unique_key)

    async def handle_text_message(self, chat_id, user_id, message_id, text):
        logger.info(f"Received message from user {user_id} in chat {chat_id}: {text[:50]}...")

        if text.lower() == '/start':
            await self.send_start_menu(chat_id, message_id)
        elif text.lower() == '/help':
            await self.send_help_message(chat_id, message_id)
        elif text.lower() == '/reset':
            response = reset_dialog(chat_id)
            await self.send_message_func(chat_id, message_id, response)
        else:
            cached_response = get_cached_response(text)
            if cached_response:
                await self.send_message_func(chat_id, message_id, cached_response.decode('utf-8'))
            else:
                await self.start_typing_action(message_id, chat_id)
                await self.sender.send(compress({
                    'chat_id': chat_id,
                    'user_id': user_id,
                    'text': text,
                    'message_id': message_id,
                    'type': 'text'
                }))

    async def send_start_menu(self, chat_id, message_id):
        ## Path to your local image file
        image_path = 'robots-AI.jpg'  ## Replace with your actual image path
        
        try:
            ## Read the entire file content first
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            ## Create form data for the request
            data = aiohttp.FormData()
            data.add_field('chat_id', str(chat_id))
            data.add_field('photo', image_data, filename='logo.jpg', 
                          content_type='image/jpeg')
            data.add_field('caption', "Привет! Я AI-ассистент. Выберите действие:")
            data.add_field('reply_to_message_id', str(message_id))
            data.add_field('reply_markup', json.dumps({
                "inline_keyboard": [
                    [
                        {"text": "🤖 О боте", "callback_data": "about"},
                        {"text": "💬 Возможности", "callback_data": "features"}
                    ],
                    [
                        {"text": "🔧 Сбросить диалог", "callback_data": "reset"},
                        {"text": "❓ Справка", "callback_data": "help"}
                    ]
                ]
            }))
            
            ## Send request to Telegram API
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Error sending image: {error_text}")
                    return await response.json()
                    
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

    async def update_menu_message(self, chat_id, message_id, new_text, show_back_button=True):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText"
        keyboard = [
            [{"text": "🏠 Главное меню", "callback_data": "main_menu"}]
        ] if show_back_button else [
            [
                {"text": "🤖 О боте", "callback_data": "about"},
                {"text": "💬 Возможности", "callback_data": "features"}
            ],
            [
                {"text": "🔧 Сбросить диалог", "callback_data": "reset"},
                {"text": "❓ Справка", "callback_data": "help"}
            ]
        ]
        
        data = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": new_text,
            "parse_mode": "Markdown",
            "reply_markup": json.dumps({
                "inline_keyboard": keyboard
            })
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=data)

    async def handle_callback_query(self, callback_query):
        chat_id = callback_query['message']['chat']['id']
        message_id = callback_query['message']['message_id']
        data = callback_query['data']
        
        menu_texts = {
            "about": """
    🤖 *О боте*

    Привет! Я AI-ассистент, созданный на основе современных языковых моделей. 
    Моя цель - помогать пользователям решать различные задачи: от генерации текста 
    до анализа документов и кода.

    *Технологии*:
    - Phi-3.5-mini LLM
    - Whisper для распознавания речи
    - XTTS для синтеза голоса
    """,
            "features": """
    💬 *Возможности*

    - Общение на различные темы
    - Помощь в написании кода
    - Анализ текстовых документов
    - Распознавание голосовых сообщений
    - Генерация текста
    - Ответы на вопросы
    """,
            "help": """
    📘 *Справка по боту*

    - Отправьте текстовое сообщение для общения
    - Отправьте голосовое сообщение, бот ответит голосом спросившего
    - Поддерживается работа с текстовыми файлами (.txt, .py, .h, .cpp)

    *Команды*:
    - /start - Перезапуск бота
    - /help - Показать справку
    - /reset - Сбросить текущий диалог
    """,
            "main_menu": "Привет! Я AI-ассистент. Выберите действие:"
        }
        
        if data == "reset":
            reset_response = reset_dialog(chat_id)
            await self.update_menu_message(chat_id, message_id, reset_response)
        elif data in menu_texts:
            await self.update_menu_message(
                chat_id, 
                message_id, 
                menu_texts[data],
                show_back_button=(data != "main_menu")
            )
        
        ## Answer callback query
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery"
        async with aiohttp.ClientSession() as session:
            await session.post(url, json={"callback_query_id": callback_query['id']})


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
            await self.send_message_func(chat_id, message_id, "Пожалуйста, отправьте текстовый файл (.txt, .py, .h, .cpp)")

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


    async def send_about_message(self, chat_id, message_id):
        about_text = """
        🤖 *О боте*:

        Привет! Я AI-ассистент, созданный на основе современных языковых моделей. 
        Моя цель - помогать пользователям решать различные задачи: от генерации текста 
        до анализа документов и кода.

        *Технологии*:
        - Phi-3.5-mini LLM
        - Whisper для распознавания речи
        - XTTS для синтеза голоса
        """
        await self.send_menu_message(chat_id, message_id, about_text)

    async def send_features_message(self, chat_id, message_id):
        features_text = """
        💬 *Возможности*:

        - Общение на различные темы
        - Помощь в написании кода
        - Анализ текстовых документов
        - Распознавание голосовых сообщений
        - Генерация текста
        - Ответы на вопросы
        """
        await self.send_menu_message(chat_id, message_id, features_text)

    async def send_reset_message(self, chat_id, message_id):
        reset_response = reset_dialog(chat_id)
        await self.send_menu_message(chat_id, message_id, reset_response)

    async def send_help_message(self, chat_id, message_id):
        help_text = """
        📘 *Справка по боту*:

        - Отправьте текстовое сообщение для общения
        - Отправьте голосовое сообщение, бот ответить голосом спросившего
        - Поддерживается работа с текстовыми файлами (.txt, .py, .h, .cpp)

        *Команды*:
        - /start - Перезапуск бота
        - /help - Показать справку
        - /reset - Сбросить текущий диалог
        """
        await self.send_menu_message(chat_id, message_id, help_text)

    async def send_menu_message(self, chat_id, message_id, text):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_to_message_id": message_id,
            "reply_markup": json.dumps({
                "inline_keyboard": [
                    [
                        {"text": "🏠 Главное меню", "callback_data": "main_menu"}
                    ]
                ]
            })
        }
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=data)


#async def send_message(chat_id, message_id, text, typing_tasks):
#    code_block, code_start = get_code_block(text)
#    if len(text) <= 4096:
#        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
#        data = {
#            "chat_id": chat_id,
#            "text": text,
#            "reply_to_message_id": message_id,
#            "parse_mode": "Markdown"
#        }
#        async with aiohttp.ClientSession() as session:
#            async with session.post(url, json=data) as response:
#                if response.status != 200:
#                    logger.error(f"Failed to send message. Status code: {response.status}, Response: {await response.text()}")
#    else:
#        pre_text = text[:code_start] if 0 < code_start < 4096 else text[:50]
#        await send_message(chat_id, message_id, f"Ответ слишком большой: {pre_text}...", typing_tasks)
#        
#        file = StringIO(text)
#        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
#        data = aiohttp.FormData()
#        data.add_field('chat_id', str(chat_id))
#        data.add_field('document', file, filename='response.txt')
#        async with aiohttp.ClientSession() as session:
#            async with session.post(url, data=data) as response:
#                if response.status != 200:
#                    logger.error(f"Failed to send document. Status code: {response.status}, Response: {await response.text()}")
#    unique_key = f"{chat_id}:{message_id}"
#    # завершение отображение печати
#    if unique_key in typing_tasks:
#        typing_task = typing_tasks[unique_key]
#        del typing_tasks[unique_key]
#        if not typing_task.done():
#            typing_task.cancel()
#            try:
#                await typing_task
#            except asyncio.CancelledError:
#                pass

async def send_message(chat_id, message_id, text, typing_tasks):

    print ("-------------------->")
    code_block, code_start = get_code_block(text)
    
    ## Создаем разметку с кнопкой сброса
    reply_markup = {
        "inline_keyboard": [
            [{"text": "🔄 Сбросить диалог", "callback_data": "reset"}]
        ]
    }
    
    if len(text) <= 4096:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "reply_to_message_id": message_id,
            "parse_mode": "Markdown",
            "reply_markup": json.dumps(reply_markup)
        }
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

        
async def send_voice(chat_id, message_id, audio_content):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVoice"
    data = aiohttp.FormData()
    data.add_field('chat_id', str(chat_id))
    data.add_field('reply_to_message_id', str(message_id))
    data.add_field('voice', audio_content, filename='voice.ogg', content_type='audio/ogg')

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=data) as response:
            if response.status != 200:
                logger.error(f"Failed to send voice message. Status code: {response.status}")


async def edit_message(chat_id, message_id, new_text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText"
    data = {
        "chat_id": chat_id,
        "message_id": message_id,
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

async def process_responses(receiver, send_message_func):
    status_messages = {}  # Словарь для хранения ID сообщений статуса
    print ("--------->PROCESS_RESPONSES<--------------------")
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

    while True:
        try:
            response = await receiver.recv()
            response = decompress(response)
            chat_id = response['chat_id']
            message_id = response['message_id']
            
            if response['type'] == 'status_update':
                await update_status_message(chat_id, message_id, response['status'])
            else:
                processed_text = response['text']
                input_type = response.get('type', 'text')
                
                if input_type == 'voice':
                    audio = response['audio']
                    await send_voice(chat_id, message_id, audio)
                
                save_message_to_db(chat_id, processed_text, "assistant")
                cache_response(processed_text, processed_text)
                
                await send_message_func(chat_id, message_id, processed_text)
                await delete_status_message(chat_id, message_id)

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
            send_message_func=lambda chat_id, message_id, text: send_message(chat_id, message_id, text, typing_tasks),
            typing_tasks=typing_tasks
        )),
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
    io_loop.add_callback(process_responses, receiver, lambda chat_id, message_id, text: send_message(chat_id, message_id, text, typing_tasks))
    io_loop.start()
