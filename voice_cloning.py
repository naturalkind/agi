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
from zmq.asyncio import Context
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline, 
    SpeechT5ForTextToSpeech, 
    SpeechT5HifiGan, 
    SpeechT5Processor
)
import torch
import redis
import sqlite3
from datetime import datetime
from io import StringIO, BytesIO
import aiohttp
import asyncio
import time
import os
import multiprocessing
from multiprocessing import Process, Queue
from pydub import AudioSegment
import numpy as np
import soundfile as sf
#multiprocessing.set_start_method('spawn')
#torch.multiprocessing.set_start_method('spawn')
# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Загрузка данных конфигурации из JSON файла
with open('config.bot', 'r') as json_file:
    data = json.load(json_file)

# Константы
BOT_TOKEN = data['BOT_TOKEN']
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
SQLITE_DB = 'dialogs.db'
ZMQ_PIPELINE_ADDRESS = "tcp://127.0.0.1:5555"
ZMQ_RESULT_ADDRESS = "tcp://127.0.0.1:5556"
WHISPER_MODEL_ID = "whisper-large-v3"
XTTS_MODEL_ID = "XTTS-v2"
VOICE_SAMPLES_DIR = "voice_samples"

# Инициализация Redis для кэширования
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# Инициализация SQLite для хранения диалогов
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

def compress(obj):
    return zlib.compress(pickle.dumps(obj))

def decompress(pickled):
    return pickle.loads(zlib.decompress(pickled))

def synthesize_speech(text, model, config):
    outputs = model.synthesize(
            text,
            config,
            speaker_wav="/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/TTS/audio_2024-10-10_22-47-02.wav",
            #gpt_cond_len=3,
            language="ru",
            )
    wav_path = "clon_out.wav"
    sf.write(wav_path, outputs["wav"], samplerate=config.audio.output_sample_rate)
    #print(f"Saved WAV file: {wav_path}", config.audio)
    return wav_path

def save_message_to_db(chat_id, message, role):
    cursor.execute('''
        INSERT INTO dialogs (chat_id, message, role)
        VALUES (?, ?, ?)
    ''', (chat_id, message, role))
    conn.commit()

def get_cached_response(question):
    return redis_client.get(question)

def cache_response(question, answer):
    redis_client.setex(question, 3600, answer)  # Кэшируем ответ на 1 час

def reset_dialog(chat_id):
    if chat_id in dialogs:
        del dialogs[chat_id]
    cursor.execute('DELETE FROM dialogs WHERE chat_id = ?', (chat_id,))
    conn.commit()
    return "Диалог сброшен. Начнем сначала!"

def pipeline_worker():
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(ZMQ_PIPELINE_ADDRESS)

    sender = context.socket(zmq.PUSH)
    sender.bind(ZMQ_RESULT_ADDRESS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/Phi-3.5-mini-instruct/"
    
    model = AutoModelForCausalLM.from_pretrained( 
        model_path,  
        device_map="cuda",  
        torch_dtype="auto",  
        trust_remote_code=True,  
        attn_implementation="flash_attention_2"
    ) 
    tokenizer = AutoTokenizer.from_pretrained(model_path) 

    pipe = pipeline( 
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
    ) 

    generation_args = { 
        "max_new_tokens": 100000, 
        "return_full_text": False, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    # Initialize Whisper model for voice recognition
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)

    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        torch_dtype=torch.float16,
        device=device,
    )
    
    config_xtts = XttsConfig()
    config_xtts.load_json("/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/XTTS-v2/config.json")
    model_xtts = Xtts.init_from_config(config_xtts)
    model_xtts.load_checkpoint(config_xtts, checkpoint_dir="/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/XTTS-v2/", eval=True)
    model_xtts.cuda()

    while True:
        try:
            message = decompress(receiver.recv())
            chat_id = message['chat_id']
            message_id = message['message_id']
            message_type = message['type']
            
            if message_type == 'text':
                text = message['text']
            elif message_type == 'voice':
                audio_content = message['audio_content']
                result = whisper_pipe(audio_content)
                text = result["text"]
            elif message_type == 'file':
                text = message['text']
            elif message_type == 'tts':
                text = message['text']
                voice_sample = message['voice_sample']
            else:
                continue

            # Получаем историю диалога из базы данных
            cursor.execute('SELECT message, role FROM dialogs WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 10', (chat_id,))
            history = cursor.fetchall()
            history.reverse()
            messages = [{"role": role, "content": msg} for msg, role in history]
            messages.append({"role": "user", "content": text})

            output = pipe(messages, **generation_args)
            response = output[0]['generated_text']

            if message_type == 'tts':
                # Generate speech using XTTS
                output_path = synthesize_speech(response, xtts_model, xtts_config)
                
                with open(output_path, 'rb') as audio_file:
                    audio_content = audio_file.read()
                
                os.remove(output_path)  # Clean up temporary file
                
                sender.send(compress({
                    'chat_id': chat_id,
                    'audio': audio_content,
                    'text': response,
                    'message_id': message_id
                }))
            else:
                sender.send(compress({
                    'chat_id': chat_id,
                    'text': response,
                    'message_id': message_id
                }))

            # Очищаем кэш CUDA после каждой обработки
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in pipeline worker: {e}")

class MessageHandler(tornado.web.RequestHandler):
    def initialize(self, sender, send_message_func, typing_tasks):
        self.sender = sender
        self.send_message_func = send_message_func
        self.typing_tasks = typing_tasks
        
    async def post(self):
        try:
            data = json.loads(self.request.body)
            message = data['message']
            chat_id = message['chat']['id']
            message_id = message["message_id"]
            
            if 'text' in message:
                text = message['text']
                logger.info(f"Received message: {text[:50]}...")

                if text.lower().startswith('/tts'):
                    voice_sample = self.get_user_voice_sample(chat_id)
                    if voice_sample:
                        tts_text = text[5:].strip()
                        await self.sender.send(compress({
                            'chat_id': chat_id,
                            'text': tts_text,
                            'voice_sample': voice_sample,
                            'message_id': message_id,
                            'type': 'tts'
                        }))
                    else:
                        await self.send_message_func(chat_id, message_id, "Please set a voice sample first using /setvoice command.")
                elif text.lower() == '/setvoice':
                    await self.send_message_func(chat_id, message_id, "Please send a voice message to set as your TTS voice sample.")
                elif text.lower() == '/start':
                    response = "Привет! Я чат-бот на основе LLM с возможностью TTS. Чем могу помочь?"
                    await self.send_message_func(chat_id, message_id, response)
                elif text.lower() == '/reset':
                    response = reset_dialog(chat_id)
                    await self.send_message_func(chat_id, message_id, response)
                else:
                    cached_response = get_cached_response(text)
                    if cached_response:
                        await self.send_message_func(chat_id, message_id, cached_response.decode('utf-8'))
                    else:
                        self.typing_tasks[chat_id] = asyncio.create_task(self.continuous_typing_action(chat_id))
                        await self.sender.send(compress({
                            'chat_id': chat_id,
                            'text': text,
                            'message_id': message_id,
                            'type': 'text'
                        }))
            
            elif 'voice' in message:
                file_id = message['voice']['file_id']
                self.typing_tasks[chat_id] = asyncio.create_task(self.continuous_typing_action(chat_id))
                audio_content = await self.get_file_content(file_id, is_voice=True)
                
                if self.is_waiting_for_voice_sample(chat_id):
                    voice_sample_path = os.path.join(VOICE_SAMPLES_DIR, f"{chat_id}.ogg")
                    with open(voice_sample_path, 'wb') as f:
                        f.write(audio_content)
                    self.set_user_voice_sample(chat_id, voice_sample_path)
                    await self.send_message_func(chat_id, message_id, "Voice sample set successfully. You can now use /tts command.")
                else:
                    audio = AudioSegment.from_ogg(BytesIO(audio_content))
                    wav_audio = BytesIO()
                    audio.export(wav_audio, format="wav")
                    wav_audio.seek(0)
                    
                    await self.sender.send(compress({
                        'chat_id': chat_id,
                        'audio_content': wav_audio.getvalue(),
                        'message_id': message_id,
                        'type': 'voice'
                    }))
            
            elif 'document' in message:
                file_id = message['document']['file_id']
                file_name = message['document']['file_name']
                caption = message.get('caption', '')
                
                if file_name.endswith(('.txt', '.py', '.h', '.cpp')):
                    self.typing_tasks[chat_id] = asyncio.create_task(self.continuous_typing_action(chat_id))
                    file_content = await self.get_file_content(file_id)
                    full_content = f"Caption: {caption}\n\nFile Content:\n{file_content}" if caption else file_content
                    await self.sender.send(compress({
                        'chat_id': chat_id,
                        'text': full_content,
                        'message_id': message_id,
                        'type': 'file'
                    }))
                else:
                    await self.send_message_func(chat_id, message_id, "Пожалуйста, отправьте текстовый файл (.txt, .py, .h, .cpp)")
            
            else:
                await self.send_message_func(chat_id, message_id, "Пожалуйста, отправьте текстовое сообщение или текстовый файл")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def continuous_typing_action(self, chat_id):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendChatAction"
        data = {
            "chat_id": chat_id,
            "action": "typing"
        }
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        if response.status != 200:
                            logger.error(f"Failed to send typing action. Status code: {response.status}")
                await asyncio.sleep(4)
            except asyncio.CancelledError:
                logger.info(f"Typing action cancelled for chat_id: {chat_id}")
                break
            except Exception as e:
                logger.error


# ... (previous code remains the same)

class MessageHandler(tornado.web.RequestHandler):
    def initialize(self, sender, send_message_func, typing_tasks):
        self.sender = sender
        self.send_message_func = send_message_func
        self.typing_tasks = typing_tasks
        self.user_settings = {}  # Store user settings
        
    async def post(self):
        try:
            data = json.loads(self.request.body)
            message = data['message']
            chat_id = message['chat']['id']
            message_id = message["message_id"]
            
            if 'text' in message:
                text = message['text']
                logger.info(f"Received message: {text[:50]}...")

                if text.lower().startswith('/'):
                    await self.handle_command(chat_id, message_id, text.lower())
                else:
                    await self.handle_text_message(chat_id, message_id, text)
            
            elif 'voice' in message:
                await self.handle_voice_message(chat_id, message_id, message['voice']['file_id'])
            
            elif 'document' in message:
                await self.handle_document(chat_id, message_id, message['document'])
            
            else:
                await self.send_message_func(chat_id, message_id, "Пожалуйста, отправьте текстовое сообщение, голосовое сообщение или текстовый файл")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def handle_command(self, chat_id, message_id, command):
        if command == '/start':
            response = "Привет! Я чат-бот на основе LLM с возможностью TTS. Вот что я умею:\n\n" \
                       "/tts <текст> - Преобразовать текст в речь\n" \
                       "/setvoice - Установить образец вашего голоса для TTS\n" \
                       "/reset - Сбросить историю диалога\n" \
                       "/help - Показать это сообщение\n" \
                       "/settings - Настройки бота\n\n" \
                       "Чем могу помочь?"
            await self.send_message_func(chat_id, message_id, response)
        elif command.startswith('/tts'):
            await self.handle_tts_command(chat_id, message_id, command[5:].strip())
        elif command == '/setvoice':
            await self.send_message_func(chat_id, message_id, "Пожалуйста, отправьте голосовое сообщение для установки образца вашего голоса.")
            self.user_settings[chat_id] = {'waiting_for_voice': True}
        elif command == '/reset':
            response = reset_dialog(chat_id)
            await self.send_message_func(chat_id, message_id, response)
        elif command == '/help':
            await self.handle_command(chat_id, message_id, '/start')  # Reuse /start command for help
        elif command == '/settings':
            await self.show_settings(chat_id, message_id)
        else:
            await self.send_message_func(chat_id, message_id, "Неизвестная команда. Используйте /help для списка доступных команд.")

    async def handle_text_message(self, chat_id, message_id, text):
        cached_response = get_cached_response(text)
        if cached_response:
            await self.send_message_func(chat_id, message_id, cached_response.decode('utf-8'))
        else:
            self.typing_tasks[chat_id] = asyncio.create_task(self.continuous_typing_action(chat_id))
            await self.sender.send(compress({
                'chat_id': chat_id,
                'text': text,
                'message_id': message_id,
                'type': 'text'
            }))

    async def handle_voice_message(self, chat_id, message_id, file_id):
        self.typing_tasks[chat_id] = asyncio.create_task(self.continuous_typing_action(chat_id))
        audio_content = await self.get_file_content(file_id, is_voice=True)
        
        if self.user_settings.get(chat_id, {}).get('waiting_for_voice', False):
            voice_sample_path = os.path.join(VOICE_SAMPLES_DIR, f"{chat_id}.ogg")
            with open(voice_sample_path, 'wb') as f:
                f.write(audio_content)
            self.set_user_voice_sample(chat_id, voice_sample_path)
            self.user_settings[chat_id]['waiting_for_voice'] = False
            await self.send_message_func(chat_id, message_id, "Образец голоса успешно установлен. Теперь вы можете использовать команду /tts.")
        else:
            audio = AudioSegment.from_ogg(BytesIO(audio_content))
            wav_audio = BytesIO()
            audio.export(wav_audio, format="wav")
            wav_audio.seek(0)
            
            await self.sender.send(compress({
                'chat_id': chat_id,
                'audio_content': wav_audio.getvalue(),
                'message_id': message_id,
                'type': 'voice'
            }))

    async def handle_document(self, chat_id, message_id, document):
        file_id = document['file_id']
        file_name = document['file_name']
        caption = document.get('caption', '')
        
        if file_name.endswith(('.txt', '.py', '.h', '.cpp')):
            self.typing_tasks[chat_id] = asyncio.create_task(self.continuous_typing_action(chat_id))
            file_content = await self.get_file_content(file_id)
            full_content = f"Caption: {caption}\n\nFile Content:\n{file_content}" if caption else file_content
            await self.sender.send(compress({
                'chat_id': chat_id,
                'text': full_content,
                'message_id': message_id,
                'type': 'file'
            }))
        else:
            await self.send_message_func(chat_id, message_id, "Пожалуйста, отправьте текстовый файл (.txt, .py, .h, .cpp)")

    async def handle_tts_command(self, chat_id, message_id, text):
        voice_sample = self.get_user_voice_sample(chat_id)
        if voice_sample:
            await self.sender.send(compress({
                'chat_id': chat_id,
                'text': text,
                'voice_sample': voice_sample,
                'message_id': message_id,
                'type': 'tts'
            }))
        else:
            await self.send_message_func(chat_id, message_id, "Пожалуйста, сначала установите образец голоса с помощью команды /setvoice.")

    async def show_settings(self, chat_id, message_id):
        settings = self.user_settings.get(chat_id, {})
        voice_sample = "Установлен" if self.get_user_voice_sample(chat_id) else "Не установлен"
        
        settings_message = f"Настройки:\n\n" \
                           f"Образец голоса: {voice_sample}\n" \
                           f"Язык интерфейса: {settings.get('language', 'Русский')}\n" \
                           f"Максимальная длина ответа: {settings.get('max_response_length', 'Не ограничено')}\n\n" \
                           f"Для изменения настроек используйте соответствующие команды:"

        await self.send_message_func(chat_id, message_id, settings_message)

    def get_user_voice_sample(self, chat_id):
        voice_sample_path = os.path.join(VOICE_SAMPLES_DIR, f"{chat_id}.ogg")
        return voice_sample_path if os.path.exists(voice_sample_path) else None

    def set_user_voice_sample(self, chat_id, voice_sample_path):
        self.user_settings[chat_id] = self.user_settings.get(chat_id, {})
        self.user_settings[chat_id]['voice_sample'] = voice_sample_path

    async def get_file_content(self, file_id, is_voice=False):
        file_path = await self.get_file_path(file_id)
        url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    if is_voice:
                        return content
                    else:
                        return content.decode('utf-8')
                else:
                    logger.error(f"Failed to get file content. Status code: {response.status}")
                    return None

    async def get_file_path(self, file_id):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getFile"
        params = {"file_id": file_id}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['result']['file_path']
                else:
                    logger.error(f"Failed to get file path. Status code: {response.status}")
                    return None

    async def continuous_typing_action(self, chat_id):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendChatAction"
        data = {
            "chat_id": chat_id,
            "action": "typing"
        }
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        if response.status != 200:
                            logger.error(f"Failed to send typing action. Status code: {response.status}")
                await asyncio.sleep(4)
            except asyncio.CancelledError:
                logger.info(f"Typing action cancelled for chat_id: {chat_id}")
                break
            except Exception as e:
                logger.error(f"Error in typing action: {e}")
                await asyncio.sleep(1)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Telegram Bot is running")

def make_app(sender, send_message_func):
    typing_tasks = {}
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/message", MessageHandler, dict(sender=sender, send_message_func=send_message_func, typing_tasks=typing_tasks)),
    ])

async def send_message(chat_id, message_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "reply_to_message_id": message_id
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            if response.status != 200:
                logger.error(f"Failed to send message. Status code: {response.status}")

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

async def result_handler(receiver, send_message_func, typing_tasks):
    while True:
        try:
            result = decompress(await receiver.recv())
            chat_id = result['chat_id']
            message_id = result['message_id']
            
            if chat_id in typing_tasks:
                typing_tasks[chat_id].cancel()
                del typing_tasks[chat_id]
            
            if 'audio' in result:
                await send_voice(chat_id, message_id, result['audio'])
                await send_message_func(chat_id, message_id, result['text'])
            else:
                await send_message_func(chat_id, message_id, result['text'])
            
            save_message_to_db(chat_id, result['text'], 'assistant')
            cache_response(result['text'], result['text'])
        except Exception as e:
            logger.error(f"Error in result handler: {e}")

if __name__ == "__main__":
    context = Context.instance()
    sender = context.socket(zmq.PUSH)
    sender.connect(ZMQ_PIPELINE_ADDRESS)

    receiver = context.socket(zmq.PULL)
    receiver.connect(ZMQ_RESULT_ADDRESS)

    ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_ctx.load_cert_chain("YOURPUBLIC.pem", "YOURPRIVATE.key")

    http_server = tornado.httpserver.HTTPServer(
        make_app(sender, send_message),
        ssl_options=ssl_ctx
    )
    http_server.listen(8443)

    pipeline_process = Process(target=pipeline_worker)
    pipeline_process.start()

    loop = asyncio.get_event_loop()
    loop.create_task(result_handler(receiver, send_message, {}))
    tornado.ioloop.IOLoop.current().start()
