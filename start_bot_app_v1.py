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
from pydub import AudioSegment
from io import BytesIO
#from TTS.api import TTS
import soundfile as sf
import numpy as np
import librosa

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add new constants for voice processing
#WHISPER_MODEL_ID = "openai/whisper-large-v3"
#WHISPER_MODEL_ID = "openai/whisper-base"
WHISPER_MODEL_ID = "whisper-large-v3"

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

#def split_text(text, max_length=100):
#    sentences = re.split('(?<=[.!?]) +', text)
#    chunks = []
#    current_chunk = ""
#    
#    for sentence in sentences:
#        if len(current_chunk) + len(sentence) <= max_length:
#            current_chunk += sentence + " "
#        else:
#            chunks.append(current_chunk.strip())
#            current_chunk = sentence + " "
#    
#    if current_chunk:
#        chunks.append(current_chunk.strip())
#    
#    return chunks

def split_text(text, max_length=100):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:  # +1 for space
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
            #gpt_cond_len=3,
            language="ru",
        )
        wav_chunks.append(outputs["wav"])
    wav_path = f"data_users/clon_out.wav"
    sf.write(wav_path, np.concatenate(wav_chunks), samplerate=config.audio.output_sample_rate) #config.audio.output_sample_rate
    #print(f"Saved WAV file: {wav_path}", config.audio)
    return wav_path    


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
        "max_new_tokens": 5000, 
        "return_full_text": False, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    # Initialize Whisper model for voice recognition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    whisper_model.to(device)

    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    # клонирование голоса
    xtts_config = XttsConfig()
    xtts_config.load_json("/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/XTTS-v2/config.json")
    xtts_model = Xtts.init_from_config(xtts_config)
    xtts_model.load_checkpoint(xtts_config, checkpoint_dir="/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/XTTS-v2/", eval=True)
    xtts_model.cuda()
    
    while True:
        try:
            message = decompress(receiver.recv())
            chat_id = message['chat_id']
            user_id = message['user_id'] 
            message_id = message['message_id']
            message_type = message['type']
            if message_type == 'text':
                text = message['text']
            elif message_type == 'voice':
                audio_content = message['audio_content']
                # Convert audio content to text using Whisper
                result = whisper_pipe(audio_content)
                torch.cuda.empty_cache()
                text = result["text"]
            elif message_type == 'file':
                text = message['text']
            else:
                continue  # Skip unsupported message types
            # Получаем историю диалога из базы данных
            cursor.execute('SELECT message, role FROM dialogs WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 5', (chat_id,))
            history = cursor.fetchall()
            history.reverse()
            messages = [{"role": role, "content": msg} for msg, role in history]
            messages.append({"role": "user", "content": text})

            output = pipe(messages, **generation_args)
            response = output[0]['generated_text']
            
            torch.cuda.empty_cache()
            
            #if message_type == 'tts':
            if message_type == 'voice':
                # Generate speech using XTTS
                output_path = synthesize_speech(response, xtts_model, xtts_config, user_id)
                response = f"Перевод: {text} Ответ: {response}"
                with open(output_path, 'rb') as audio_file:
                    audio_content = audio_file.read()
                
#                os.remove(output_path)  # Clean up temporary file
                
                sender.send(compress({
                    'chat_id': chat_id,
                    'audio': audio_content,
                    'text': response,
                    'message_id': message_id,
                    'type': 'voice'
                }))
            else:
                sender.send(compress({
                    'chat_id': chat_id,
                    'text': response,
                    'message_id': message_id,
                    'type': 'text'
                }))
            
            
            # Очищаем кэш CUDA после каждой обработки
            torch.cuda.empty_cache()
        except Exception as e:
            # Очищаем кэш CUDA после каждой обработки
            torch.cuda.empty_cache()
            logger.error(f"Error in pipeline worker: {e}")

def start_pipeline_worker():
    Process(target=pipeline_worker).start()

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
            message = data['message']
            chat_id = message['chat']['id']
            user_id = message['from']['username']
            message_id = message["message_id"]
            
            # Create a unique key for each user in each chat
            unique_key = f"{chat_id}:{message_id}"
                     
            # Обработка сообщений...
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

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.stop_typing_action(unique_key)

    async def handle_text_message(self, chat_id, user_id, message_id, text):
        logger.info(f"Received message from user {user_id} in chat {chat_id}: {text[:50]}...")

        if text.lower() == '/start':
            response = "Привет! Я чат-бот на основе LLM. Чем могу помочь?"
            await self.send_message_func(chat_id, message_id, response)
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
            await asyncio.sleep(5)  # Мониторинг каждую минуту

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
            # Убедимся, что задача удалена из словаря typing_tasks
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

async def send_message(chat_id, message_id, text, typing_tasks):
    code_block, code_start = get_code_block(text)
    if len(text) <= 4096:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "reply_to_message_id": message_id,
            "parse_mode": "Markdown"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    logger.error(f"Failed to send message. Status code: {response.status}, Response: {await response.text()}")
    else:
        pre_text = text[:code_start] if 0 < code_start < 4096 else text[:50]
        await send_message(chat_id, message_id, f"Ответ слишком большой: {pre_text}...", typing_tasks)
        
        file = StringIO(text)
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        data = aiohttp.FormData()
        data.add_field('chat_id', str(chat_id))
        data.add_field('document', file, filename='response.txt')
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status != 200:
                    logger.error(f"Failed to send document. Status code: {response.status}, Response: {await response.text()}")
    unique_key = f"{chat_id}:{message_id}"
    # завершение отображение печати
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


async def process_responses(receiver, send_message_func):
    while True:
#        try:
            response = await receiver.recv()
            response = decompress(response)
            chat_id = response['chat_id']
            processed_text = response['text']
            message_id = response['message_id']
            input_type = response.get('type', 'text')
            
            if input_type == 'voice':
                audio = response['audio']
                await send_voice(chat_id, message_id, audio)
            
            save_message_to_db(chat_id, processed_text, "assistant")
            cache_response(processed_text, processed_text)
            
            await send_message_func(chat_id, message_id, processed_text)
#        except Exception as e:
#            logger.error(f"Error processing response: {e}")
            await asyncio.sleep(0.1)
        
        
def main():
    start_pipeline_worker()

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

    port = 8443
    http_server.listen(port)
    logger.info(f"Server started on port {port}")

    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.add_callback(process_responses, receiver, lambda chat_id, message_id, text: send_message(chat_id, message_id, text, typing_tasks))
    io_loop.start()

if __name__ == '__main__':
    main()
