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

def pipeline_worker():
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
                # Convert audio content to text using Whisper
                result = whisper_pipe(audio_content)
                text = result["text"]
            elif message_type == 'file':
                text = message['text']
            else:
                continue  # Skip unsupported message types
            # Получаем историю диалога из базы данных
            cursor.execute('SELECT message, role FROM dialogs WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 10', (chat_id,))
            history = cursor.fetchall()
            history.reverse()
            messages = [{"role": role, "content": msg} for msg, role in history]
            messages.append({"role": "user", "content": text})

            output = pipe(messages, **generation_args)
            response = output[0]['generated_text']
            if message_type == 'voice':
                response = f"Перевод: {text} Ответ: {response}"
            sender.send(compress({
                'chat_id': chat_id,
                'text': response,
                'message_id': message_id
            }))
        except Exception as e:
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
        
    async def post(self):
        try:
            data = json.loads(self.request.body)
            message = data['message']
            chat_id = message['chat']['id']
            message_id = message["message_id"]
            
            if 'text' in message:
                text = message['text']
                logger.info(f"Received message: {text[:50]}...")

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
                        # Start typing action
                        self.typing_tasks[chat_id] = asyncio.create_task(self.continuous_typing_action(chat_id))
                        
                        await self.sender.send(compress({
                            'chat_id': chat_id,
                            'text': text,
                            'message_id': message_id,
                            'type': 'text'
                        }))
            
            elif 'voice' in message:
                file_id = message['voice']['file_id']
                # Start typing action
                self.typing_tasks[chat_id] = asyncio.create_task(self.continuous_typing_action(chat_id))
                audio_content = await self.get_file_content(file_id, is_voice=True)
                # Convert OGG to WAV
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
                    # Start typing action
                    self.typing_tasks[chat_id] = asyncio.create_task(self.continuous_typing_action(chat_id))
                    
                    file_content = await self.get_file_content(file_id)
                    # Combine file content with caption
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
                await asyncio.sleep(4)  # Telegram's typing action lasts about 5 seconds, so we refresh it every 4 seconds
            except asyncio.CancelledError:
                logger.info(f"Typing action cancelled for chat_id: {chat_id}")
                break
            except Exception as e:
                logger.error(f"Error in continuous_typing_action: {e}")
                await asyncio.sleep(1)
                

    async def get_file_content(self, file_id, is_voice=False):
        file_path = await self.get_file_path(file_id)
        url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    if is_voice:
                        return await response.read()
                    else:
                        return await response.text()
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
    # Cancel the typing task if it exists
    if chat_id in typing_tasks:
        typing_task = typing_tasks.pop(chat_id)
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass
    
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


async def process_responses(receiver, send_message_func):
    while True:
        try:
            response = await receiver.recv()
            response = decompress(response)
            chat_id = response['chat_id']
            text = response['text']
            message_id = response['message_id']
            input_type = response.get('type', 'text')
            
            if input_type == 'file':
                # Process file content
                processed_text = f"Обработанное содержимое файла: {text[:100]}..."  # Process the file content as needed
            else:
                # Process regular text input
                processed_text = text
            
            save_message_to_db(chat_id, processed_text, "assistant")
            cache_response(processed_text, processed_text)
            
            await send_message_func(chat_id, message_id, processed_text)
        except Exception as e:
            logger.error(f"Error processing response: {e}")
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
