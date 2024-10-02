import tornado.httpserver
import tornado.ioloop
import tornado.web
import ssl
import json
import requests
import logging
import uuid
import zlib
import pickle
import zmq
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import is_flash_attn_2_available
import torch
import redis
import sqlite3
import re
from datetime import datetime
from io import StringIO
import time


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы
TOPIC = 'chatbot'
RECEIVE_PORT = 5555
SEND_PORT = 5556

# Загрузка данных конфигурации из JSON файла
with open('config.bot', 'r') as json_file:
    data = json.load(json_file)
BOT_TOKEN = data['BOT_TOKEN']
# Инициализация модели и токенизатора
device = "cuda" if torch.cuda.is_available() else "cpu"

#model = AutoModelForCausalLM.from_pretrained("Mistral-7B-Instruct-v0 .2", torch_dtype=torch.float16, device_map=device)
#tokenizer = AutoTokenizer.from_pretrained("Mistral-7B-Instruct-v0.2")

#model_path = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/Phi-3.5-vision-instruct/"
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
    "max_new_tokens": 10000, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

# Инициализация Redis для кэширования
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Инициализация SQLite для хранения диалогов
conn = sqlite3.connect('dialogs.db')
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


# Функции для работы с ZeroMQ
def compress(obj):
    return zlib.compress(pickle.dumps(obj))

def decompress(pickled):
    return pickle.loads(zlib.decompress(pickled))

def start_zmq():
    global work_publisher
    context = zmq.Context()
    work_publisher = context.socket(zmq.PUB)
    work_publisher.connect(f'tcp://127.0.0.1:{SEND_PORT}')

def send_zmq(args, model=None, topic=TOPIC):
    id = str(uuid.uuid4())
    message = {'body': args["title"], 'model': model, 'id': id}
    compressed_message = compress(message)
    work_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
    return id

def get_zmq(id, topic=TOPIC):
    context = zmq.Context()
    result_subscriber = context.socket(zmq.SUB)
    result_subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode('utf8'))
    result_subscriber.connect(f'tcp://127.0.0.1:{RECEIVE_PORT}')
    while True:
        result = decompress(result_subscriber.recv()[len(topic) + 1:])
        if result['id'] == id:
            result_subscriber.close()
            if result.get('error'):
                raise Exception(result['error_msg'])
            return result

def send_and_get(args, model=None):
    id = send_zmq(args, model=model)
    return get_zmq(id)         

# Класс для работы с очередью задач
class SerializedWorker:
    def __init__(self):
        self.q = Queue()
        self.p = Process(target=self.forked_process, args=(self.q,))
        self.p.start()

    def forked_process(self, q):
        start_zmq()
        while True:
            work = q.get()
            try:
                _temp_dict = {"title": work[0]}
                logger.info(f"Processing: {work}")
                answer = send_and_get(_temp_dict, model='LLM')
                text = answer['prediction']
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    def queue_work(self, text):
        self.q.put(text)

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

def generate_response(chat_id, user_message):
#    cached_response = get_cached_response(user_message)
#    if cached_response:
#        return cached_response.decode('utf-8')

    if chat_id not in dialogs:
        dialogs[chat_id] = []
    
    dialogs[chat_id].append({"role": "user", "content": user_message})
    save_message_to_db(chat_id, user_message, "user")
    
    # Ограничиваем количество сообщений в истории
    if len(dialogs[chat_id]) > 5:
        dialogs[chat_id] = dialogs[chat_id][-5:]
    
    messages = dialogs[chat_id]
    
    #encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    #model_inputs = encodeds.to(device)
    
    #generated_ids = model.generate(model_inputs, max_new_tokens=10000, do_sample=True)
    #response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    output = pipe(messages, **generation_args) 
    response = output[0]['generated_text']
    
    dialogs[chat_id].append({"role": "assistant", "content": response})
    save_message_to_db(chat_id, response, "assistant")
    
    cache_response(user_message, response)
    
    return response

#    time.sleep(40)
#    print (f"..........{chat_id}, {len(messages)}")
#    return "LLM OFF"

def get_code_block(generated_text):
    # Находим начало и конец блока кода
    code_start = generated_text.find('```python')
    code_end = generated_text.find('```', code_start + 1)

    # Извлекаем блок кода
    if code_start != -1 and code_end != -1:
        code_block = generated_text[code_start+9:code_end].strip()
        return code_block, code_start
    else:
        return 0, 0 

class MessageHandler(tornado.web.RequestHandler):
    def initialize(self, worker):
        self.worker = worker
        
    def post(self):
        try:
            data = json.loads(self.request.body)
            message = data['message']
            chat_id = message['chat']['id']
            message_id = message["message_id"]
            text = message.get('text', '')
            print (f"POST ----------->: {message_id}")
            if text.lower() == '/start':
                response = "Привет! Я чат-бот на основе LLM. Чем могу помочь?"
            elif text.lower() == '/reset':
                response = reset_dialog(chat_id)
            else:
                self.worker.queue_work([text, str(chat_id), str(message_id)])
                self.send_typing_action(chat_id)
#                response = generate_response(chat_id, text)
#            self.send_message(chat_id, message_id, response)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def send_message(self, chat_id, message_id, text):
        _code, _code_start = get_code_block(text)
        if 3900 > len(text) > 0:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": text,
                "reply_to_message_id": message_id,
                "parse_mode": "Markdown"
    #            "parse_mode": "MarkdownV2"
    #            "parse_mode": "html"
            }
            response = requests.post(url, json=data)
            if response.status_code != 200:
                logger.error(f"Failed to send message. Status code: {response.status_code}, Response: {response.text}")
        elif text != "":
            if 3900 > _code_start > 0:
                pre_text = text[:_code_start]
            else:
                pre_text = text[:20]
            data = {
                "chat_id": chat_id,
                "text": f"Ответ слишком большой: {pre_text}...",
                "reply_to_message_id": message_id,
                "parse_mode": "Markdown" 
            }
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            response = requests.post(url, json=data).json()
            file = StringIO(text)
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
            files = {"document": file}
            data = {"chat_id" : chat_id}
            response = requests.post(url, files=files, data=data).json()

if __name__ == '__main__':
    worker = SerializedWorker()
    application = tornado.web.Application([
        (r'/', MessageHandler, dict(worker=worker)),
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
    tornado.ioloop.IOLoop.current().start()
