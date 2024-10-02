## -*- coding: utf-8 -*-
#import tornado.httpserver
#import tornado.ioloop
#import tornado.web
#import ssl
#import json
#import time
#import requests
#import numpy as np
#import cv2
#import random
#import os
#import sys
#import logging
#import certifi
#from io import StringIO
#from fake_useragent import UserAgent
#from multiprocessing import Process, Queue
#import uuid
#import zlib
#import pickle
#import zmq
#import zmq.asyncio
#from zmq.asyncio import Context
#from md2tgmd import escape
#import markdown2
#import re
#from bs4 import BeautifulSoup

## Настройка логирования
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#logger = logging.getLogger(__name__)

## Константы
#TOPIC = 'snaptravel'
#RECEIVE_PORT = 5555
#SEND_PORT = 5556

## Загрузка данных конфигурации из JSON файла
#with open('config.bot', 'r') as json_file:
#    data = json.load(json_file)
#ACC_KEY = data['BOT_TOKEN']  # Замените на ваш токен

## Функции для работы с ZeroMQ
#def compress(obj):
#    return zlib.compress(pickle.dumps(obj))

#def decompress(pickled):
#    return pickle.loads(zlib.decompress(pickled))

#def start_zmq():
#    global work_publisher
#    context = zmq.Context()
#    work_publisher = context.socket(zmq.PUB)
#    work_publisher.connect(f'tcp://127.0.0.1:{SEND_PORT}')

#def send_zmq(args, model=None, topic=TOPIC):
#    id = str(uuid.uuid4())
#    message = {'body': args["title"], 'model': model, 'id': id}
#    compressed_message = compress(message)
#    work_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
#    return id

#def get_zmq(id, topic=TOPIC):
#    context = zmq.Context()
#    result_subscriber = context.socket(zmq.SUB)
#    result_subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode('utf8'))
#    result_subscriber.connect(f'tcp://127.0.0.1:{RECEIVE_PORT}')
#    while True:
#        result = decompress(result_subscriber.recv()[len(topic) + 1:])
#        if result['id'] == id:
#            result_subscriber.close()
#            if result.get('error'):
#                raise Exception(result['error_msg'])
#            return result

#def send_and_get(args, model=None):
#    id = send_zmq(args, model=model)
#    return get_zmq(id)

## Вспомогательные функции
#def split_text_into_chunks(text, chunk_size):
#    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

#def extract_special_chars(markdown_text):
#    md = markdown2.Markdown()
#    html = md.convert(markdown_text)
#    special_chars_pattern = re.compile(r'[^\w \t ()<>+!?.,;:\"\-_@{}=]')
#    return special_chars_pattern.findall(markdown_text)

## Класс для работы с HTTP-запросами
#class RequestLib:
#    def __init__(self):
#        self.session = requests.session()
#        self.session.proxies = {}
#        self.headers = {
#            'User-agent': UserAgent().random,
#            'Accept-Language': "en,en-US;q=0,5",
#            'Content-Type': "application/x-www-form-urlencoded",
#            'Connection': "keep-alive",
#            'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
#        }

#    def get(self, url, proxy=False):
#        return self.session.get(url, headers=self.headers)

#    def post(self, url, json_data):
#        return self.session.post(url, json=json_data, headers=self.headers)

## Инициализация HTTP-сессии
#sess = RequestLib()

## Класс для работы с очередью задач
#class SerializedWorker:
#    def __init__(self):
#        self.q = Queue()
#        self.p = Process(target=self.forked_process, args=(self.q,))
#        self.p.start()

#    def forked_process(self, q):
#        start_zmq()
#        while True:
#            work = q.get()
#            try:
#                _temp_dict = {"title": work[0] + " ответ на русском"}
#                logger.info(f"Processing: {work}")
#                answer = send_and_get(_temp_dict, model='Kandinsky-2.0')
#                text = answer['prediction']
#                text_parse = escape(text)

#                if len(text_parse) < 4000:
#                    self.send_message(work[1], text_parse, work[2])
#                else:
#                    self.send_large_message(work[1], text, work[2])
#            except Exception as e:
#                logger.error(f"Error processing message: {e}")

#    def send_message(self, chat_id, text, reply_to_message_id=None):
#        if not text:
#            logger.error("Attempted to send empty message")
#            return

#        code_context = {
#            "chat_id": chat_id,
#            "text": text,
#            "parse_mode": "MarkdownV2"
#        }

#        # Добавляем reply_to_message_id только если он предоставлен
#        if reply_to_message_id:
#            code_context["reply_to_message_id"] = reply_to_message_id

#        # Экранируем специальные символы для MarkdownV2
#        escaped_text = self.escape_markdown(text)
#        code_context["text"] = escaped_text

#        try:
#            response = sess.post(f"https://api.telegram.org/bot{ACC_KEY}/sendMessage", code_context)
#            response.raise_for_status()  # Вызовет исключение для неуспешных статус-кодов
#            logger.info(f"Message sent successfully: {response.json()}")
#        except requests.exceptions.RequestException as e:
#            logger.error(f"Failed to send message: {e}")
#            if response:
#                logger.error(f"Response: {response.text}")

#    def escape_markdown(self, text):
#        """Экранирование специальных символов для MarkdownV2."""
#        escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
#        return ''.join('\\' + char if char in escape_chars else char for char in text)

#    def send_large_message(self, chat_id, text, reply_to_message_id):
#        self.send_message(chat_id, "Ответ слишком большой, поэтому сохраняю в отдельный файл:", reply_to_message_id)
#        file = StringIO(text)
#        url = f"https://api.telegram.org/bot{ACC_KEY}/sendDocument"
#        files = {"document": file}
#        data = {"chat_id": chat_id}
#        response = requests.post(url, files=files, data=data)
#        logger.info(f"Large message sent as file: {response.json()}")

#    def queue_work(self, text):
#        self.q.put(text)

## Обработчик Tornado для входящих сообщений
#class MessageHandler(tornado.web.RequestHandler):
#    def initialize(self, worker):
#        self.worker = worker
#    def get(self):
#        print ("GET-------------->")
#    def post(self):
#        print("POST-------------->", self.request, self.request.body)
#        try:
#            data = json.loads(self.request.body)
#            
#            # Проверяем, есть ли ключ 'message' в данных
#            if 'message' not in data:
#                logger.warning("Received update without 'message' key")
#                return

#            message = data['message']
#            chat_id = message['chat']['id']
#            
#            # Проверяем наличие текста в сообщении
#            if 'text' not in message:
#                logger.warning(f"Received message without text for chat_id {chat_id}")
#                return

#            text_mess = message['text']
#            mess_id = message['message_id']

#            print(f"Received message: {text_mess} from chat_id: {chat_id}")

#            if text_mess.strip().lower() == "/start":
#                welcome_message = "Добрый день, я помощник @naturalkind. Вы можете задать мне любой вопрос, и я постараюсь вам помочь."
#                self.send_message(chat_id, welcome_message)
#                logger.info(f"Sent welcome message to chat_id {chat_id}")
#            elif text_mess:
#                self.worker.queue_work([text_mess, str(chat_id), str(mess_id)])
#                self.send_typing_action(chat_id)
#                logger.info(f"Queued work for message: {text_mess[:20]}... from chat_id {chat_id}")
#        except json.JSONDecodeError:
#            logger.error("Failed to decode JSON from request body")
#        except KeyError as e:
#            logger.error(f"KeyError in message processing: {e}")
#        except Exception as e:
#            logger.error(f"Unexpected error: {e}")

#    def send_message(self, chat_id, text):
#        url = f"https://api.telegram.org/bot{ACC_KEY}/sendMessage"
#        payload = {
#            "chat_id": chat_id,
#            "text": text,
#            "parse_mode": "HTML"
#        }
#        response = requests.post(url, json=payload)
#        if response.status_code != 200:
#            logger.error(f"Failed to send message. Status code: {response.status_code}, Response: {response.text}")

#    def send_typing_action(self, chat_id):
#        url = f"https://api.telegram.org/bot{ACC_KEY}/sendChatAction?chat_id={chat_id}&action=typing"
#        sess.get(url)

#if __name__ == '__main__':
#    worker = SerializedWorker()
#    application = tornado.web.Application([
#        (r'/', MessageHandler, dict(worker=worker)),
#    ])

#    http_server = tornado.httpserver.HTTPServer(
#        application,
#        ssl_options={
#            "certfile": "YOURPUBLIC.pem",
#            "keyfile": "YOURPRIVATE.key",
#            "ssl_version": ssl.PROTOCOL_TLSv1_2
#        }
#    )

#    port = 8443  # Выберите подходящий порт
#    http_server.listen(port)
#    logger.info(f"Server started on port {port}")
#    tornado.ioloop.IOLoop.current().start()




# -*- coding: utf-8 -*-
from fake_useragent import UserAgent
import tornado.httpserver
import tornado.ioloop
import tornado.web
import ssl
import json
import time
import requests as R
import numpy as np
import cv2
import random
import os
import sys
import certifi
from io import StringIO
import markdown2
import re
from bs4 import BeautifulSoup

# CLIENT
import uuid
import os
import json
import zlib
import pickle

import zmq
import zmq.asyncio
from zmq.asyncio import Context
from multiprocessing import Process, Queue
############################# LLM


from md2tgmd import escape # форматирование строк


work_publisher = None
result_subscriber = None
TOPIC = 'snaptravel'

RECEIVE_PORT = 5555
SEND_PORT = 5556 


def compress(obj):
    p = pickle.dumps(obj)
    return zlib.compress(p)


def decompress(pickled):
    p = zlib.decompress(pickled)
    return pickle.loads(p)
    
def start():
    global work_publisher, result_subscriber
    context = zmq.Context()
    work_publisher = context.socket(zmq.PUB)
    work_publisher.connect(f'tcp://127.0.0.1:{SEND_PORT}') 

def _parse_recv_for_json(result, topic=TOPIC):
    compressed_json = result[len(topic) + 1:]
    return decompress(compressed_json)

def send(args, model=None, topic=TOPIC):
    id = str(uuid.uuid4())
    message = {'body': args["title"], 'model': model, 'id': id}
    compressed_message = compress(message)
    work_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
    return id

def get(id, topic=TOPIC):
    context = zmq.Context()
    result_subscriber = context.socket(zmq.SUB)
    result_subscriber.setsockopt(zmq.SUBSCRIBE, topic.encode('utf8'))
    result_subscriber.connect(f'tcp://127.0.0.1:{RECEIVE_PORT}')
    result = _parse_recv_for_json(result_subscriber.recv())
    while result['id'] != id:
        result = _parse_recv_for_json(result_subscriber.recv())
    result_subscriber.close()
    if result.get('error'):
        raise Exception(result['error_msg'])
    return result

def send_and_get(args, model=None):
    id = send(args, model=model)
    res = get(id)
    return res

def split_text_into_chunks(text, chunk_size):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size

    return chunks


def extract_special_chars(markdown_text):
    md = markdown2.Markdown()
    html = md.convert(markdown_text)
    
    
    special_chars_pattern = re.compile(r'[^\\w \t ()<>+!?.,;:\"\-_@{}=]')
    markdown_soup = BeautifulSoup(markdown_text, "html.parser")
    print(html, dir(markdown2))
#    special_chars = markdown_soup.findAll(text=lambda text: isinstance(text, markdown2.inlinepatterns.Sup))
#    for char in special_chars:
#        print(char.name)
    return special_chars_pattern.findall(markdown_text)




class SerializedWorker():
    '''holds a forked process worker and an in memory queue for passing messages'''
    def __init__(self):
        self.q = Queue()
        # for a process with the forked_process function
        self.p = Process(target=self.forked_process, args=(self.q,))
        self.p.start()

  # this function runs out of process
    def forked_process(self, q):
        start()
        while True:
            work = q.get()
            _temp_dict = {}
            _temp_dict["title"] = work[0] + " ответ на русском"
            print ("START ---------->", work)
            ANSW = send_and_get(_temp_dict, model='Kandinsky-2.0')
            text = ANSW['prediction']#
            
    #        chunks = split_text_into_chunks(text, 2000)
    #        for chunk in chunks:
    #            print(chunk, "-------------------------")
    #            url = "https://api.telegram.org/bot"+acc_key+"/sendMessage?chat_id="+work[1]+"&text="+chunk+"&parse_mode=html"
    #            r = sess.get(url) 
    #            time.sleep(1)
            print (text, type(text), len(text))
            text_parse = escape(text)
            if len(text_parse) < 4000:
                code_context = {
                    "chat_id": work[1],
                    "text": text_parse,
                    "reply_to_message_id": work[2],
                    "parse_mode" : "MarkdownV2"#"html" #
                }
                
                error_req = sess.session.post("https://api.telegram.org/bot"+acc_key+"/sendMessage", json=code_context).json()
                print ("END --------------->", error_req)
                special_chars = extract_special_chars(text_parse)
                
            else:  
                code_context = {
                    "chat_id": work[1],
                    "text": f"Ответ слишком большой поэтому сохраняю в отдельный файл:",
                    "reply_to_message_id": work[2],
                    "parse_mode" : "html" 
                }
                error_req = sess.session.post("https://api.telegram.org/bot"+acc_key+"/sendMessage", json=code_context).json()
                print ("ERORRRRRRRRRR", error_req)
                file = StringIO(text)
                url = "https://api.telegram.org/bot"+acc_key+"/sendDocument";
                files = {"document": file}
                data = {"chat_id" : work[1]}
                r = R.post(url, files=files, data=data).json()
                print ("END ELSE --------------->", r)
                
        
    def queue_work(self, text):
        '''queue some work for the process to handle'''
        # a hash could be pass through for more complex values
        self.q.put(text)


#print (dir(ssl), certifi.where())
class RequestLib(object):
    def __init__(self):
        self.session = R.session()
        self.session.proxies = {}
        self.headers = {}
        self.headers['User-agent'] = UserAgent().random
        self.headers['Accept-Language'] = "en,en-US;q=0,5"
        self.headers['Content-Type'] = "application/x-www-form-urlencoded"
        self.headers['Connection'] = "keep-alive"
        self.headers['Accept'] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"

    def get(self, http, proxy=False):
        get_page = self.session.get(http, headers=self.headers)#, timeout=(10, 10)) 
        return get_page
            
sess = RequestLib()
with open('config.bot', 'r') as json_file:
    data = json.load(json_file)
    
acc_key = data['BOT_TOKEN']


class getToken(tornado.web.RequestHandler):
    def initialize(self, worker):
        self.worker = worker
    def get(self):
        print ("GET")
        self.write("hello")
    def post(self):
        data = json.loads(self.request.body)
        print ("POST")
        try:
            chat_id = data["message"]["from"]["id"]
            text_mess = data["message"]["text"]
            mess_id = data["message"]["message_id"]
            
    #        if "photo" in data["message"].keys():
    #            photo_id = data["message"]["photo"][-1]["file_id"]
    #            width_img = data["message"]["photo"][-1]["width"]
    #            height_img = data["message"]["photo"][-1]["height"]
    #            url = "https://api.telegram.org/bot"+acc_key+"/getFile?file_id="+ photo_id
    #            files = sess.get(url)
    #            data = json.loads(files.text)
    #            file_path = data["result"]["file_path"]
    #            files = sess.get("https://api.telegram.org/file/bot"+acc_key+"/"+file_path)
    #            img = files.content
    #            nparr = np.fromstring(img, np.uint8)
    #            img_t = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #            answer = loadimage
    #            url = "https://api.telegram.org/bot"+acc_key+"/sendMessage?chat_id="+str(chat_id)+"&text="+ANSW+"&parse_mode=html"
    #            r = sess.get(url)
    #            url = "https://api.telegram.org/bot"+acc_key+"/sendPhoto";
    #            files = {'photo': answer}
    #            data = {'chat_id' : chat_id}
    #            r = R.post(url, files=files, data=data) 

            #--------------------->
    #        print ("POST", chat_id, data)
            if text_mess.strip().lower() == "/start":
                welcome_message = "Добрый день, я помощник @naturalkind. Вы можете задать мне любой вопрос, и я постараюсь вам помочь."
                url = f"https://api.telegram.org/bot{acc_key}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": welcome_message,
                    "parse_mode": "HTML"
                }
                response = sess.session.post(url, json=payload)
            if text_mess != "":
                self.worker.queue_work([text_mess, str(chat_id), str(mess_id)])
#                url = "https://api.telegram.org/bot"+acc_key+"/sendMessage?chat_id="+str(chat_id)+"&text="+"скоро отвечу"+"&parse_mode=html"
                
                url = "https://api.telegram.org/bot"+acc_key+"/sendChatAction?chat_id="+str(chat_id)+"&action=typing"
                r = sess.get(url)
        except KeyError:
            print ("KeyError")
worker = SerializedWorker()
application = tornado.web.Application([
    (r'/', getToken, dict(worker=worker)),
])




if __name__ == '__main__':
#    start()
    http_server = tornado.httpserver.HTTPServer(application, ssl_options={"certfile":"YOURPUBLIC.pem",
                                                                          "keyfile":"YOURPRIVATE.key",
                                                                          "ssl_version": ssl.PROTOCOL_TLSv1_2})
                                                                          
#                                                                          "ssl_version": ssl.PROTOCOL_TLS_SERVER
    #http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8443)
    tornado.ioloop.IOLoop.instance().start()
#    
    
    
##



class MainHandler(tornado.web.RequestHandler):
  def initialize(self, worker):
    self.worker = worker

  def get(self):
    self.worker.queue_work()
    self.write("Queued processing of %d\n" % worker.counter)


if __name__ == "__main__":
  worker = SerializedWorker()
  application = tornado.web.Application([
    (r"/", MainHandler, dict(worker=worker)),
  ])
  application.listen(8888)
  tornado.ioloop.IOLoop.instance().start()




###
    
    
    
#    ['ALERT_DESCRIPTION_ACCESS_DENIED', 'ALERT_DESCRIPTION_BAD_CERTIFICATE', 'ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE', 'ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE', 'ALERT_DESCRIPTION_BAD_RECORD_MAC', 'ALERT_DESCRIPTION_CERTIFICATE_EXPIRED', 'ALERT_DESCRIPTION_CERTIFICATE_REVOKED', 'ALERT_DESCRIPTION_CERTIFICATE_UNKNOWN', 'ALERT_DESCRIPTION_CERTIFICATE_UNOBTAINABLE', 'ALERT_DESCRIPTION_CLOSE_NOTIFY', 'ALERT_DESCRIPTION_DECODE_ERROR', 'ALERT_DESCRIPTION_DECOMPRESSION_FAILURE', 'ALERT_DESCRIPTION_DECRYPT_ERROR', 'ALERT_DESCRIPTION_HANDSHAKE_FAILURE', 'ALERT_DESCRIPTION_ILLEGAL_PARAMETER', 'ALERT_DESCRIPTION_INSUFFICIENT_SECURITY', 'ALERT_DESCRIPTION_INTERNAL_ERROR', 'ALERT_DESCRIPTION_NO_RENEGOTIATION', 'ALERT_DESCRIPTION_PROTOCOL_VERSION', 'ALERT_DESCRIPTION_RECORD_OVERFLOW', 'ALERT_DESCRIPTION_UNEXPECTED_MESSAGE', 'ALERT_DESCRIPTION_UNKNOWN_CA', 'ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY', 'ALERT_DESCRIPTION_UNRECOGNIZED_NAME', 'ALERT_DESCRIPTION_UNSUPPORTED_CERTIFICATE', 'ALERT_DESCRIPTION_UNSUPPORTED_EXTENSION', 'ALERT_DESCRIPTION_USER_CANCELLED', 'AlertDescription', 'CERT_NONE', 'CERT_OPTIONAL', 'CERT_REQUIRED', 'CHANNEL_BINDING_TYPES', 'CertificateError', 'DER_cert_to_PEM_cert', 'DefaultVerifyPaths', 'HAS_ALPN', 'HAS_ECDH', 'HAS_NEVER_CHECK_COMMON_NAME', 'HAS_NPN', 'HAS_SNI', 'HAS_SSLv2', 'HAS_SSLv3', 'HAS_TLSv1', 'HAS_TLSv1_1', 'HAS_TLSv1_2', 'HAS_TLSv1_3', 'MemoryBIO', 'OPENSSL_VERSION', 'OPENSSL_VERSION_INFO', 'OPENSSL_VERSION_NUMBER', 'OP_ALL', 'OP_CIPHER_SERVER_PREFERENCE', 'OP_ENABLE_MIDDLEBOX_COMPAT', 'OP_NO_COMPRESSION', 'OP_NO_RENEGOTIATION', 'OP_NO_SSLv2', 'OP_NO_SSLv3', 'OP_NO_TICKET', 'OP_NO_TLSv1', 'OP_NO_TLSv1_1', 'OP_NO_TLSv1_2', 'OP_NO_TLSv1_3', 'OP_SINGLE_DH_USE', 'OP_SINGLE_ECDH_USE', 'Options', 'PEM_FOOTER', 'PEM_HEADER', 'PEM_cert_to_DER_cert', 'PROTOCOL_SSLv23', 'PROTOCOL_TLS', 'PROTOCOL_TLS_CLIENT', 'PROTOCOL_TLS_SERVER', 'PROTOCOL_TLSv1', 'PROTOCOL_TLSv1_1', 'PROTOCOL_TLSv1_2', 'Purpose', 'RAND_add', 'RAND_bytes', 'RAND_pseudo_bytes', 'RAND_status', 'SOCK_STREAM', 'SOL_SOCKET', 'SO_TYPE', 'SSLCertVerificationError', 'SSLContext', 'SSLEOFError', 'SSLError', 'SSLErrorNumber', 'SSLObject', 'SSLSession', 'SSLSocket', 'SSLSyscallError', 'SSLWantReadError', 'SSLWantWriteError', 'SSLZeroReturnError', 'SSL_ERROR_EOF', 'SSL_ERROR_INVALID_ERROR_CODE', 'SSL_ERROR_SSL', 'SSL_ERROR_SYSCALL', 'SSL_ERROR_WANT_CONNECT', 'SSL_ERROR_WANT_READ', 'SSL_ERROR_WANT_WRITE', 'SSL_ERROR_WANT_X509_LOOKUP', 'SSL_ERROR_ZERO_RETURN', 'TLSVersion', 'VERIFY_ALLOW_PROXY_CERTS', 'VERIFY_CRL_CHECK_CHAIN', 'VERIFY_CRL_CHECK_LEAF', 'VERIFY_DEFAULT', 'VERIFY_X509_PARTIAL_CHAIN', 'VERIFY_X509_STRICT', 'VERIFY_X509_TRUSTED_FIRST', 'VerifyFlags', 'VerifyMode', '_ASN1Object', '_DEFAULT_CIPHERS', '_Enum', '_GLOBAL_DEFAULT_TIMEOUT', '_IntEnum', '_IntFlag', '_OPENSSL_API_VERSION', '_PROTOCOL_NAMES', '_RESTRICTED_SERVER_CIPHERS', '_SSLContext', '_SSLMethod', '_SSLv2_IF_EXISTS', '_TLSAlertType', '_TLSContentType', '_TLSMessageType', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_create_default_https_context', '_create_stdlib_context', '_create_unverified_context', '_dnsname_match', '_inet_paton', '_ipaddress_match', '_nid2obj', '_socket', '_ssl', '_sslcopydoc', '_txt2obj', 'base64', 'cert_time_to_seconds', 'create_connection', 'create_default_context', 'errno', 'get_default_verify_paths', 'get_protocol_name', 'get_server_certificate', 'match_hostname', 'namedtuple', 'os', 'socket', 'socket_error', 'sys', 'warnings', 'wrap_socket']
