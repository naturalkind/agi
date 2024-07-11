import zmq
import zlib
import pickle
import torch
import threading
import cv2
import uuid
import os, time
from types import SimpleNamespace
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

mp = torch.multiprocessing


QUEUE_SIZE = mp.Value('i', 0)

def compress(obj):
    p = pickle.dumps(obj)
    return zlib.compress(p)


def decompress(pickled):
    p = zlib.decompress(pickled)
    return pickle.loads(p)


TOPIC = 'snaptravel'
prediction_functions = {}

RECEIVE_PORT = 5556 #os.getenv("RECEIVE_PORT")
SEND_PORT = 5555 #os.getenv("SEND_PORT")

# переводчик
#tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
#model_translater = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

# llm
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16, device_map=device)
tokenizer_llm = AutoTokenizer.from_pretrained("Mistral-7B-Instruct-v0.2")


def _parse_recv_for_json(result, topic=TOPIC):
    compressed_json = result[len(topic) + 1:]
    return decompress(compressed_json)

def _decrease_queue():
    with QUEUE_SIZE.get_lock():
        QUEUE_SIZE.value -= 1

def _increase_queue():
    with QUEUE_SIZE.get_lock():
        QUEUE_SIZE.value += 1
    
def send_prediction(message, result_publisher, topic=TOPIC):
    _increase_queue()
    model_name = message['model']
    body = message['body']
    id = message['id']
    
#    # Tokenize text
#    tokenized_text = tokenizer([str(body).lower()], return_tensors='pt')

#    # Perform translation and decode the output
#    translation = model_translater.generate(**tokenized_text)
#    body = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]

    # Print translated text
    messages = [
                {"role": "user", "content": body},
                ]
    # LLM
    encodeds = tokenizer_llm.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=10000, do_sample=True)
    decoded = tokenizer_llm.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    result = {"result": decoded[0]}

  #---------------------->

    if result.get('result') is None:
        time.sleep(1)
        compressed_message = compress({'error': True, 'error_msg': 'No result was given: ' + str(result), 'id': id})
        result_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
        _decrease_queue()
        return
  
  
    prediction = result['result']

    compressed_message = compress({'prediction': prediction, 'id': id})
    result_publisher.send(f'{topic} '.encode('utf8') + compressed_message)
    _decrease_queue()
    print ("SERVER", message, f'{topic} '.encode('utf8'))

def queue_size():
    return QUEUE_SIZE.value

def load_models():
    models = SimpleNamespace()
    return models

def start():
    global prediction_functions

    models = load_models()
    prediction_functions = {
    'queue': queue_size
    }

    print(f'Connecting to {RECEIVE_PORT} in server', TOPIC.encode('utf8'))
    context = zmq.Context()
    work_subscriber = context.socket(zmq.SUB)
    work_subscriber.setsockopt(zmq.SUBSCRIBE, TOPIC.encode('utf8'))
    work_subscriber.bind(f'tcp://127.0.0.1:{RECEIVE_PORT}')

    # send work
    print(f'Connecting to {SEND_PORT} in server')
    result_publisher = context.socket(zmq.PUB)
    result_publisher.bind(f'tcp://127.0.0.1:{SEND_PORT}')

    print('Server started')
    while True:
        message = _parse_recv_for_json(work_subscriber.recv())
        threading.Thread(target=send_prediction, args=(message, result_publisher), kwargs={'topic': TOPIC}).start()

if __name__ == '__main__':
  start()


#----------------------------->




