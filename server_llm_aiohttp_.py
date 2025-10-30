import aiohttp
from aiohttp import web
import asyncio
import json
import os
import glob
import pickle
import faiss
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import ssl
from datetime import datetime

# Импортируем ваши существующие модули
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, pipeline
import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup

from threading import Thread
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация модели (ваш существующий код)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"#, 2, 3"
#model_id = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/OpenCodeReasoning-Nemotron-14B"
#model_id = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/AceReason-Nemotron-14B"
model_id = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/Vistral-24B-Instruct"  # Уточните точное название репозитория
#model_id = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/gpt-oss-20b-ru-reasoner"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

#pipe = pipeline(
#    "text-generation",
#    model=model_id,  # ваша модель
#    tokenizer=tokenizer,
#    torch_dtype=torch.bfloat16,
#    device_map="auto",
#)


# Синхронизация устройств
for device in [0, 1]: #, 2, 3
    torch.cuda.synchronize(device=device)

print(torch.cuda.memory_summary(device=None, abbreviated=False))

# Класс RAGAnalyzer (ваш существующий код)
class RAGAnalyzer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        self.vector_store: Optional[FAISS] = None
        self.indexed_files: set = set()

    def load_documents(self, file_path: str):
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)
        return loader.load()

    def build_index(self, directory_path: str, exclude_dirs: list = None, exclude_files: list = None):
        exclude_dirs = exclude_dirs or []
        exclude_files = exclude_files or []
        python_files = glob.glob(os.path.join(directory_path, '**', '*.py'), recursive=True)
        
        if not python_files:
            return {"success": False, "message": "Python-файлы не найдены в указанной директории!"}

        documents = []
        new_indexed_files = set()
        
        for path in python_files:
            abs_path = os.path.abspath(path)
            filename = os.path.basename(path)
            
            if any(ex_dir in abs_path.split(os.sep) for ex_dir in exclude_dirs):
                continue
                
            if filename in exclude_files:
                continue
                
            try:
                docs = self.load_documents(path)
                docs = self.text_splitter.split_documents(docs)
                documents.extend(docs)
                new_indexed_files.add(abs_path)
            except Exception as e:
                return {"success": False, "message": f"Ошибка при обработке {path}: {str(e)}"}

        if not documents:
            return {"success": False, "message": "Нет документов для индексации после фильтрации!"}
            
        new_vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        if self.vector_store:
            self.vector_store.merge_from(new_vector_store)
            self.indexed_files.update(new_indexed_files)
        else:
            self.vector_store = new_vector_store
            self.indexed_files = new_indexed_files
        
        return {
            "success": True,
            "message": f"Индекс обновлён! Файлов: {len(self.indexed_files)}, Фрагментов: {self.vector_store.index.ntotal}",
            "files_count": len(self.indexed_files),
            "chunks_count": self.vector_store.index.ntotal
        }

    def remove_files(self, file_paths: List[str]):
        if not self.vector_store:
            return {"success": False, "message": "Индекс не инициализирован!"}

        abs_paths = [os.path.abspath(p) for p in file_paths]
        ids_to_remove = []
        
        for i, doc in enumerate(self.vector_store.docstore._dict.values()):
            if doc.metadata.get('source') in abs_paths:
                ids_to_remove.append(i)

        if not ids_to_remove:
            return {"success": False, "message": "Указанные файлы не найдены в индексе"}

        remaining_ids = [
            i for i in range(len(self.vector_store.docstore._dict))
            if i not in ids_to_remove
        ]
        
        new_index = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(self.vector_store.index.d),
            docstore=self.vector_store.docstore,
            index_to_docstore_id=self.vector_store.index_to_docstore_id
        )
        
        remaining_vectors = self.vector_store.index.reconstruct_batch(remaining_ids)
        new_index.add_vectors(remaining_ids, remaining_vectors)

        self.vector_store = new_index
        self.indexed_files = self.indexed_files - set(abs_paths)
        
        return {
            "success": True,
            "message": f"Удалено файлов: {len(abs_paths)}\nОсталось файлов: {len(self.indexed_files)}\nФрагментов: {self.vector_store.index.ntotal}"
        }

    def get_indexed_files(self) -> List[str]:
        return list(self.indexed_files)

    def save_index(self, save_path: str):
        if not self.vector_store:
            return {"success": False, "message": "Индекс не инициализирован!"}
            
        os.makedirs(save_path, exist_ok=True)
        self.vector_store.save_local(os.path.join(save_path, "faiss_index"))
        
        metadata = {"indexed_files": list(self.indexed_files)}
        with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
            
        return {"success": True, "message": f"Индекс сохранён в: {save_path}"}

    def load_index(self, load_path: str):
        try:
            self.vector_store = FAISS.load_local(
                os.path.join(load_path, "faiss_index"),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            with open(os.path.join(load_path, "metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)
                
            self.indexed_files = set(metadata["indexed_files"])
            return {
                "success": True,
                "message": f"Индекс загружен! Файлов: {len(self.indexed_files)}, Фрагментов: {self.vector_store.index.ntotal}"
            }
        except Exception as e:
            return {"success": False, "message": f"Ошибка загрузки индекса: {str(e)}"}

    def clear_index(self):
        self.vector_store = None
        self.indexed_files = set()
        return {"success": True, "message": "Индекс полностью очищен"}
        
    def search_documents(self, query: str, k: int = 5) -> str:
        if not self.vector_store:
            return ""
        
        results = self.vector_store.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])
        return f"### Контекст из документов:\n{context}"

# Инициализация RAG
rag = RAGAnalyzer()

# Функция для форматирования истории чата
def get_model_response(outputs, tokenizer):
    # Получаем полную декодированную последовательность
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Определяем маркеры начала/конца ответа
    assistant_start = "<|assistant|>"
    assistant_end = "<|end|>"
    
    # Извлекаем ответ между маркерами
    start_idx = full_text.rfind(assistant_start)
    if start_idx != -1:
        start_idx += len(assistant_start)
        end_idx = full_text.find(assistant_end, start_idx)
        response = full_text[start_idx:end_idx].strip() if end_idx != -1 else full_text[start_idx:].strip()
    else:
        response = full_text.strip()
    
    # Чистим артефакты токенизации
    response = tokenizer.clean_up_tokenization(response)
    return response

# Вспомогательная функция для потоковой генерации
async def generate_response_stream(query: str, use_rag: bool):
    if use_rag:
        context = rag.search_documents(query)
        prompt = f"{context}\n\n### Запрос:\n{query}\n\nОтвет должен содержать конкретные примеры кода из контекста."
    else:
        prompt = query
    
    messages = [{'role': 'user', 'content': prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        truncation=True,
        max_length=24576
    ).to(model.device)
    
    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True
    )
    
    generation_kwargs = dict(
        inputs=inputs,
        streamer=streamer,
        max_new_tokens=24576,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    try:
        for new_token in streamer:
            yield f"data: {json.dumps({'token': new_token})}\n\n"
    except Exception as e:
        logger.error(f"Error in stream generation: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        torch.cuda.empty_cache()

# CORS middleware
@web.middleware
async def cors_middleware(request, handler):
    if request.method == 'OPTIONS':
        response = web.Response()
    else:
        try:
            response = await handler(request)
        except Exception as e:
            logger.error(f"Server error: {e}")
            response = web.json_response({'error': str(e)}, status=500)
    
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# API endpoints
async def root_handler(request):
    return web.json_response({"message": "RAG Code Analyzer API"})

async def chat_stream_handler(request):
    try:
        data = await request.json()
        message = data.get('message', '')
        use_rag = data.get('use_rag', True)
        
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
            }
        )
        
        await response.prepare(request)
        
        try:
            async for chunk in generate_response_stream(message, use_rag):
                try:
                    await response.write(chunk.encode('utf-8'))
                    await response.drain()  # Важно для асинхронной записи
                except Exception as e:
                    logger.info(f"Client disconnected: {e}")
                    break
            
            await response.write(b"data: [DONE]\n\n")
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            error_msg = f"data: {json.dumps({'error': str(e)})}\n\n"
            await response.write(error_msg.encode('utf-8'))
        
        return response
        
    except json.JSONDecodeError:
        return web.json_response({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def build_index_handler(request):
    try:
        data = await request.json()
        directory = data.get('directory', '')
        exclude_dirs = data.get('exclude_dirs', '')
        exclude_files = data.get('exclude_files', '')
        
        exclude_dirs_list = [d.strip() for d in exclude_dirs.split(",") if d.strip()]
        exclude_files_list = [f.strip() for f in exclude_files.split(",") if f.strip()]
        
        result = rag.build_index(
            directory,
            exclude_dirs=exclude_dirs_list or ["venv", ".git", "__pycache__", "migrations"],
            exclude_files=exclude_files_list or ["config.py", "secret_keys.py"]
        )
        
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Build index error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def get_indexed_files_handler(request):
    try:
        return web.json_response({"files": rag.get_indexed_files()})
    except Exception as e:
        logger.error(f"Get files error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def save_index_handler(request):
    try:
        data = await request.json()
        path = data.get('path', '')
        result = rag.save_index(path)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Save index error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def load_index_handler(request):
    try:
        data = await request.json()
        path = data.get('path', '')
        result = rag.load_index(path)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Load index error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def remove_files_handler(request):
    try:
        data = await request.json()
        file_paths = data.get('file_paths', [])
        result = rag.remove_files(file_paths)
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Remove files error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def clear_index_handler(request):
    try:
        result = rag.clear_index()
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Clear index error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def upload_files_handler(request):
    return web.json_response({"message": "Загрузка файлов будет реализована отдельно"})

############
### Тестирование народная броня
############

### Получать данные

def get_context():
    # Открываем и читаем HTML-файл
    with open('/home/sadko/Загрузки/Telegram Desktop/ChatExport_2025-09-29/messages.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Создаем объект BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Находим все div с классом 'text'
    text_divs = soup.find_all('div', class_='text')

    # Извлекаем текст из каждого div и объединяем его
    #all_text = '\n'.join(div.get_text(strip=True) for div in text_divs)
    #all_text = '\n'.join("Запись народная броня #{}. {}".format(i+1, div.get_text(strip=True)) for i, div in enumerate(text_divs))
    N = 11 # максимально для Nemotron 14b 2xGPU = 17
    all_text = '\n'.join(f"Продукт {i+1}\n{div.get_text(strip=True)} Номер телефона для заказа: +79493061593" for i, div in enumerate(text_divs[:N]))
    print (len(all_text))
    # Конкретный диапазон [start:end]
#    start, end = 2, 8
#    all_text = '\n'.join(f"{i+1}. {div.get_text(strip=True)}" for i, div in enumerate(text_divs[start:end]))

#    # С шагом (каждый второй элемент)
#    all_text = '\n'.join(f"{i+1}. {div.get_text(strip=True)}" for i, div in enumerate(text_divs[::2]))
    
    
    return all_text

CONTEXT1 = get_context()

async def generate_handler(request):
    try:
        data = await request.json()
        messages = data.get('messages', [])
        print(messages)
        max_new_tokens = data.get('max_new_tokens', 500)
        temperature = data.get('temperature', 0.7)
        
        logger.info(f"GENERATE --> {messages}")
        
        # Определяем системный промпт для продавца спортивной экипировки
        # Определяем системный промпт для продавца спортивной экипировки
        telegram_username = "NaodnayaBronya_Bot"
#        system_prompt = f"""Вы - профессиональный консультант в магазине экипировки. Ваша задача - помогать клиентам с выбором товаров, давать профессиональные советы и отвечать на вопросы.

#При представлении товаров ВСЕГДА используйте следующий шаблон для каждого товара:


#Вот ассортимент нашего магазина:
#{CONTEXT1}

#Всегда будьте вежливы, предлагайте дополнительные товары и уточняйте детали, если нужно. Отвечайте точно и по делу, используя профессиональные знания о спортивной экипировке.

#Магазин "Народная броня" находится по адресу: г. Донецк, пр. Театральный 15
#Режим работы: с понедельника по субботу с 9.00 до 18.00, воскресенье с 10.00 до 17.00
#Телефон магазина: +79493061593
#Телеграм: {telegram_username}

#ВАЖНО: Всегда используйте указанный шаблон при перечислении товаров!"""

        system_prompt = f"""Вы - профессиональный консультант в магазине экипировки. Ваша задача - помогать клиентам с выбором товаров, давать профессиональные советы и отвечать на вопросы.

Вот ассортимент нашего магазина:
{CONTEXT1}

Всегда будьте вежливы, предлагайте дополнительные товары и уточняйте детали, если нужно. Отвечайте точно и по делу, используя профессиональные знания о спортивной экипировке.
Народная броня находится по адресу г. Донецк, пр. Театральный 15
работает с понедельника по субботу с 9.00 до 18.00
Воскресенье с 10.00 до 17.00
Телефон магазина +79493061593
Телеграм {telegram_username}"""

        # Проверяем, есть ли уже системный промпт в сообщениях
        has_system_prompt = any(msg.get('role') == 'system' for msg in messages)
        
        # Добавляем системный промпт только если его еще нет
        if not has_system_prompt:
            messages_with_system = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages_with_system = messages
        
        # Оставляем только системный промпт и последний пользовательский запрос
        filtered_messages = []
        
        # Добавляем все системные промпты
        system_messages = [msg for msg in messages_with_system if msg.get('role') == 'system']
        filtered_messages.extend(system_messages)
        
        # Находим последний пользовательский запрос
        user_messages = [msg for msg in messages_with_system if msg.get('role') == 'user']
        if user_messages:
            # Берем самый последний пользовательский запрос
            last_user_message = user_messages[-1]
            filtered_messages.append(last_user_message)
        
        print("Filtered messages:", filtered_messages)
        start_time = time.time()
        
        # Генерация текста
        formatted_prompt = tokenizer.apply_chat_template(
            filtered_messages,  # Используем отфильтрованные сообщения
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            top_p=0.9
        )
        
        # Декодирование результата
        outputs = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        outputs = outputs.replace('**@NaodnayaBronya_Bot** ', '@NaodnayaBronya_Bot')
        logger.info(f"END GENERATE ----------------> {outputs} \n End time {time.time()-start_time}")
        torch.cuda.empty_cache()
        return web.json_response({"response": outputs})
    
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return web.json_response({"error": "Generation failed"}, status=500)


### Рабочая стабильная версия

#async def generate_handler(request):
#    try:
#        data = await request.json()
#        messages = data.get('messages', [])
#        print (messages)
#        max_new_tokens = data.get('max_new_tokens', 2000)
#        temperature = data.get('temperature', 0.6)
#        
#        logger.info(f"GENERATE --> {messages}")
#        
#        start_time = time.time()
#        # Генерация текста
#        formatted_prompt = tokenizer.apply_chat_template(
#            messages,
#            tokenize=False,
#            add_generation_prompt=True
#        )
#        inputs = tokenizer(formatted_prompt, return_tensors="pt")
#        
#        # V1
#        outputs = model.generate(
#            **inputs,
#            max_new_tokens=max_new_tokens,
#            #temperature=temperature,
##            do_sample=True,
#            do_sample=False,
#            eos_token_id=tokenizer.eos_token_id,
#            pad_token_id=tokenizer.pad_token_id
#        )
#        
#        # Декодирование результата
#        #outputs = get_model_response(outputs, tokenizer)
#        outputs = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
#        logger.info(f"END GENERATE ----------------> {outputs} \n End time {time.time()-start_time}")
#        torch.cuda.empty_cache()
#        return web.json_response({"response": outputs})
#    
#    except Exception as e:
#        logger.error(f"Generation error: {str(e)}")
#        return web.json_response({"error": "Generation failed"}, status=500)


### pipeline wersion

#async def generate_handler(request):
#    try:
#        data = await request.json()
#        print(data)
#        messages = data.get('messages', [])
#        max_new_tokens = data.get('max_new_tokens', 2000)
#        temperature = data.get('temperature', 0.6)
#        
#        logger.info(f"GENERATE --> {messages}")
#        
#        start_time = time.time()
#        
#        # Форматируем промпт с помощью чатового шаблона
#        formatted_prompt = tokenizer.apply_chat_template(
#            messages,
#            tokenize=False,
#            add_generation_prompt=True
#        )
#        
#        # Генерация с помощью pipeline
#        outputs = pipe(
#            formatted_prompt,
#            max_new_tokens=max_new_tokens,
#            return_full_text=False,  # возвращать только сгенерированный текст
#            num_return_sequences=1
#        )
#        
#        # Извлекаем сгенерированный текст
#        generated_text = outputs[0]['generated_text']
#        
#        logger.info(f"END GENERATE ----------------> {generated_text} \n End time {time.time()-start_time}")
#        torch.cuda.empty_cache()
#        return web.json_response({"response": generated_text})
#    
#    except Exception as e:
#        logger.error(f"Generation error: {str(e)}")
#        return web.json_response({"error": "Generation failed"}, status=500)


async def health_check(request):
    return web.json_response({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available()
    })

# Создание и настройка приложения
def create_app():
    app = web.Application(middlewares=[cors_middleware])
    
    # Добавление маршрутов
    app.router.add_get('/', root_handler)
    app.router.add_post('/api/chat', chat_stream_handler)
    app.router.add_post('/api/index/build', build_index_handler)
    app.router.add_get('/api/index/files', get_indexed_files_handler)
    app.router.add_post('/api/index/save', save_index_handler)
    app.router.add_post('/api/index/load', load_index_handler)
    app.router.add_post('/api/index/remove', remove_files_handler)
    app.router.add_post('/api/index/clear', clear_index_handler)
    app.router.add_post('/api/upload', upload_files_handler)
    app.router.add_get('/health', health_check)
    app.router.add_post('/generate', generate_handler)
    
    # Обработка OPTIONS запросов для CORS
    async def options_handler(request):
        return web.Response(status=200)
    
    app.router.add_route('OPTIONS', '/{path:.*}', options_handler)
    
    return app

if __name__ == "__main__":
    app = create_app()
    
    # Создание SSL контекста
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_verify_locations(cafile='ssl/ca.crt')
    ssl_context.load_cert_chain(
        certfile='ssl/server.crt', 
        keyfile='ssl/server.key'
    )
    ssl_context.verify_mode = ssl.CERT_REQUIRED

    # Запуск приложения с SSL
    web.run_app(
        app,
        host="0.0.0.0",
        port=5000,
        ssl_context=ssl_context,
        access_log=logger,
        print=None
    )
