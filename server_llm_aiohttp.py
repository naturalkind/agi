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
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup

from threading import Thread
import time
import pdfplumber
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTFigure, LTRect
from pdf2image import convert_from_path
import pytesseract
import pymupdf  # Импортируем библиотеку
from PIL import Image
import pypdf

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация модели (ваш существующий код)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"#, 2,3"
model_id = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/Vistral-24B-Instruct"  # Уточните точное название репозитория
#model_id = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/gpt-oss-20b-ru-reasoner"
#model_id = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/OpenCodeReasoning-Nemotron-14B"

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

# Синхронизация устройств
for device in [0, 1, 2, 3]: #
    torch.cuda.synchronize(device=device)

#print(torch.cuda.memory_summary(device=None, abbreviated=False))

def get_context_full():
    with open('/home/sadko/Загрузки/Telegram Desktop/ChatExport_2025-09-29/messages.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    text_divs = soup.find_all('div', class_='text')

    products = []
    for i, div in enumerate(text_divs):
        product_text = f"Продукт {i+1}: {div.get_text(strip=True)} Номер телефона для заказа: +79493061593"
        products.append(product_text)
    
    print(f"Найдено продуктов: {len(products)}")
    return products

# Класс RAGAnalyzer (ваш существующий код)
class RAGAnalyzer:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            #chunk_size=512,
            chunk_size=300,
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

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        
        if not texts:
            return {"success": False, "message": "Нет текстов для добавления!"}
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Создаем документы с метаданными
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        documents = []
        for text, metadata in zip(texts, metadatas):
            documents.append(Document(page_content=text, metadata=metadata))
        
        # Разбиваем на чанки
        split_documents = self.text_splitter.split_documents(documents)
        
        # Добавляем в векторное хранилище
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                documents=split_documents,
                embedding=self.embeddings
            )
        else:
            self.vector_store.add_documents(split_documents)
        
        return {
            "success": True, 
            "message": f"Добавлено {len(split_documents)} фрагментов из {len(texts)} текстов"
        }


# Инициализация RAG
rag = RAGAnalyzer()
rag.clear_index()

# Получение данных из телеграмм
products = get_context_full()

# Добавление в индекс с метаданными
metadatas = [{"source": "telegram_export", "product_id": i+1} for i in range(len(products))]
result = rag.add_texts(products, metadatas)

print(result["message"], len(products), type(products))

gost0 = rag.load_documents("/home/sadko/Загрузки/GOST0.pdf")
gost0 = rag.text_splitter.split_documents(gost0)

# Извлекаем текст из каждого документа и создаем метаданные
texts = [doc.page_content for doc in gost0]  # Извлекаем текст
metadatas = [{"source": "pdf file gost", "product_id": i} for i in range(len(texts))]  # Начинаем с 0

result = rag.add_texts(texts, metadatas)  # Передаем тексты, а не документы


gost1 = rag.load_documents("/home/sadko/Загрузки/GOST1.pdf")
gost1 = rag.text_splitter.split_documents(gost1)

# Извлекаем текст из каждого документа и создаем метаданные
texts = [doc.page_content for doc in gost1]  # Извлекаем текст
metadatas = [{"source": "pdf file gost", "product_id": i} for i in range(len(texts))]  # Начинаем с 0

result = rag.add_texts(texts, metadatas)  # Передаем тексты, а не документы



# Тестирование
# Поиск релевантных продуктов
#query = "подсумок"
#context = rag.search_documents(query, k=3)
#print(context)


# Вспомогательная функция для потоковой генерации
async def generate_response_stream(query: str, use_rag: bool):
    if use_rag:
        context = rag.search_documents(query)
        prompt = f"{context}\n\n### Запрос:\n{query}\n\nОтвет должен содержать конкретные примеры кода из контекста."
    else:
        prompt = query
    messages = [{'role': 'user', 'content': prompt}]

    telegram_username = "NaodnayaBronya_Bot"
    system_prompt = f"""Вы - профессиональный консультант в магазине военной экипировки. Ваша задача - помогать клиентам с выбором товаров, давать профессиональные советы и отвечать на вопросы. Предлагать не больше двух товаров, общая длина текста не больше 240 слов.
Если клиен просит показать категории товара сформировать на основе ассортимента.
Использовать режим разметки ответов markdownv2 telegram api

Вот ассортимент нашего магазина:
{context}

Всегда будьте вежливы, предлагайте дополнительные товары и уточняйте детали, если нужно. Отвечайте точно и по делу, используя профессиональные знания о спортивной экипировке.
Народная броня находится по адресу г. Донецк, пр. Театральный 15
работает с понедельника по субботу с 9.00 до 18.00
Воскресенье с 10.00 до 17.00
Телефон магазина +79493061593
Телеграм {telegram_username}"""

    has_system_prompt = any(msg.get('role') == 'system' for msg in messages)
    
    if not has_system_prompt:
        messages_with_system = [{"role": "system", "content": system_prompt}] + messages
    else:
        messages_with_system = messages
    
    filtered_messages = []
    
    system_messages = [msg for msg in messages_with_system if msg.get('role') == 'system']
    filtered_messages.extend(system_messages)
    
    user_messages = [msg for msg in messages_with_system if msg.get('role') == 'user']
    if user_messages:
        last_user_message = user_messages[-1]
        filtered_messages.append(last_user_message)
    
    print("Filtered messages:", filtered_messages)
    
    
    inputs = tokenizer.apply_chat_template(
        filtered_messages,
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

# Обновленный root_handler для обслуживания index.html
async def root_handler(request):
    print ("------------<")
    # Путь к вашему index.html файлу
    index_path = Path('index.html')
    
    if not index_path.exists():
        # Если файл не найден, возвращаем сообщение об ошибке
        return web.json_response({"error": "index.html not found"}, status=404)
    
    return web.FileResponse(index_path)

# Статический файловый хендлер для обслуживания других ресурсов (CSS, JS и т.д.)
async def static_handler(request):
    path = request.match_info.get('path', '')
    static_path = Path(path)
    
    if not static_path.exists():
        return web.json_response({"error": "File not found"}, status=404)
    
    return web.FileResponse(static_path)

# API endpoints
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
                    await response.drain()
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

def get_context():
    with open('/home/sadko/Загрузки/Telegram Desktop/ChatExport_2025-09-29/messages.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    text_divs = soup.find_all('div', class_='text')

    N = 11
    all_text = '\n'.join(f"Продукт {i+1}\n{div.get_text(strip=True)} Номер телефона для заказа: +79493061593" for i, div in enumerate(text_divs[:N]))
    print(len(all_text))
    
    return all_text

#CONTEXT1 = get_context()

#async def generate_handler(request):
#    try:
#        data = await request.json()
#        messages = data.get('messages', [])
#        print(messages)
#        max_new_tokens = data.get('max_new_tokens', 500)
#        temperature = data.get('temperature', 0.7)
#        
#        logger.info(f"GENERATE --> {messages}")
#        
#        telegram_username = "NaodnayaBronya_Bot"
#        system_prompt = f"""Вы - профессиональный консультант в магазине экипировки. Ваша задача - помогать клиентам с выбором товаров, давать профессиональные советы и отвечать на вопросы. Предлагать не больше двух товаров, общая длина текста не больше 240 слов.

#Вот ассортимент нашего магазина:
#{CONTEXT1}

#Всегда будьте вежливы, предлагайте дополнительные товары и уточняйте детали, если нужно. Отвечайте точно и по делу, используя профессиональные знания о спортивной экипировке.
#Народная броня находится по адресу г. Донецк, пр. Театральный 15
#работает с понедельника по субботу с 9.00 до 18.00
#Воскресенье с 10.00 до 17.00
#Телефон магазина +79493061593
#Телеграм {telegram_username}"""

#        has_system_prompt = any(msg.get('role') == 'system' for msg in messages)
#        
#        if not has_system_prompt:
#            messages_with_system = [{"role": "system", "content": system_prompt}] + messages
#        else:
#            messages_with_system = messages
#        
#        filtered_messages = []
#        
#        system_messages = [msg for msg in messages_with_system if msg.get('role') == 'system']
#        filtered_messages.extend(system_messages)
#        
#        user_messages = [msg for msg in messages_with_system if msg.get('role') == 'user']
#        if user_messages:
#            last_user_message = user_messages[-1]
#            filtered_messages.append(last_user_message)
#        
#        print("Filtered messages:", filtered_messages)
#        start_time = time.time()
#        
#        formatted_prompt = tokenizer.apply_chat_template(
#            filtered_messages,
#            tokenize=False,
#            add_generation_prompt=True
#        )
#        inputs = tokenizer(formatted_prompt, return_tensors="pt")
#        
#        outputs = model.generate(
#            **inputs,
#            max_new_tokens=max_new_tokens,
#            temperature=temperature,
#            do_sample=True,
#            eos_token_id=tokenizer.eos_token_id,
#            pad_token_id=tokenizer.pad_token_id,
#            top_p=0.9
#        )
#        
#        outputs = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
#        outputs = outputs.replace('**@NaodnayaBronya_Bot** ', '@NaodnayaBronya_Bot')
#        logger.info(f"END GENERATE ----------------> {outputs} \n End time {time.time()-start_time}")
#        torch.cuda.empty_cache()
#        return web.json_response({"response": outputs})
#    
#    except Exception as e:
#        logger.error(f"Generation error: {str(e)}")
#        return web.json_response({"error": "Generation failed"}, status=500)


async def generate_handler(request):
    try:
        data = await request.json()
        messages = data.get('messages', [])
        print(messages)
        max_new_tokens = data.get('max_new_tokens', 500)
        temperature = data.get('temperature', 0.7)
        
        # Извлекаем последнее сообщение пользователя для поиска
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        if user_messages:
            query = user_messages[-1].get('content', '')
        else:
            query = ""
            
        context = rag.search_documents(query, k=3)
        logger.info(f"GENERATE --> {messages}")
        logger.info(f"CONTEXT --> {context}")
        telegram_username = "NaodnayaBronya_Bot"
#        system_prompt = f"""Вы - профессиональный консультант в магазине экипировки. Ваша задача - помогать клиентам с выбором товаров, давать профессиональные советы и отвечать на вопросы. Предлагать не больше трёх товаров, общая длина текста не больше 240 слов.

#Вот ассортимент нашего магазина:
#{context}

#Всегда будьте вежливы, предлагайте дополнительные товары и уточняйте детали, если нужно. Отвечайте точно и по делу, используя профессиональные знания о спортивной экипировке.
#Народная броня находится по адресу г. Донецк, пр. Театральный 15
#работает с понедельника по субботу с 9.00 до 18.00
#Воскресенье с 10.00 до 17.00
#Телефон магазина +79493061593
#Телеграм {telegram_username}"""

        system_prompt = f"""Вы - профессиональный консультант в магазине военной экипировки. Ваша задача - помогать клиентам с выбором товаров, давать профессиональные советы и отвечать на вопросы. Предлагать не больше двух товаров, общая длина текста не больше 240 слов.
        Если клиен просит показать категории товара сформировать на основе ассортимента.

        Вот ассортимент нашего магазина и госты которым соответствует наша продукция:
        {context}

        Всегда будьте вежливы, предлагайте дополнительные товары и уточняйте детали, если нужно. Отвечайте точно и по делу, используя профессиональные знания о спортивной экипировке.
        Народная броня находится по адресу г. Донецк, пр. Театральный 15
        работает с понедельника по субботу с 9.00 до 18.00
        Воскресенье с 10.00 до 17.00
        Телефон магазина +79493061593
        Телеграм {telegram_username}
        Режим разметки ответов markdownv2 telegram api bot"""


        has_system_prompt = any(msg.get('role') == 'system' for msg in messages)
        
        if not has_system_prompt:
            messages_with_system = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages_with_system = messages
        
        filtered_messages = []
        
        system_messages = [msg for msg in messages_with_system if msg.get('role') == 'system']
        filtered_messages.extend(system_messages)
        
        user_messages = [msg for msg in messages_with_system if msg.get('role') == 'user']
        if user_messages:
            last_user_message = user_messages[-1]
            filtered_messages.append(last_user_message)
        
        print("Filtered messages:", filtered_messages)
        start_time = time.time()
        
        formatted_prompt = tokenizer.apply_chat_template(
            filtered_messages,
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
        
        outputs = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        outputs = outputs.replace('**', '*')
        logger.info(f"END GENERATE ----------------> {outputs} \n End time {time.time()-start_time}")
        torch.cuda.empty_cache()
        return web.json_response({"response": outputs})
    
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return web.json_response({"error": "Generation failed"}, status=500)








######################################

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
    
    # Добавляем статический маршрут для других файлов (CSS, JS и т.д.)
    app.router.add_get('/{path:.*}', static_handler)
    
    # API endpoints
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
    ssl_context.check_hostname = False
    
    # Запуск приложения с SSL
    web.run_app(
        app,
        host="0.0.0.0",
        port=5000,
        ssl_context=ssl_context,
        access_log=logger,
        print=None
    )
    
