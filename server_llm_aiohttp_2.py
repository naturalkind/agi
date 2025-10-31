import aiohttp
from aiohttp import web
import asyncio
import json
import os
import glob
import pickle
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import ssl
from datetime import datetime

# Импортируем ваши существующие модули
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, pipeline
import torch
from bs4 import BeautifulSoup
from threading import Thread
from PIL import Image
from ragutil import RAGAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация модели (ваш существующий код)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"#, 2,3"
model_id = "/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/Vistral-24B-Instruct"  # Уточните точное название репозитория

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

## Получение данных из телеграмм
#products = get_context_full()
## Добавление в индекс с метаданными
#metadatas = [{"source": "telegram_export", "product_id": i+1} for i in range(len(products))]
#result = rag.add_texts(products, metadatas)
#print(result["message"], len(products), type(products))


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

async def generate_handler(request):
    try:
        data = await request.json()
        messages = data.get('messages', [])
        max_new_tokens = data.get('max_new_tokens', 500)
        temperature = data.get('temperature', 0.7)
        last_n_messages = data.get('last_n_messages', 3)  # По умолчанию только последнее сообщение
        
        # Извлекаем последние N сообщений пользователя для поиска
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        if user_messages:
            # Берем последние N сообщений
            recent_user_messages = user_messages[-last_n_messages:]
            # Объединяем содержимое последних N сообщений для поиска
            query = " ".join([msg.get('content', '') for msg in recent_user_messages])
        else:
            query = ""
            
        context = rag.search_documents(query, k=5)
        logger.info(f"MAX_NEW_TOKENS --> {max_new_tokens}")
        #logger.info(f"GENERATE --> {messages}")
        #logger.info(f"CONTEXT --> {context}")
        logger.info(f"USING LAST {last_n_messages} USER MESSAGES")
        telegram_username = "NaodnayaBronya_Bot"

#        system_prompt = f"""Вы - профессиональный консультант в магазине военной экипировки. Ваша задача - помогать клиентам с выбором товаров, давать профессиональные советы и отвечать на вопросы. Предлагать не больше двух товаров, общая длина текста не больше 300 слов.
#        Если клиен просит показать категории товара сформировать на основе ассортимента.

#        Вот ассортимент нашего магазина и госты которым соответствует наша продукция:
#        {context}

#        Всегда будьте вежливы, предлагайте дополнительные товары и уточняйте детали, если нужно. Отвечайте точно и по делу, используя профессиональные знания о спортивной экипировке.
#        Народная броня находится по адресу г. Донецк, пр. Театральный 15
#        работает с понедельника по субботу с 9.00 до 18.00
#        Воскресенье с 10.00 до 17.00
#        Телефон магазина +79493061593
#        Телеграм {telegram_username}
#        """ #Режим разметки ответов markdownv2 telegram api bot



#######################
        system_prompt = f"""Вы - юридический консультант, задача изучить правомерность решения.
        Используешь порядок рассмотрения отдельных обращений ФЗ-59 "О порядке рассмотрения обращений граждан РФ":
        Основная суть: Закон регулирует, как государственные органы, органы местного самоуправления и их должностные лица должны рассматривать обращения граждан.
        Ключевые положения:
        Виды обращений:
        Предложение – рекомендация по улучшению законов, работы госорганов и т.д.
        Заявление – просьба о содействии в реализации прав или сообщение о нарушениях.
        Жалоба – просьба о восстановлении нарушенных прав.
        Формы обращений: Письменная (включая бумажную и электронную через Единый портал госуслуг) и устная (на личном приеме).
        Обязательные реквизиты письменного обращения:
        ФИО заявителя, почтовый адрес для ответа.
        Для электронного обращения: ФИО и адрес электронной почты или ID в личном кабинете на портале госуслуг.
        Суть обращения.
        Личная подпись и дата (для письменного).
        Сроки рассмотрения:
        30 дней – стандартный срок.
        В исключительных случаях срок может быть продлен еще на 30 дней с уведомлением заявителя.
        Права гражданина:
        Представлять дополнительные документы.
        Знакомиться с материалами рассмотрения (если нет ограничений).
        Получать письменный ответ по существу.
        Обжаловать решение.
        Подать заявление о прекращении рассмотрения обращения.
        Важные гарантии:
        Рассмотрение обращений – бесплатно.
        Запрещено преследование за критику в обращении.
        Запрещено разглашение сведений из обращения без согласия заявителя.
        Основания для отказа в рассмотрении по существу (ответ не дается):
        Анонимное обращение (кроме сообщений о преступлениях).
        Текст нечитаем или не позволяет понять суть.
        Содержит нецензурную брань, оскорбления, угрозы.
        Вопрос уже многократно отвечен, а новых доводов нет (прекращение переписки).
        Обжалуется судебное решение (разъясняется порядок его обжалования).
        права не должны нарушиться при отказе учитывая законы....."
        Законы для анализа:
        {context}

        """ #Режим разметки ответов markdownv2 telegram api bot

##########################

        has_system_prompt = any(msg.get('role') == 'system' for msg in messages)
        
        if not has_system_prompt:
            messages_with_system = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages_with_system = messages
        
        filtered_messages = []
        
        # Всегда добавляем системные сообщения
        system_messages = [msg for msg in messages_with_system if msg.get('role') == 'system']
        filtered_messages.extend(system_messages)
        
        # Обрабатываем пользовательские и ассистентские сообщения
        non_system_messages = [msg for msg in messages_with_system if msg.get('role') != 'system']
        
        # Берем последние N сообщений (включая как пользователей, так и ассистентов)
        recent_messages = non_system_messages[-last_n_messages * 2:]  # Умножаем на 2, т.к. диалог обычно user-assistant пары
        
        # Добавляем отфильтрованные сообщения, сохраняя порядок
        filtered_messages.extend(recent_messages)
        
        print(f"Filtered messages (last {last_n_messages}):", filtered_messages)
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

# Инициализация RAG
rag = RAGAnalyzer()
rag.load_index("savedata_rag")

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
    
