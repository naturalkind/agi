import aiohttp
from aiohttp import web
import ssl
import json
import uuid
import os
import logging
import asyncio
import shutil
import torch
import OpenSSL
from pathlib import Path
from datetime import datetime
import time
from werkzeug.utils import secure_filename
from typing import Dict, List, Any, Optional

# Импорт модулей OCR
from transformers import AutoModel, AutoTokenizer
import torch

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# Установка GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Конфигурация
API_KEY = os.getenv("API_KEY", "default-api-key-change-me")
UPLOAD_DIR = Path("uploads_ocr")
OUTPUT_DIR = Path("outputs_ocr")
MODEL_NAME = 'DeepSeek-OCR'

async def verify_client_cert(request: web.Request) -> bool:
    ssl_info = request.transport.get_extra_info('ssl_object')
    
    if not ssl_info or not ssl_info.getpeercert():
        raise web.HTTPForbidden(reason="Client certificate required")
        
    # Пример проверки отпечатка
    client_cert = ssl_info.getpeercert(binary_form=True)
    cert_dn = dict(x[0] for x in ssl_info.getpeercert()['subject'])
    client_cn = cert_dn.get('commonName', 'Unknown')    
    # Конвертируем DER в PEM и вычисляем SHA1
    pem_cert = ssl.DER_cert_to_PEM_cert(client_cert)
    cert = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, pem_cert)
    cert_fingerprint = cert.digest("sha1").decode().replace(":", "").lower()    
    
    print("------------>", cert_fingerprint)    
    
    expected_fingerprint = '2A420DA5F9E08CF6475EBD1C395E2456AC638193'
    
    return True

async def verify_api_key(request: web.Request) -> bool:
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        raise web.HTTPUnauthorized(reason="Invalid API key")
    return True

async def init_models(app: web.Application):
    """Инициализация OCR моделей"""
    logger.info("Initializing OCR models...")
    
    try:
        
        # Загрузка токенизатора и модели
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_NAME, 
            #_attn_implementation='flash_attention_2', rtx 30xx
            _attn_implementation='eager', # rtx 20xx
            trust_remote_code=True, 
            use_safetensors=True
        )
        model = model.eval().cuda().to(torch.bfloat16)
        
        app['models'] = {
            'tokenizer': tokenizer,
            'model': model,
            'device': torch.device("cuda")
        }
        logger.info("OCR models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize OCR models: {e}")
        raise

async def cleanup_models(app: web.Application):
    """Очистка ресурсов"""
    if 'models' in app:
        del app['models']
    torch.cuda.empty_cache()

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.task_queue = asyncio.Queue()
        self.is_processing = False
    
    def create_task(self, task_data: Dict) -> str:
        task_id = str(uuid.uuid4())
        task_data.update({
            "task_id": task_id,
            "status": "queued",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None
        })
        self.tasks[task_id] = task_data
        
        # Add task to the queue
        asyncio.create_task(self.task_queue.put(task_id))
        
        # Start queue processing if not already running
        if not self.is_processing:
            asyncio.create_task(self._process_queue())
            
        return task_id
    
    async def _process_queue(self):
        """Process tasks in the queue one after another"""
        self.is_processing = True
        
        while True:
            try:
                # Get the next task from the queue
                task_id = await self.task_queue.get()
                
                if task_id not in self.tasks:
                    self.task_queue.task_done()
                    continue
                
                # Get the application context
                app = self.tasks[task_id].get('app')
                if not app:
                    self.tasks[task_id]['status'] = 'failed'
                    self.tasks[task_id]['error'] = 'Application context not found'
                    self.task_queue.task_done()
                    continue
                
                # Process the task
                try:
                    self.tasks[task_id]['status'] = 'processing'
                    await process_ocr_task(app, task_id)
                    self.tasks[task_id]['status'] = 'completed'
                    self.tasks[task_id]['completed_at'] = datetime.now().isoformat()
                except Exception as e:
                    self.tasks[task_id]['status'] = 'failed'
                    self.tasks[task_id]['error'] = str(e)
                    self.tasks[task_id]['completed_at'] = datetime.now().isoformat()
                
                # Mark task as done in the queue
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Unexpected error in queue processing: {e}")
                await asyncio.sleep(1)  # Avoid tight loop on persistent errors
        
        self.is_processing = False

async def handle_ocr(request: web.Request) -> web.Response:
    """Обработка запроса на распознавание текста"""
    await verify_client_cert(request)
    
    data = await request.post()
    task_mgr = request.app['task_manager']
    
    # Сохранение файла изображения
    file_paths = {}
    file = data.get('image')
    if file and isinstance(file, web.FileField):
        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        save_path.write_bytes(file.file.read())
        file_paths['image'] = save_path
    else:
        raise web.HTTPBadRequest(reason="Image file is required")
    
    # Получение параметров
    params = {
        'prompt': data.get('prompt', '<image>\n<|grounding|>Convert the document to markdown. '),
        'output_path': data.get('output_path', ''),
        'base_size': int(data.get('base_size', 1024)),
        'image_size': int(data.get('image_size', 640)),
        'crop_mode': data.get('crop_mode', 'true').lower() == 'true',
        'save_results': data.get('save_results', 'false').lower() == 'true',
        'test_compress': data.get('test_compress', 'true').lower() == 'true'
    }
    
    # Создание задачи
    task_id = task_mgr.create_task({
        **file_paths,
        "params": params,
        "callback_url": data.get('callback_url'),
        "bot_type": data.get('bot_type'),
        "app": request.app
    })
    
    return web.json_response({
        "task_id": task_id,
        "status": "queued",
        "created_at": task_mgr.tasks[task_id]['created_at']
    })

async def process_ocr_task(app: web.Application, task_id: str):
    """Фоновая задача обработки OCR"""
    task = app['task_manager'].tasks[task_id]
    models = app['models']
    
    try:
        task['status'] = "processing"
        task['progress'] = 0.3
        
        # Получение параметров
        params = task['params']
        image_file = str(task['image'])
        
        # Подготовка output_path
        output_path = params.get('output_path')
        if not output_path:
            output_path = str(OUTPUT_DIR / f"{task_id}_results")
            os.makedirs(output_path, exist_ok=True)
        
        # Выполнение OCR
        task['progress'] = 0.6
        
        result = await asyncio.to_thread(
            models['model'].infer_simple,
            tokenizer=models['tokenizer'],
            prompt=params['prompt'],
            image_file=image_file
        )
        
        task['progress'] = 0.9
        
        # Сохранение результата
        if isinstance(result, str):
            extracted_text = result
        else:
            # Если результат в другом формате, преобразуем в строку
            extracted_text = str(result)
        
        
        task['result'] = extracted_text
        task['output_path'] = output_path
        task['progress'] = 1.0
        task['status'] = 'completed'
        print ("SEND_OCR_CALLBACK -----------------", extracted_text, task.get('callback_url'))
        # Отправка webhook
        #if task.get('callback_url'):
        await send_ocr_callback(task)
            
    except asyncio.CancelledError:
        logger.info(f"OCR task {task_id} cancelled")
        task.update({
            "status": "cancelled",
            "completed_at": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"OCR task {task_id} failed: {str(e)}", exc_info=True)
        task.update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
    finally:
        # Очистка GPU памяти
        torch.cuda.empty_cache()

async def send_ocr_callback(task: Dict):
    """Отправка webhook уведомления для OCR"""
    ssl_context = ssl.create_default_context(cafile='LLM/hallo/ssl/client_trust.pem') 
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        callback_data = {
            "task_id": str(task['task_id']),
            "status": task['status'],
            "progress": str(task.get('progress', 0.0)),
            "type": "ocr"
        }
        
        if task["status"] == "completed":
            callback_data.update({
                "extracted_text": task.get('result', ''),
                "output_path": task.get('output_path', '')
            })
        elif task["status"] == "failed":
            callback_data["error"] = task.get('error', 'Unknown error')
        print ("--------------SEND_OCR_CALLBACK--------------", callback_data)
        await session.post(
            "https://178.158.131.41:8443/callback",
            json=callback_data
        )

async def handle_download(request: web.Request) -> web.Response:
    """Обработчик скачивания результатов OCR"""
    await verify_api_key(request)
    task_id = request.match_info['task_id']
    task = request.app['task_manager'].tasks.get(task_id)
    
    if not task:
        raise web.HTTPNotFound(text=json.dumps({"error": "Task not found"}), 
                             content_type="application/json")

    if task['status'] != 'completed':
        raise web.HTTPBadRequest(text=json.dumps({"error": "OCR result not ready"}), 
                               content_type="application/json")

    # Возвращаем распознанный текст
    return web.json_response({
        "task_id": task_id,
        "status": "completed",
        "bot_type": task.get('bot_type', ''),
        "extracted_text": task.get('result', ''),
        "output_path": task.get('output_path', '')
    })

async def handle_status(request: web.Request) -> web.Response:
    """Получение статуса задачи OCR"""
    await verify_api_key(request)
    task_id = request.match_info['task_id']
    task = request.app['task_manager'].tasks.get(task_id)
    
    if not task:
        raise web.HTTPNotFound(reason="Task not found")
    
    response_data = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "created_at": task["created_at"]
    }
    
    if task["status"] == "completed":
        response_data["completed_at"] = task["completed_at"]
    elif task["status"] == "failed":
        response_data["error"] = task.get("error")
    
    return web.json_response(response_data)

async def handle_cancel(request: web.Request) -> web.Response:
    """Отмена задачи OCR"""
    await verify_api_key(request)
    task_id = request.match_info['task_id']
    task = request.app['task_manager'].tasks.get(task_id)
    
    if not task:
        raise web.HTTPNotFound(reason="Task not found")
    
    if task['status'] in ['completed', 'failed']:
        raise web.HTTPBadRequest(reason="Cannot cancel finished task")
    
    task['status'] = 'cancelled'
    task['completed_at'] = datetime.now().isoformat()
    
    return web.json_response({"status": "cancelled"})

async def health_check(request: web.Request) -> web.Response:
    """Проверка здоровья сервиса"""
    return web.json_response({
        "status": "ok",
        "service": "ocr",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": 'models' in request.app
    })

def init_app() -> web.Application:
    """Инициализация приложения"""
    app = web.Application(client_max_size=1024*1024*100)  # 100MB max file size
    app['task_manager'] = TaskManager()
    
    # Инициализация моделей при старте
    app.on_startup.append(init_models)
    app.on_cleanup.append(cleanup_models)
    
    # Регистрация роутов
    app.router.add_post('/ocr', handle_ocr)
    app.router.add_get('/ocr/{task_id}', handle_download)  # Получение результата
    app.router.add_get('/ocr/status/{task_id}', handle_status)
    app.router.add_delete('/ocr/cancel/{task_id}', handle_cancel)
    app.router.add_get('/health', health_check)
    
    return app

if __name__ == '__main__':
    # Создание директорий
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # SSL конфигурация
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain('LLM/hallo/ssl/server.crt', 'LLM/hallo/ssl/server.key')
    ssl_context.load_verify_locations('LLM/hallo/ssl/ca.crt')
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    # Запуск приложения
    web.run_app(init_app(), port=5001, ssl_context=ssl_context)
