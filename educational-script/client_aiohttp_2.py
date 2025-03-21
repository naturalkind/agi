import asyncio
import time
import ssl
import uuid
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import logging
import aiohttp

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("video_gen_client")

class VideoGenClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        client_cert: str,
        client_key: str,
        ca_cert: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client_cert = client_cert
        self.client_key = client_key
        self.ca_cert = ca_cert
        self.ssl_context = self._create_ssl_context()

        # Проверка существования файлов сертификатов
        self._validate_paths()

    def _validate_paths(self):
        if not Path(self.client_cert).exists():
            raise FileNotFoundError(f"Client cert not found: {self.client_cert}")
        if not Path(self.client_key).exists():
            raise FileNotFoundError(f"Client key not found: {self.client_key}")
        if self.ca_cert and not Path(self.ca_cert).exists():
            raise FileNotFoundError(f"CA cert not found: {self.ca_cert}")

    def _create_ssl_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_cert_chain(self.client_cert, self.client_key)
        if self.ca_cert:
            context.load_verify_locations(self.ca_cert)
            context.verify_mode = ssl.CERT_REQUIRED
        else:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context

    async def _request(
        self, method: str, endpoint: str, **kwargs
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = kwargs.get("headers", {})
        headers.update({"X-API-Key": self.api_key})

        # Явно создаем SSL-контекст, если он не был передан ранее
        ssl_context = kwargs.get('ssl', self.ssl_context)
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_context)
        ) as session:
            try:
                async with session.request(
                    method, url, headers=headers, ssl=ssl_context, **kwargs
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientResponseError as e:
                error_text = await response.text()
                logger.error(f"HTTP error: {e.status} - {error_text}")
                raise
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                raise

    async def generate_video(
        self,
        audio_path: str,
        image_path: str,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        data = aiohttp.FormData()
        data.add_field("audio", open(audio_path, "rb"), filename="audio.wav")
        data.add_field("image", open(image_path, "rb"), filename="image.png")
        data.add_field("video_params", json.dumps(params or {}))
        data.add_field("callback_url", params["callback_url"])
        return await self._request(
            "POST",
            "/generate_video",
            data=data,
        )

    async def download_video(self, task_id: str, save_path: str) -> None:
        print (f"------------->{self.base_url}/download/{task_id}<------------", "\n", save_path, self.ssl_context)
        headers = {"X-API-Key": self.api_key}
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=self.ssl_context)
        ) as session:
            async with session.get(f"{self.base_url}/download/{task_id}", headers=headers) as response:
                response.raise_for_status()
                with open(save_path, "wb") as f:
                    f.write(await response.read())
        
        logger.info(f"Video saved to: {save_path}")

    async def get_status(self, task_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/status/{task_id}")

    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/cancel/{task_id}")

    async def health_check(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")

class TaskProcessingError(Exception):
    """Исключение для ошибок обработки задачи."""
    pass

class TaskTimeoutError(Exception):
    """Исключение для таймаутов задачи."""
    pass

async def wait_for_completion(
    client,
    task_id: str,
    check_interval: float = 5.0,
    initial_interval: float = 1.0,
    max_interval: float = 30.0,
    timeout: Optional[float] = 3600.0,  # 1 час по умолчанию
    backoff_factor: float = 1.5,
    max_retries: int = 3,
    retry_statuses: list = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Ожидает завершения задачи с адаптивным интервалом проверки и обработкой ошибок.
    
    Args:
        client: Экземпляр VideoGenClient
        task_id: ID задачи для проверки
        check_interval: Начальный интервал между проверками (сек)
        initial_interval: Минимальный интервал между проверками (сек)
        max_interval: Максимальный интервал между проверками (сек)
        timeout: Общий таймаут ожидания (сек, None для бесконечного ожидания)
        backoff_factor: Множитель для увеличения интервала ожидания
        max_retries: Максимальное число повторных попыток при временных ошибках
        retry_statuses: Список статусов задачи, при которых нужно повторить запрос
        progress_callback: Функция обратного вызова для отображения прогресса
    
    Returns:
        Dict с окончательным статусом задачи
        
    Raises:
        TaskProcessingError: При ошибке обработки задачи
        TaskTimeoutError: При превышении времени ожидания
        aiohttp.ClientError: При проблемах с сетевым соединением
    """
    if retry_statuses is None:
        retry_statuses = ["error", "failed"]
    
    start_time = time.time()
    current_interval = initial_interval
    retries = 0
    last_progress = None
    
    logger.info(f"Начинаем ожидание завершения задачи {task_id}")
    
    while True:
        try:
            status = await client.get_status(task_id)
            
            # Обработка различных статусов
            if status["status"] == "completed":
                logger.info(f"Задача {task_id} успешно завершена")
                return status
            
            elif status["status"] == "processing":
                # Обработка прогресса, если доступен
                current_progress = status.get("progress", None)
                if current_progress != last_progress:
                    last_progress = current_progress
                    logger.info(f"Прогресс задачи {task_id}: {current_progress}%")
                    if progress_callback:
                        progress_callback(status)
                
                # Динамически регулируем интервал ожидания
                # Если прогресс движется медленно, увеличиваем интервал
                if current_progress and last_progress:
                    progress_delta = current_progress - last_progress
                    if progress_delta < 5 and current_interval < max_interval:
                        current_interval = min(current_interval * backoff_factor, max_interval)
                
                # Сбрасываем счетчик повторов при нормальном статусе
                retries = 0
                
            elif status["status"] in retry_statuses:
                retries += 1
                logger.warning(f"Получен статус {status['status']} для задачи {task_id}, попытка {retries}/{max_retries}")
                
                if retries >= max_retries:
                    detailed_error = status.get("error", "Неизвестная ошибка")
                    raise TaskProcessingError(f"Задача {task_id} завершилась с ошибкой после {max_retries} попыток: {detailed_error}")
            
            elif status["status"] == "cancelled":
                raise TaskProcessingError(f"Задача {task_id} была отменена")
            
            elif status["status"] == "error":
                detailed_error = status.get("error", "Неизвестная ошибка")
                raise TaskProcessingError(f"Задача {task_id} завершилась с ошибкой: {detailed_error}")
            
        except aiohttp.ClientError as e:
            retries += 1
            logger.warning(f"Сетевая ошибка при проверке статуса задачи {task_id}: {e}, попытка {retries}/{max_retries}")
            
            if retries >= max_retries:
                raise
            
            # Увеличиваем интервал при сетевых ошибках
            current_interval = min(current_interval * backoff_factor, max_interval)
        
        # Проверка таймаута
        elapsed_time = time.time() - start_time
        if timeout is not None and elapsed_time > timeout:
            time_str = f"{elapsed_time:.1f} сек"
            if elapsed_time > 60:
                time_str = f"{elapsed_time/60:.1f} мин"
            raise TaskTimeoutError(f"Превышено время ожидания ({time_str}) для задачи {task_id}")
        
        # Вывод периодической информации о статусе
        elapsed_min = elapsed_time / 60
        next_check_time = datetime.now().strftime("%H:%M:%S")
        logger.debug(f"Задача {task_id} ещё выполняется (прошло {elapsed_min:.1f} мин). Следующая проверка в {next_check_time}")
        
        # Ожидаем до следующей проверки
        await asyncio.sleep(current_interval)

async def main():
    # Настройки (замените на реальные пути и значения)
    CLIENT_CERT = "ssl/client.crt"
    CLIENT_KEY = "ssl/client.key"
    CA_CERT = "ssl/ca.crt"  # Если используется самоподписанный сертификат сервера
    API_KEY = "default-api-key-change-me"
    BASE_URL = "https://192.168.1.50:5000"
    
    client = VideoGenClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        client_cert=CLIENT_CERT,
        client_key=CLIENT_KEY,
        ca_cert=CA_CERT,
    )
    
    # Проверка здоровья
    health = await client.health_check()
    logger.info(f"Health check: {health}")
    
    # Генерация видео
    task = await client.generate_video(
        audio_path="/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/hallo/examples/driving_audios/1.wav",
#        image_path="/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/FILE_APP/IMAGE/2402_c0a02d4e-de6.png", # 522_2b0fc29b-05c.jpeg, 164_f84a0540-726.jpeg
        image_path="/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/FILE_APP/IMAGE/164_f84a0540-726.jpeg",
        params={
            "callback_url": "https://192.168.1.60:5000",
            "pose_weight": 1.0,
            "face_weight": 1.0,
            "lip_weight": 1.0,
            "face_expand_ratio": 1.0
        }
    )
    logger.info(f"Task created: {task}")
    task_id = task["task_id"]

    # Определяем функцию для отображения прогресса
    def show_progress(status):
        progress = status.get("progress", 0)
        print(f"\rПрогресс генерации видео: [{progress}%] {'#' * int(progress/2)}{' ' * (50-int(progress/2))}", end="")
    
    try:
        # Ожидаем завершения с таймаутом 30 минут и отображением прогресса
        final_status = await wait_for_completion(
            client, 
            task_id, 
            timeout=1800.0*4,  # 30 минут
            progress_callback=show_progress
        )
        
        print("\n")  # Новая строка после прогресс-бара
        logger.info(f"Задача успешно завершена. Статус: {final_status}")
        
        # Скачивание видео (после завершения)
        if final_status["status"] == "completed":
            output_path = f"result_{task_id}.mp4"
            await client.download_video(task_id, output_path)
            logger.info(f"Видео успешно скачано и сохранено как {output_path}")
    
    except TaskTimeoutError as e:
        logger.error(f"Превышено время ожидания: {e}")
        # Здесь можно добавить логику для отправки уведомления
    
    except TaskProcessingError as e:
        logger.error(f"Ошибка обработки задачи: {e}")
        # Здесь можно добавить логику обработки ошибок, например повторную отправку
    
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка сетевого соединения: {e}")
        # Здесь можно добавить логику обработки сетевых ошибок
    
    except Exception as e:
        logger.exception(f"Непредвиденная ошибка: {e}")
    
    logger.info("Завершение работы клиента")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
