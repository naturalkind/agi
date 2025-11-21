# LLM

Подключение искуственных нейросетевых моделей к `telegram bot api`, для 
создания персонального помошника с множеством функций. [90% кода сгенерировано языковыми моделями](https://habr.com/ru/articles/881944/) Примеры, бот 🤖 https://t.me/digital_ark

> nvidia gpu python 3.10.0

```
pip install -r requirements-cuda.txt
python start_bot_app_cuda_v3.py
```
> intel xpu python 3.11.0

```
pip install -r requirements-intel.txt
python start_bot_app_intel_v3.py
```

### Технологии:
- LLM модель поддерживаемая transformers 
- Whisper распознавание речи https://huggingface.co/openai/whisper-large-v3
- XTTS синтез голоса https://github.com/coqui-ai/TTS & https://huggingface.co/coqui/XTTS-v2
- Hallo анимация портретных изображений https://github.com/fudan-generative-vision/hallo
- Helsinki-NLP перевод с русского на английский https://huggingface.co/Helsinki-NLP/opus-mt-en-ru
- Dreamshaper облегчённая Stable-Diffusion 1.5 для intel GPU https://huggingface.co/OpenVINO/LCM_Dreamshaper_v7-int8-ov
- DeepSeek OCR https://huggingface.co/deepseek-ai/DeepSeek-OCR

### Системные требования:
Для запуска всех функций одновременно в минимальной конфигурации необходимо 40gb видео памяти. Пример
запуска на нескольких компьютерах с gpu разных производителей nvidia rtx 3090 24gb и intel arc a770 16gb

> nvidia gpu: Hallo

```
python web_app_aiohttp.py 
```
> intel xpu: Helsinki-NLP, Dreamshaper, LLM

```
python llm_fastapi.py
```
> intel xpu: Whisper, XTTS

```
python llm_hub_server.py
```
> nvidia gpu: DeepSeek OCR
```
python server_ocr_aiohttp.py
```
![Иллюстрация к проекту](https://github.com/naturalkind/agi/blob/v0.1/media/example.png)

### Нужно сделать
- [x] генерация изображения   
- [x] анимация изображения   
- [x] ансамбль llm моделей   
- [x] параллельная работа GPU, распределённые вычисления   
- [x] очередь задач баланс между участниками   
- [x] анимации лица на изображении с помощью голоса   
- [x] reasoning | rag | rl   
- [x] перевод голоса в текст с дальнейшей генерацией текста   
- [x] языковая модель чат бот   
- [x] отображение выполнения задач пользователю   
- [x] OCR
- [x] обработка PDF   
- [x] переводчик (8 из 10 😳)   


