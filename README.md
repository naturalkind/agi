# LLM

Подключение искуственных нейросетевых моделей к `telegram bot api`, для 
создания персонального помошника с множеством функций. [90% кода сгенерировано языковыми моделями](https://habr.com/ru/articles/881944/) 

> nvidia gpu python 3.10.0

```
pip install -r requirements-cuda.txt
start_bot_app_cuda_v3.py
```
> intel xpu python 3.11.0

```
pip install -r requirements-intel.txt
start_bot_app_intel_v3.py
```

### Технологии:
- LLM модель поддерживаемая transformers 
- Whisper распознавание речи https://huggingface.co/openai/whisper-large-v3
- XTTS синтез голоса https://github.com/coqui-ai/TTS & https://huggingface.co/coqui/XTTS-v2
- Hallo анимация портретных изображений https://github.com/fudan-generative-vision/hallo

![Иллюстрация к проекту](https://github.com/naturalkind/agi/blob/v0.1/media/example.png)

### Нужно сделать
- [ ] генерация изображения   
- [x] анимация изображения   
- [x] ансамбль llm моделей   
- [x] параллельная работа GPU, распределённые вычисления   
- [x] очередь задач баланс между участниками   
- [x] анимации лица на изображении с помощью голоса   
- [ ] reasoning | rag | rl   
- [x] перевод голоса в текст с дальнейшей генерацией текста   
- [x] языковая модель чат бот   
- [x] отображение выполнения задач пользователю   
- [ ] обработка PDF   
- [ ] переводчик   


