# LLM

подключение LLM модели к `telegram bot api`, реализована простая очередь обработки запросов в GPU
> nvidia gpu python 3.10.0

```
pip install -r requirements-cuda.txt
python start_bot_app_v1.py
```
> intel xpu python 3.11.0

```
pip install -r requirements-intel.txt
python start_bot_app_v3_intel.py
```

### Технологии:
- LLM модель поддерживаемая transformers
- Whisper для распознавания речи https://huggingface.co/openai/whisper-large-v3
- XTTS для синтеза голоса https://github.com/coqui-ai/TTS & https://huggingface.co/coqui/XTTS-v2

![Иллюстрация к проекту](https://github.com/naturalkind/agi/blob/v0.1/media/example.png)

### Нужно сделать
- [ ] генерация изображения   
- [ ] анимация изображения   
- [ ] ансамбль llm моделей   
- [ ] параллельная работа GPU, распределённые вычисления   
- [ ] очередь задач баланс между участниками   
- [x] анимации лица на изображении с помощью голоса   
- [ ] rag   
- [ ] reasoning   
- [x] перевод голоса в текст с дальнейшей генерацией текста   
- [x] языковая модель чат бот   
- [x] отображение выполнения задач пользователю   
- [ ] обработка PDF   
- [ ] переводчик   


