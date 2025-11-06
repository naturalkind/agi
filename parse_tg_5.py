from bs4 import BeautifulSoup
import os
import re
from datetime import datetime
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob  # для анализа тональности
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

############################################
# START LLM
# ИСПРАВЛЕННЫЙ ChatAnalyzer
############################################

class ChatAnalyzer:
    def __init__(self, model_name="/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/PLAYGROUND/LLM/Vistral-24B-Instruct"):
        print(f"🔄 Загрузка модели {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            
            # Добавляем pad token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Модель успешно загружена!")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
            self.tokenizer = None
    
    def analyze_conversation_batch(self, conversations, author):
        """
        Анализирует батч диалогов с помощью LLM
        """
        if self.model is None:
            return ["LLM модель не доступна для анализа"]
        
        prompt_template = """Проанализируй эффективность сотрудника в обслуживании клиентов.

Сообщения сотрудника {author}:
{conversation_text}

Проанализируй по критериям:
1. Профессионализм и экспертиза
2. Клиентоориентированность  
3. Эффективность решения проблем
4. Коммуникативные навыки

Дайте развернутую оценку по каждому пункту и общий вывод.

Оценка: от 1 до 10"""
        analyses = []
        for i, conv in enumerate(conversations[:3]):  # Ограничиваем для скорости
            print(f"🔍 Анализ диалога {i+1}/{len(conversations[:3])} для {author}...")
            
            conversation_text = "\n".join([f"{msg['author']}: {msg['text']}" 
                                         for msg in conv])
            
            prompt = prompt_template.format(
                author=author,
                conversation_text=conversation_text
            )
            try:
                analysis = self.generate_analysis(prompt)
                analyses.append(analysis)
            except Exception as e:
                print(f"❌ Ошибка анализа диалога: {e}")
                analyses.append(f"Ошибка анализа: {str(e)}")
        print ("-"*40, analysis)
        return analyses
    
    def generate_analysis(self, prompt):
        """
        Генерирует анализ с помощью LLM
        """
        if self.model is None:
            return "LLM модель не доступна"
            
        try:
            # Токенизируем промпт
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Перемещаем на устройство модели
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Генерация с параметрами
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Декодируем ответ
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Убираем оригинальный промпт из ответа
#            if response.startswith(prompt):
#                response = response[len(prompt):].strip()
                
            return response
            
        except Exception as e:
            return f"Ошибка генерации: {str(e)}"


def advanced_llm_analysis(messages, conversation_threads):
    """
    Продвинутый анализ с использованием LLM
    """
    print("\n=== LLM АНАЛИЗ ЭФФЕКТИВНОСТИ ===")
    
    # Инициализируем анализатор
    analyzer = ChatAnalyzer()
    
    if analyzer.model is None:
        print("❌ LLM анализ недоступен - модель не загружена")
        return
    
    def normalize_author(author):
        """Приводим имена авторов к стандартному виду"""
        author_lower = author.lower()
        if 'михаил' in author_lower:
            return 'Михаил'
        elif 'евгения' in author_lower:
            return 'Евгения'
        elif any(keyword in author_lower for keyword in ['магазин', 'консультант', 'нардная', 'анастасия', 'карина', 'сергей']):
            return 'Магазин'
        elif any(keyword in author_lower for keyword in ['лиза', 'елизавета']):
            return 'Елизавета'
        return author
    
    def get_author_participation(thread, target_author):
        """Получаем сообщения целевого автора и проверяем значимость участия"""
        author_messages = [msg for msg in thread if normalize_author(msg['author']) == target_author]
        
        if len(author_messages) < 2:  # Минимум 2 сообщения
            return None
        
        # Проверяем, что есть содержательные сообщения
        meaningful_msgs = [
            msg for msg in author_messages 
            if len(msg['text'].strip()) > 15  # Более 15 символов
            and not msg['text'].startswith('@')
            and not any(word in msg['text'].lower() for word in ['ок', 'да', 'нет', 'хорошо', 'понятно'])
        ]
        
        if len(meaningful_msgs) < 1:  # Хотя бы 1 содержательное сообщение
            return None
            
        return author_messages
    
    # Собираем диалоги для каждого автора
    author_conversations = {
        'Евгения': [],
        'Михаил': [],
        'Магазин': []
    }
    
    processed_threads = set()
    
    for thread in conversation_threads:
        if len(thread) < 3:  # Пропускаем слишком короткие диалоги
            continue
            
        # Создаем уникальный идентификатор диалога
        thread_id = tuple(sorted((msg['time'], msg['author'], msg['text'][:50]) for msg in thread))
        if thread_id in processed_threads:
            continue
        processed_threads.add(thread_id)
        
        # Для каждого автора проверяем участие
        for author in author_conversations.keys():
            author_messages = get_author_participation(thread, author)
            if author_messages:
                # Сохраняем только сообщения этого автора для анализа
                author_conversations[author].append({
                    'thread_id': thread_id,
                    'messages': author_messages,
                    'total_messages': len(thread),
                    'author_messages_count': len(author_messages)
                })
    
    # Анализируем каждого автора
    llm_results = {}
    for author, conversations in author_conversations.items():
        if not conversations:
            print(f"\n⚠️ Нет подходящих диалогов для анализа {author}")
            continue
            
        print(f"\n🔍 Анализ эффективности {author}:")
        print(f"Подходящих диалогов: {len(conversations)}")
        
        # Статистика
        total_author_msgs = sum(conv['author_messages_count'] for conv in conversations)
        avg_participation = total_author_msgs / len(conversations)
        print(f"Всего сообщений {author}: {total_author_msgs}")
        print(f"Среднее участие в диалоге: {avg_participation:.1f} сообщений")
        
        # Выбираем лучшие диалоги для анализа
        # Сортируем по количеству сообщений автора и убираем дубликаты
        unique_conversations = []
        seen_content = set()
        
        for conv in sorted(conversations, key=lambda x: x['author_messages_count'], reverse=True):
            # Создаем уникальный идентификатор содержания
            content_id = tuple(sorted(msg['text'][:100] for msg in conv['messages'][:3]))
            if content_id not in seen_content:
                seen_content.add(content_id)
                unique_conversations.append(conv)
        
        # Берем только 2 самых репрезентативных диалога
        sample_conversations = unique_conversations[:2]
        
        if not sample_conversations:
            print(f"❌ Не удалось найти уникальные диалоги для {author}")
            continue
            
        print(f"Анализируем {len(sample_conversations)} наиболее репрезентативных диалогов:")
        for i, conv in enumerate(sample_conversations, 1):
            print(f"  Диалог {i}: {conv['author_messages_count']} сообщений {author} (всего в диалоге: {conv['total_messages']})")
        
        try:
            analyses = []
            for i, conv_data in enumerate(sample_conversations):
                print(f"\n🔍 Анализ диалога {i+1}/{len(sample_conversations)} для {author}...")
                # Передаем только сообщения целевого автора
                analysis = analyzer.analyze_conversation_batch(
                    [conv_data['messages']],  # Передаем только сообщения автора
                    author
                )
                if analysis and len(analysis) > 0:
                    analyses.extend(analysis)
            
            llm_results[author] = analyses
            
            # Выводим результаты
            for i, analysis in enumerate(analyses, 1):
                print(f"\n📊 Анализ диалога {i} для {author}:")
                print("=" * 60)
                if analysis:
                    # Разбиваем анализ на логические части для лучшей читаемости
                    lines = analysis.split('\n')
                    for line in lines:
                        if line.strip() and any(marker in line for marker in ['##', '###', '**', 'Оценка:', '/10']):
                            print(line)
                        elif len(line.strip()) > 50:  # Только значимые строки
                            print(line[:200] + "..." if len(line) > 200 else line)
                    print("=" * 60)
                else:
                    print("❌ Анализ не получен")
                    
        except Exception as e:
            print(f"❌ Ошибка при анализе LLM для {author}: {e}")
            llm_results[author] = [f"Ошибка: {str(e)}"]
    
    # Сохраняем результаты LLM анализа
    save_llm_results(llm_results)
    return llm_results

###################

def save_llm_results(llm_results):
    """
    Сохраняет результаты LLM анализа в файл
    """
    with open('llm_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== LLM АНАЛИЗ ЭФФЕКТИВНОСТИ СОТРУДНИКОВ ===\n\n")
        
        for author, analyses in llm_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"АНАЛИЗ ЭФФЕКТИВНОСТИ: {author}\n")
            f.write(f"{'='*60}\n\n")
            
            for i, analysis in enumerate(analyses, 1):
                f.write(f"Диалог {i}:\n")
                f.write("-" * 50 + "\n")
                f.write(analysis + "\n")
                f.write("-" * 50 + "\n\n")
    
    print("✅ Результаты LLM анализа сохранены в llm_analysis_results.txt")

def extract_key_metrics_for_llm(messages, author):
    """
    Извлекает ключевые метрики для конкретного автора
    """
    author_messages = [m for m in messages if author in m['author']]
    
    if not author_messages:
        return {}
        
    metrics = {
        'total_messages': len(author_messages),
        'avg_message_length': sum(len(m['text'].split()) for m in author_messages) / len(author_messages),
        'response_rate': 0,
        'topic_coverage': set(),
        'problem_solving_phrases': 0
    }
    
    # Анализ фраз решения проблем
    solving_phrases = ['можем', 'предлагаю', 'рекомендую', 'вариант', 'решение', 'помочь', 'помощь']
    for msg in author_messages:
        text = msg['text'].lower()
        if any(phrase in text for phrase in solving_phrases):
            metrics['problem_solving_phrases'] += 1
    
    return metrics

############################################
########### END LLM
############################################


def parse_all_telegram_files(file_list):
    """
    Парсит все указанные файлы и извлекает только прямые сообщения людей
    """
    all_human_messages = []
    
    for file_path in file_list:
        if os.path.exists(file_path):
            print(f"Парсим файл: {file_path}")
            human_messages = parse_single_file(file_path)
            all_human_messages.extend(human_messages)
        else:
            print(f"Файл {file_path} не найден, пропускаем")
    
    return all_human_messages

def parse_single_file(file_path):
    """
    Парсит один файл и возвращает только прямые сообщения людей
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        human_messages = []
        
        # Ищем все сообщения
        messages = soup.find_all('div', class_='message default clearfix')
        
        for msg in messages:
            message_data = extract_human_message(msg)
            if message_data and message_data['text'].strip():
                human_messages.append(message_data)
        
        print(f"Найдено {len(human_messages)} сообщений людей")
        return human_messages
        
    except Exception as e:
        print(f"Ошибка при парсинге {file_path}: {e}")
        return []

def extract_human_message(message_element):
    """
    Извлекает сообщения только от людей (не пересланные ботом)
    """
    try:
        # Проверяем, что это НЕ сообщение от бота
        from_name_elem = message_element.find('div', class_='from_name')
        if not from_name_elem:
            return None
            
        from_name = from_name_elem.get_text().strip()
        
        # Пропускаем сообщения от бота
        if 'NarodnayaBronya_Bot' in from_name:
            return None
        
        # Пропускаем пересланные сообщения (у них есть forwarded body)
        if message_element.find('div', class_='forwarded body'):
            return None
        
        # Извлекаем текст сообщения
        text_elem = message_element.find('div', class_='text')
        if not text_elem:
            return None
            
        text = clean_message_text(text_elem.get_text())
        
        # Извлекаем время и дату
        time_elem = message_element.find('div', class_='pull_right date details')
        time_str = time_elem.get('title', '') if time_elem else ''
        
        # Парсим дату и время
        datetime_obj = parse_datetime(time_str)
        
        return {
            'author': from_name,
            'text': text,
            'time': time_str,
            'datetime': datetime_obj,
            'is_reply': bool(message_element.find('div', class_='reply_to details')),
            'words_count': len(text.split()),
            'characters_count': len(text)
        }
        
    except Exception as e:
        print(f"Ошибка извлечения сообщения: {e}")
        return None

def parse_datetime(time_str):
    """
    Парсит дату и время из строки формата '15.05.2023 12:48:39 UTC+03:00'
    """
    try:
        if time_str:
            # Удаляем временную зону и лишние пробелы
            # Формат: "15.05.2023 12:48:39 UTC+03:00"
            datetime_part = time_str.split(' UTC')[0].strip()
            return datetime.strptime(datetime_part, '%d.%m.%Y %H:%M:%S')
    except Exception as e:
        print(f"Ошибка парсинга даты '{time_str}': {e}")
    return None

def clean_message_text(text):
    """
    Очищает текст сообщения
    """
    if not text:
        return ""
    
    # Удаляем лишние пробелы и переносы
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# НОВЫЕ ФУНКЦИИ АНАЛИЗА

def analyze_conversation_flow(messages):
    """
    Анализ последовательностей и структуры диалогов
    """
    print("\n=== АНАЛИЗ ПОСЛЕДОВАТЕЛЬНОСТИ ДИАЛОГОВ ===")
    
    # Группируем сообщения по авторам и времени
    authors = ['Евгения', 'Михаил 3D', 'Михаил 3д']
    conversation_flow = []
    
    # Сортируем сообщения по времени
    sorted_messages = sorted([m for m in messages if m['datetime']], 
                           key=lambda x: x['datetime'])
    
    # Анализ паттернов ответов
    response_times = defaultdict(list)
    conversation_threads = []
    current_thread = []
    
    for i in range(1, len(sorted_messages)):
        prev_msg = sorted_messages[i-1]
        curr_msg = sorted_messages[i]
        
        # Если разница во времени меньше 2 часов - считаем одним диалогом
        time_diff = (curr_msg['datetime'] - prev_msg['datetime']).total_seconds() / 3600
        
        if time_diff < 2:
            if not current_thread:
                current_thread.append(prev_msg)
            current_thread.append(curr_msg)
        else:
            if current_thread:
                conversation_threads.append(current_thread)
            current_thread = []
    
    if current_thread:
        conversation_threads.append(current_thread)
    
    # Статистика по диалогам
    print(f"Всего диалогов: {len(conversation_threads)}")
    
    thread_lengths = [len(thread) for thread in conversation_threads]
    if thread_lengths:
        print(f"Средняя длина диалога: {sum(thread_lengths)/len(thread_lengths):.1f} сообщений")
        print(f"Максимальная длина диалога: {max(thread_lengths)} сообщений")
    
    # Анализ инициативы в диалогах
    initiative_count = defaultdict(int)
    for thread in conversation_threads:
        if thread:
            first_author = thread[0]['author']
            initiative_count[first_author] += 1
    
    print("\nИНИЦИАТИВА В НАЧАЛЕ ДИАЛОГОВ:")
    for author, count in sorted(initiative_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {author}: {count} раз")
    
    return conversation_threads

def analyze_seasonality(messages):
    """
    Анализ сезонности активности
    """
    print("\n=== АНАЛИЗ СЕЗОННОСТИ АКТИВНОСТИ ===")
    
    dated_messages = [m for m in messages if m['datetime']]
    
    if not dated_messages:
        print("Нет данных о датах для анализа сезонности")
        return
    
    # Анализ по месяцам
    monthly_activity = defaultdict(int)
    daily_activity = defaultdict(int)
    hourly_activity = defaultdict(int)
    weekday_activity = defaultdict(int)
    
    weekdays_rus = {
        0: 'Понедельник', 1: 'Вторник', 2: 'Среда', 
        3: 'Четверг', 4: 'Пятница', 5: 'Суббота', 6: 'Воскресенье'
    }
    
    for msg in dated_messages:
        month_key = msg['datetime'].strftime('%Y-%m')
        day_key = msg['datetime'].strftime('%Y-%m-%d')
        hour_key = msg['datetime'].hour
        weekday_key = msg['datetime'].weekday()
        
        monthly_activity[month_key] += 1
        daily_activity[day_key] += 1
        hourly_activity[hour_key] += 1
        weekday_activity[weekday_key] += 1
    
    print("АКТИВНОСТЬ ПО МЕСЯЦАМ:")
    for month, count in sorted(monthly_activity.items()):
        print(f"  {month}: {count} сообщений")
    
    print("\nАКТИВНОСТЬ ПО ЧАСАМ СУТОК:")
    for hour in sorted(hourly_activity.keys()):
        count = hourly_activity[hour]
        bar = '█' * (count // max(1, max(hourly_activity.values()) // 20))
        print(f"  {hour:02d}:00 - {count:3d} сообщений {bar}")
    
    print("\nАКТИВНОСТЬ ПО ДНЯМ НЕДЕЛИ:")
    for weekday, count in sorted(weekday_activity.items()):
        day_name = weekdays_rus[weekday]
        bar = '█' * (count // max(1, max(weekday_activity.values()) // 10))
        print(f"  {day_name}: {count} сообщений {bar}")
    
    # Пиковые дни
    peak_days = sorted(daily_activity.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nСАМЫЕ АКТИВНЫЕ ДНИ:")
    for day, count in peak_days:
        date_obj = datetime.strptime(day, '%Y-%m-%d')
        weekday_name = weekdays_rus[date_obj.weekday()]
        print(f"  {day} ({weekday_name}): {count} сообщений")

def analyze_topics(messages):
    """
    Анализ тематического распределения
    """
    print("\n=== ТЕМАТИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ ===")
    
    # Ключевые слова для категорий
    topic_keywords = {
        'бронежилеты': ['бронежилет', 'жилет', 'броня', 'плит', 'бр4', 'бр5', 'бр6', 'баллистическ'],
        'разгрузки': ['разгрузк', 'подсумк', 'рмп', 'подвес', 'stich', 'molle'],
        'шлемы': ['шлем', 'каск', 'защита голов'],
        'собаки': ['собак', 'пёс', 'кобеля', 'мерк', 'размер собак'],
        'оплата': ['оплат', 'стоимост', 'цен', 'дене', 'предоплат', 'расчет'],
        'доставка': ['доставк', 'отправк', 'тк', 'транспортн', 'получен', 'забрат'],
        'технические вопросы': ['размер', 'посмотр', 'примерк', 'подойдет', 'мерк', 'параметр'],
        'приветствия': ['здравств', 'привет', 'добрый', 'здоров', 'прив'],
        'благодарности': ['спас', 'благодар', 'спасиб', 'отличн', 'хорош']
    }
    
    author_topics = defaultdict(lambda: defaultdict(int))
    all_text = ""
    
    for msg in messages:
        author = msg['author']
        text = msg['text'].lower()
        all_text += text + " "
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    author_topics[author][topic] += 1
                    break
    
    # Общая статистика по темам
    print("ОБЩАЯ СТАТИСТИКА ПО ТЕМАМ:")
    topic_totals = defaultdict(int)
    for author, topics in author_topics.items():
        for topic, count in topics.items():
            topic_totals[topic] += count
    
    for topic, count in sorted(topic_totals.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count} упоминаний")
    
    # Статистика по авторам
    print("\nТЕМАТИКА ПО АВТОРАМ:")
    for author in ['Евгения', 'Михаил 3D', 'Михаил 3д']:
        if author in author_topics:
            print(f"\n{author}:")
            topics = author_topics[author]
            total_topics = sum(topics.values())
            for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]:
                pct = (count / total_topics) * 100 if total_topics > 0 else 0
                print(f"  {topic}: {count} ({pct:.1f}%)")

def analyze_sentiment(messages):
    """
    Анализ эмоционального тона сообщений
    """
    print("\n=== АНАЛИЗ ЭМОЦИОНАЛЬНОГО ТОНА ===")
    
    # Расширенный словарь тональности
    positive_words = {
        'хорош', 'отличн', 'прекрасн', 'замечательн', 'спасиб', 'благодар', 
        'понрав', 'супер', 'отлично', 'хорошо', 'удовольств', 'рад', 'рада',
        'доволен', 'довольна', 'прекрасно', 'замечательно', 'отличный', 'хороший',
        'согласен', 'согласна', 'устроит', 'подходит', 'идеально', 'perfect'
    }
    
    negative_words = {
        'проблем', 'сложн', 'плох', 'ужасн', 'неудобн', 'дорог', 'дорого',
        'неправильн', 'ошибк', 'не', 'нет', 'не могу', 'не получается',
        'неудобно', 'плохо', 'ужасно', 'сложно', 'дорогой', 'expensive',
        'неправильно', 'ошибка', 'problem', 'issue', 'expensive'
    }
    
    author_sentiment = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
    
    for msg in messages:
        author = msg['author']
        text = msg['text'].lower()
        words = set(re.findall(r'\w+', text))  # разбиваем на слова
        
        positive_count = len(words & positive_words)
        negative_count = len(words & negative_words)
        
        if positive_count > negative_count:
            author_sentiment[author]['positive'] += 1
        elif negative_count > positive_count:
            author_sentiment[author]['negative'] += 1
        else:
            author_sentiment[author]['neutral'] += 1
    
    print("ЭМОЦИОНАЛЬНЫЙ ТОН ПО АВТОРАМ:")
    for author in ['Евгения', 'Михаил 3D', 'Михаил 3д']:
        if author in author_sentiment:
            sent = author_sentiment[author]
            total = sum(sent.values())
            if total > 0:
                pos_pct = sent['positive'] / total * 100
                neg_pct = sent['negative'] / total * 100
                neu_pct = sent['neutral'] / total * 100
                
                print(f"\n{author}:")
                print(f"  📈 Позитивных: {sent['positive']} ({pos_pct:.1f}%)")
                print(f"  📉 Негативных: {sent['negative']} ({neg_pct:.1f}%)")
                print(f"  📊 Нейтральных: {sent['neutral']} ({neu_pct:.1f}%)")
                
                # Оценка эффективности тона
                if pos_pct > 70:
                    print("  ⭐ ВЫСОКИЙ ПОЗИТИВНЫЙ ТОН - отлично!")
                elif pos_pct > 50:
                    print("  👍 ХОРОШИЙ ПОЗИТИВНЫЙ ТОН")
                elif neg_pct > 30:
                    print("  ⚠️  ВНИМАНИЕ: повышенный негативный тон")

def analyze_effectiveness(messages):
    """
    Анализ эффективности разных стилей коммуникации
    """
    print("\n=== АНАЛИЗ ЭФФЕКТИВНОСТИ СТИЛЕЙ ===")
    
    # Ключевые слова, указывающие на успешную коммуникацию
    success_indicators = {
        'заказ': ['заказ', 'оформля', 'купл', 'приобрет', 'покуп'],
        'согласие': ['соглас', 'да,', 'конечно', 'устроит', 'подходит'],
        'контакт': ['телефон', 'контакт', 'свяж', 'позвон', 'напиш'],
        'детали': ['размер', 'цвет', 'модел', 'параметр', 'характеристик'],
        'доставка': ['адрес', 'доставк', 'отправ', 'получен']
    }
    
    author_effectiveness = defaultdict(lambda: defaultdict(int))
    
    for msg in messages:
        author = msg['author']
        text = msg['text'].lower()
        
        for indicator_type, keywords in success_indicators.items():
            for keyword in keywords:
                if keyword in text:
                    author_effectiveness[author][indicator_type] += 1
                    break
    
    print("ПОКАЗАТЕЛИ ЭФФЕКТИВНОСТИ:")
    for author in ['Евгения', 'Михаил 3D', 'Михаил 3д']:
        if author in author_effectiveness:
            print(f"\n{author}:")
            effects = author_effectiveness[author]
            total_indicators = sum(effects.values())
            
            for indicator, count in sorted(effects.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_indicators * 100) if total_indicators > 0 else 0
                print(f"  {indicator}: {count} ({pct:.1f}%)")
            
            # Общая оценка эффективности
            if total_indicators > 100:
                print("  🏆 ВЫСОКАЯ ЭФФЕКТИВНОСТЬ - лидер продаж!")
            elif total_indicators > 50:
                print("  ✅ ХОРОШАЯ ЭФФЕКТИВНОСТЬ")
            elif total_indicators > 20:
                print("  📈 СРЕДНЯЯ ЭФФЕКТИВНОСТЬ")

def advanced_author_analysis(messages):
    """
    Расширенный анализ по ключевым авторам
    """
    print("\n" + "="*60)
    print("РАСШИРЕННЫЙ АНАЛИЗ КЛЮЧЕВЫХ АВТОРОВ")
    print("="*60)
    
    key_authors = ['Евгения', 'Михаил 3D', 'Михаил 3д']
    
    for author in key_authors:
        author_messages = [m for m in messages if m['author'] == author]
        if not author_messages:
            continue
            
        print(f"\n📊 {author.upper()}:")
        print(f"   📝 Всего сообщений: {len(author_messages)}")
        
        # Статистика длины сообщений
        avg_words = sum(m['words_count'] for m in author_messages) / len(author_messages)
        avg_chars = sum(m['characters_count'] for m in author_messages) / len(author_messages)
        
        print(f"   📏 Средняя длина: {avg_words:.1f} слов, {avg_chars:.1f} символов")
        
        # Процент ответов
        reply_count = sum(1 for m in author_messages if m['is_reply'])
        reply_pct = (reply_count / len(author_messages)) * 100
        print(f"   🔄 Ответов на сообщения: {reply_count} ({reply_pct:.1f}%)")
        
        # Временной анализ (если есть даты)
        dated_msgs = [m for m in author_messages if m['datetime']]
        if dated_msgs:
            dates = [m['datetime'] for m in dated_msgs]
            date_range = max(dates) - min(dates)
            days_count = max(1, date_range.days)
            msgs_per_day = len(dated_msgs) / days_count
            print(f"   📅 Сообщений в день: {msgs_per_day:.1f}")
            print(f"   🗓️  Период активности: {min(dates).strftime('%d.%m.%Y')} - {max(dates).strftime('%d.%m.%Y')}")
            print(f"   📆 Всего дней активности: {days_count}")

def save_analysis_report(messages, conversation_threads):
    """
    Сохраняет полный отчет анализа
    """
    with open('advanced_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== ПОЛНЫЙ ОТЧЕТ АНАЛИЗА ТЕЛЕГРАМ ЧАТА ===\n\n")
        
        # Базовая статистика
        f.write("БАЗОВАЯ СТАТИСТИКА:\n")
        f.write(f"Всего сообщений: {len(messages)}\n")
        
        authors = Counter(m['author'] for m in messages)
        f.write(f"Уникальных авторов: {len(authors)}\n\n")
        
        f.write("РАСПРЕДЕЛЕНИЕ ПО АВТОРАМ:\n")
        for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(messages)) * 100
            f.write(f"  {author}: {count} сообщений ({percentage:.1f}%)\n")
        
        # Анализ дат
        dated_messages = [m for m in messages if m['datetime']]
        if dated_messages:
            dates = [m['datetime'] for m in dated_messages]
            f.write(f"\nПЕРИОД АНАЛИЗА: {min(dates).strftime('%d.%m.%Y')} - {max(dates).strftime('%d.%m.%Y')}\n")
            f.write(f"ДНЕЙ АКТИВНОСТИ: {(max(dates) - min(dates)).days}\n")
        
        # Расширенная статистика
        f.write("\n" + "="*50 + "\n")
        f.write("РЕЗУЛЬТАТЫ РАСШИРЕННОГО АНАЛИЗА\n")
        f.write("="*50 + "\n\n")
        
        # Сохраняем ключевые метрики
        f.write("КЛЮЧЕВЫЕ МЕТРИКИ ЭФФЕКТИВНОСТИ:\n")
        f.write("• Анализ последовательности диалогов\n")
        f.write("• Сезонность активности\n") 
        f.write("• Тематическое распределение\n")
        f.write("• Эмоциональный тон\n")
        f.write("• Показатели эффективности коммуникации\n")
        
    print(f"\n✅ Полный отчет сохранен в advanced_analysis_report.txt")

def save_human_messages(messages, output_file='human_messages.txt'):
    """
    Сохраняет сообщения людей
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== СООБЩЕНИЯ ЛЮДЕЙ ИЗ ТЕЛЕГРАМ ЧАТА ===\n\n")
        
        for i, msg in enumerate(messages, 1):
            f.write(f"Сообщение {i}:\n")
            f.write(f"Автор: {msg['author']}\n")
            f.write(f"Время: {msg['time']}\n")
            if msg['is_reply']:
                f.write("(ответ на сообщение)\n")
            f.write(f"Текст: {msg['text']}\n")
            f.write("-" * 80 + "\n")
    
    print(f"Сохранено {len(messages)} сообщений людей в файл {output_file}")

def save_text_only(messages, output_file='human_messages_text_only.txt'):
    """
    Сохраняет только текст сообщений (для LLM контекста)
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== ТЕКСТ СООБЩЕНИЙ ЛЮДЕЙ ДЛЯ LLM ===\n\n")
        
        for i, msg in enumerate(messages, 1):
            f.write(f"{msg['text']}\n")
            if i % 10 == 0:  # Разделитель каждые 10 сообщений
                f.write("\n" + "="*50 + "\n\n")
    
    print(f"Сохранено {len(messages)} текстов сообщений в файл {output_file}")

# Основной код выполнения
if __name__ == "__main__":
    files_to_parse = [
        'messages.html', 'messages2.html', 'messages3.html', 
        'messages4.html', 'messages5.html', 'messages6.html'
    ]
    
    # Парсим все файлы
    all_human_messages = parse_all_telegram_files(files_to_parse)
    
    if all_human_messages:
        # Сохраняем полную информацию
        save_human_messages(all_human_messages)
        
        # Сохраняем только текст для LLM
        save_text_only(all_human_messages)
        
        # ЗАПУСКАЕМ РАСШИРЕННЫЙ АНАЛИЗ
        print("\n" + "="*60)
        print("ЗАПУСК РАСШИРЕННОГО АНАЛИЗА")
        print("="*60)
        
        # 1. Анализ последовательностей диалогов
        conversation_threads = analyze_conversation_flow(all_human_messages)
        
        # 2. Запускаем LLM анализ (НОВОЕ!)
        llm_results = advanced_llm_analysis(all_human_messages, conversation_threads)
        
        # 3. Анализ сезонности активности
        analyze_seasonality(all_human_messages)
        
        # 4. Тематическое распределение
        analyze_topics(all_human_messages)
        
        # 5. Анализ эмоционального тона
        analyze_sentiment(all_human_messages)
        
        # 6. Анализ эффективности стилей
        analyze_effectiveness(all_human_messages)
        
        # 7. Расширенный анализ авторов
        advanced_author_analysis(all_human_messages)
        
        # Сохраняем полный отчет
        save_analysis_report(all_human_messages, conversation_threads)
        
        # Базовая статистика
        authors = Counter(m['author'] for m in all_human_messages)
        print(f"\n=== ИТОГИ ===")
        print(f"Всего сообщений людей: {len(all_human_messages)}")
        print(f"Уникальных авторов: {len(authors)}")
        
        # Статистика по датам
        dated_messages = [m for m in all_human_messages if m['datetime']]
        if dated_messages:
            dates = [m['datetime'] for m in dated_messages]
            print(f"Период анализа: {min(dates).strftime('%d.%m.%Y')} - {max(dates).strftime('%d.%m.%Y')}")
            print(f"Дней активности: {(max(dates) - min(dates)).days}")
        
    else:
        print("Не найдено сообщений людей")
