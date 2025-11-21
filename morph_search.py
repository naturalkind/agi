import pymorphy3
import re
import logging
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import json
from dataclasses import dataclass
from functools import lru_cache
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    category: str
    confidence: float
    matched_keywords: List[str]
    method: str

class AdvancedProductClassifier:
    """
    Продвинутый классификатор продукции с поддержкой морфологического анализа,
    весов ключевых слов и контекстного поиска
    """
    
    def __init__(self, catalog_path: Optional[str] = None):
        self.morph = pymorphy3.MorphAnalyzer()
        self.product_catalog = self._load_catalog(catalog_path)
        self.keyword_weights = self._initialize_keyword_weights()
        self._patterns_cache = {}
        self._compiled_patterns = {}
        
        # Минимальный порог уверенности для классификации
        self.confidence_threshold = 0.3
        
        # Статистика классификации
        self.stats = {
            'total_classifications': 0,
            'catalog_matches': 0,
            'keyword_matches': 0,
            'fallback_classifications': 0,
            'avg_processing_time': 0
        }
    
    def _load_catalog(self, catalog_path: Optional[str]) -> Dict[str, List[str]]:
        """Загружает каталог продукции из файла или использует стандартный"""
        if catalog_path:
            try:
                with open(catalog_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить каталог из {catalog_path}: {e}")
        
        # Стандартный каталог продукции
        return {
            'Бронежилеты': [
                'бронежилет', 'бронежилеты', 'бронежилета', 'бронежилетов',
                'защитный жилет', 'бронезащита', 'бронежилетка', 'бронь'
            ],
            'Модули_защиты': [
                'модуль защиты', 'наплечник', 'напашник', 'защита шеи',
                'защита бедра', 'защита живота', 'наручи', 'поножи'
            ],
            'Разгрузочные_системы': [
                'разгрузочная система', 'разгрузочный жилет', 'тактический жилет',
                'подвесная система', 'боевая система'
            ],
            'Бронеэлементы': [
                'бронеплита', 'бронепанель', 'бронепластина', 'баллистическая плита',
                'керамическая плита', 'композитная плита'
            ]
        }
    
    def _initialize_keyword_weights(self) -> Dict[str, Dict[str, float]]:
        """Инициализирует веса ключевых слов для каждой категории"""
        return {
            'Бронежилеты': {
                'бронежилет': 1.0, 'бронежилеты': 1.0, 'бронежилета': 1.0,
                'ватник': 0.9, 'плитник': 0.9, 'защитный жилет': 0.95,
                'бронезащита': 0.8, 'бронежилетка': 0.7, 'бронежилетом': 1.0,
                'бронежилетами': 1.0, 'жилет защитный': 0.95, 'бронь':0.85
            },
            'Модули_защиты': {
                'модуль защиты': 1.0, 'наплечник': 0.9, 'напашник': 0.9,
                'защита шеи': 0.95, 'защита бедра': 0.95, 'защита живота': 0.95,
                'наручи': 0.85, 'поножи': 0.85, 'пах': 0.8, 'шейный модуль': 0.9,
                'защита голени': 0.85, 'защита предплечья': 0.85, 'маска': 0.7,
                'наплечники': 0.9, 'напашники': 0.9, 'наруч': 0.85, 'понег': 0.85
            },
            'Разгрузочные_системы': {
                'разгрузочный пояс': 1.0, 'рпс': 0.8, 'пояс ронин': 0.9,
                'тактический пояс': 0.95, 'разгрузочная система': 1.0,
                'боевой пояс': 0.9, 'разгрузка': 0.7, 'тактический жилет': 0.85,
                'подвесная система': 0.9, 'разгрузочный жилет': 0.9
            },
            'Комплект': {
                'комплект': 1.0, 'набор': 0.9, 'комплектация': 0.8,
                'сборка': 0.7, 'готовый набор': 0.95, 'комплектом': 1.0,
                'набором': 0.9, 'комплектации': 0.8, 'комплекты': 1.0,
                'наборы': 0.9
            },
            'Подсумки': {
                'подсумок': 1.0, 'аптечка': 0.8, 'кобура': 0.7, 'патронташ': 0.8,
                'сумка': 0.6, 'чехол': 0.5, 'медицинский': 0.7,
                'подсумок для рации': 0.9, 'подсумок штурмовой': 0.9,
                'мародёрка': 0.8, 'вытяжной подсумок': 0.9, 'подсумки': 1.0,
                'аптечки': 0.8, 'кобуры': 0.7, 'патронташи': 0.8
            },
            'Бронеэлементы': {
                'бронеплита': 1.0, 'бронеплит': 1.0, 'кап': 0.7, 'свмпэ': 0.8,
                'баллистический пакет': 0.9, 'броневставка': 0.95,
                'бронепанель': 0.95, 'бронепластина': 0.95, 'бронеплиты': 1.0,
                'бр4': 0.6, 'бр5': 0.6, 'керамико-композитный': 0.8,
                'бронеэлемент': 1.0, 'баллистическая вставка': 0.9,
                'защитная плита': 0.9, 'бронепластины': 0.95, 'баллистика': 0.85
            },
            'Специальное_снаряжение': {
                'носилки': 0.9, 'бронещит': 0.95, 'бронекостюм': 0.9,
                'для собак': 0.7, 'противоосколочный костюм': 0.95,
                'саперный': 0.8, 'разминирование': 0.7, 'противоосколочный щит': 0.95,
                'упор стксс': 0.6, 'бронекостюмы': 0.9, 'бронещиты': 0.95,
                'саперные': 0.8, 'носилки тактические': 0.9
            },
            'Аксессуары': {
                'шеврон': 0.8, 'упор': 0.6, 'крепление': 0.7, 'инструмент': 0.5,
                'сброс': 0.6, 'сброс магазина': 0.8, 'модуль быстрого сброса': 0.9,
                'шевроны': 0.8, 'упоры': 0.6, 'крепления': 0.7, 'инструменты': 0.5
            },
            'Сувениры': {
                'сувенир': 1.0, 'памятный': 0.8, 'символика': 0.7, 'монумент': 0.6,
                'сувениры': 1.0, 'сувенирный': 0.9, 'памятные': 0.8,
                'символики': 0.7, 'монументы': 0.6
            }
        }
    
    @lru_cache(maxsize=1000)
    def _get_word_forms(self, word: str) -> Set[str]:
        """Генерирует различные формы слова для поиска с кэшированием"""
        forms = set()
        
        try:
            parsed = self.morph.parse(word)[0]
            
            # Добавляем нормальную форму
            forms.add(parsed.normal_form)
            
            # Добавляем исходное слово
            forms.add(word)
            
            # Генерируем различные формы для существительных и прилагательных
            if parsed.tag.POS in ['NOUN', 'ADJF', 'ADJS']:
                cases = ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct']
                numbers = ['sing', 'plur']
                
                for case in cases:
                    for number in numbers:
                        form = parsed.inflect({case, number})
                        if form and form.word != word:
                            forms.add(form.word)
            
            # Для глаголов используем лексемы
            elif parsed.tag.POS in ['VERB', 'INFN']:
                for lexeme in parsed.lexeme:
                    if lexeme.word != word:
                        forms.add(lexeme.word)
                        
        except Exception as e:
            logger.debug(f"Ошибка при обработке слова '{word}': {e}")
            forms.add(word)  # В случае ошибки используем только исходное слово
        
        return forms
    
    def _prepare_search_patterns(self, keywords: List[str]) -> Set[str]:
        """Подготавливает поисковые паттерны со всеми формами слов"""
        search_patterns = set()
        
        for keyword in keywords:
            # Если ключевое слово состоит из нескольких слов
            if ' ' in keyword:
                words = keyword.split()
                word_forms = [self._get_word_forms(word) for word in words]
                
                # Генерируем комбинации форм слов
                from itertools import product
                for combination in product(*word_forms):
                    phrase = ' '.join(combination)
                    search_patterns.add(phrase)
                    
                    # Также добавляем варианты с разными разделителями
                    search_patterns.add(phrase.replace(' ', '-'))
                    search_patterns.add(phrase.replace(' ', '_'))
            else:
                search_patterns.update(self._get_word_forms(keyword))
        
        return search_patterns
    
    def _preprocess_text(self, text: str) -> str:
        """Предварительная обработка текста"""
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Заменяем различные разделители на пробелы
        text = re.sub(r'[_\-\.,;:!?]', ' ', text)
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_confidence(self, matched_keywords: List[str], category: str) -> float:
        """Рассчитывает уверенность в классификации на основе весов ключевых слов"""
        if not matched_keywords:
            return 0.0
        
        total_weight = 0.0
        max_possible_weight = sum(sorted([
            weight for keyword, weight in self.keyword_weights.get(category, {}).items()
        ], reverse=True)[:3])  # Берем 3 самых весомых ключевых слова
        
        for keyword in matched_keywords:
            weight = self.keyword_weights.get(category, {}).get(keyword, 0.5)
            total_weight += weight
        
        # Нормализуем уверенность
        confidence = min(total_weight / max(max_possible_weight, 1), 1.0)
        
        # Повышаем уверенность при множественных совпадениях
        if len(matched_keywords) > 1:
            confidence = min(confidence * (1 + 0.1 * (len(matched_keywords) - 1)), 1.0)
        
        return confidence
    
    def _classify_by_catalog(self, text: str) -> Optional[ClassificationResult]:
        """Классификация по точному соответствию с каталогом"""
        for category, products in self.product_catalog.items():
            for product_name in products:
                if product_name in text:
                    return ClassificationResult(
                        category=category,
                        confidence=1.0,
                        matched_keywords=[product_name],
                        method='catalog'
                    )
        return None
    
    def _classify_by_keywords(self, text: str) -> Optional[ClassificationResult]:
        """Классификация по ключевым словам с весами"""
        category_scores = defaultdict(float)
        category_matches = defaultdict(list)
        
        for category, weights in self.keyword_weights.items():
            for keyword, weight in weights.items():
                # Используем границы слов для точного поиска
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text):
                    category_scores[category] += weight
                    category_matches[category].append(keyword)
        
        if not category_scores:
            return None
        
        # Выбираем категорию с наибольшим score
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        best_score = category_scores[best_category]
        
        # Нормализуем score в уверенность
        max_possible_score = sum(sorted([
            weight for weight in self.keyword_weights[best_category].values()
        ], reverse=True)[:3])
        
        confidence = min(best_score / max_possible_score, 1.0)
        
        return ClassificationResult(
            category=best_category,
            confidence=confidence,
            matched_keywords=category_matches[best_category],
            method='keywords'
        )
    
    def _classify_by_partial_match(self, text: str) -> Optional[ClassificationResult]:
        """Классификация по частичным совпадениям (запасной метод)"""
        words = text.split()
        
        for category, weights in self.keyword_weights.items():
            matched_keywords = []
            for keyword in weights.keys():
                if len(keyword) > 3 and any(keyword in word for word in words):
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                confidence = self._calculate_confidence(matched_keywords, category)
                if confidence >= self.confidence_threshold:
                    return ClassificationResult(
                        category=category,
                        confidence=confidence * 0.7,  # Понижаем уверенность для частичных совпадений
                        matched_keywords=matched_keywords,
                        method='partial'
                    )
        
        return None
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Основной метод классификации текста
        
        Args:
            text: Текст для классификации
            
        Returns:
            ClassificationResult: Результат классификации
        """
        start_time = time.time()
        
        # Предварительная обработка текста
        processed_text = self._preprocess_text(text)
        
        # Обновляем статистику
        self.stats['total_classifications'] += 1
        
        # Пытаемся классифицировать разными методами по порядку приоритета
        result = self._classify_by_catalog(processed_text)
        if result:
            self.stats['catalog_matches'] += 1
            return self._finalize_result(result, start_time)
        
        result = self._classify_by_keywords(processed_text)
        if result and result.confidence >= self.confidence_threshold:
            self.stats['keyword_matches'] += 1
            return self._finalize_result(result, start_time)
        
        result = self._classify_by_partial_match(processed_text)
        if result:
            self.stats['keyword_matches'] += 1
            return self._finalize_result(result, start_time)
        
        # Если ничего не найдено
        self.stats['fallback_classifications'] += 1
        fallback_result = ClassificationResult(
            category='Другое',
            confidence=0.0,
            matched_keywords=[],
            method='fallback'
        )
        
        return self._finalize_result(fallback_result, start_time)
    
    def _finalize_result(self, result: ClassificationResult, start_time: float) -> ClassificationResult:
        """Завершает обработку результата и обновляет статистику"""
        processing_time = time.time() - start_time
        
        # Обновляем среднее время обработки
        total_time = self.stats['avg_processing_time'] * (self.stats['total_classifications'] - 1)
        self.stats['avg_processing_time'] = (total_time + processing_time) / self.stats['total_classifications']
        
        # Логируем результат если уверенность низкая
        if result.confidence < 0.5 and result.category != 'Другое':
            logger.info(f"Низкая уверенность классификации: {result.category} ({result.confidence:.2f})")
        
        return result
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику классификации"""
        return self.stats.copy()
    
    def add_custom_category(self, category: str, keywords: List[str], weights: Optional[Dict[str, float]] = None):
        """Добавляет пользовательскую категорию для классификации"""
        if weights is None:
            weights = {keyword: 1.0 for keyword in keywords}
        
        self.keyword_weights[category] = weights
        self._get_word_forms.cache_clear()  # Очищаем кэш
    
    def save_classification_rules(self, filepath: str):
        """Сохраняет правила классификации в файл"""
        rules = {
            'keyword_weights': self.keyword_weights,
            'product_catalog': self.product_catalog
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
    
    def load_classification_rules(self, filepath: str):
        """Загружает правила классификации из файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            
            self.keyword_weights = rules.get('keyword_weights', self.keyword_weights)
            self.product_catalog = rules.get('product_catalog', self.product_catalog)
            self._get_word_forms.cache_clear()
            
        except Exception as e:
            logger.error(f"Ошибка загрузки правил классификации: {e}")


# Пример использования
def main():
    # Создаем классификатор
    classifier = AdvancedProductClassifier()
    
    # Тестовые примеры
    test_texts = [
        "Бронежилет защитный новый с керамическими плитами",
        "Здравствуйте, хотел бы заказать брюки тактические Ещё одни... По старым меркам. Но с небольшими уточнениями.Такое возможно.",
        "Бронь с баллистикой и капами без плит, подсумок Абдоминальный.",
        "Сувенирный набор с символикой",
        "Название: Противоосколочные жилеты для служебных собак, помогающих российским штурмовикам на передовой, создали в Донецке на предприятии 3Д Техно. Первые два жилета сделали индивидуально для пса по кличке Бандит, но разработчик уверен, что их будут изготавливать серийно."
    ]
    
    print("Тестирование классификатора:")
    print("=" * 50)
    
    for text in test_texts:
        result = classifier.classify(text)
        print(f"Текст: {text}")
        print(f"Категория: {result.category}")
        print(f"Уверенность: {result.confidence:.2f}")
        print(f"Метод: {result.method}")
        print(f"Найденные ключевые слова: {', '.join(result.matched_keywords)}")
        print("-" * 50)
    
    # Выводим статистику
    stats = classifier.get_statistics()
    print(f"\nСтатистика классификации:")
    print(f"Всего классификаций: {stats['total_classifications']}")
    print(f"Совпадений с каталогом: {stats['catalog_matches']}")
    print(f"Совпадений по ключевым словам: {stats['keyword_matches']}")
    print(f"Резервных классификаций: {stats['fallback_classifications']}")
    print(f"Среднее время обработки: {stats['avg_processing_time']:.4f} сек")


if __name__ == "__main__":
    main()
