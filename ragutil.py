import os
import glob
import pickle
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging

# Дополнительные импорты
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader,
    UnstructuredMarkdownLoader, UnstructuredPowerPointLoader
)
import faiss
from rank_bm25 import BM25Okapi
import numpy as np

# Импорты для re-ranker
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    from torch.nn import Softmax
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("Transformers не установлен. Re-ranker будет отключен.")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])

class ReRanker:
    """Класс для повторного ранжирования результатов поиска"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = "cuda"):
        if not RERANKER_AVAILABLE:
            self.model = None
            self.tokenizer = None
            self.device = device
            logger.warning("Re-ranker отключен (transformers не установлен)")
            return
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to("cuda:3")
            self.model.eval()
            self.device = device
            self.softmax = Softmax(dim=1)
            logger.info(f"Re-ranker инициализирован с моделью {model_name} на {device}")
        except Exception as e:
            logger.error(f"Ошибка инициализации re-ranker: {str(e)}")
            self.model = None
            self.tokenizer = None

    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Tuple[Document, float]]:
        """Повторное ранжирование документов относительно запроса"""
        if not self.model or not documents:
            return [(doc, 0.0) for doc in documents][:top_k]
        
        try:
            # Подготавливаем пары запрос-документ
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Токенизация
            features = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            ).to("cuda:3") # self.device
            
            # Предсказание
            with torch.no_grad():
                scores = self.model(**features).logits
            # Применяем softmax для получения вероятностей
            # Вместо softmax + извлечения
            relevance_scores = scores.squeeze(-1).cpu().numpy()
            
            # Сортируем документы по убыванию релевантности
            scored_docs = list(zip(documents, relevance_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Re-ranker обработал {len(documents)} документов")
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Ошибка в re-ranker: {str(e)}")
            # Возвращаем оригинальные документы с нулевыми скорами
            return [(doc, 0.0) for doc in documents][:top_k]

    def is_available(self) -> bool:
        """Проверяет, доступен ли re-ranker"""
        return self.model is not None


class RAGAnalyzer:
    def __init__(self, 
                 model_name: str = "intfloat/multilingual-e5-large",
                 reranker_model: str = "bge-reranker-v2-m3",  # Многоязычная модель
                 device: str = "cuda:3"):
        
        # ИСПРАВЛЕНИЕ: Убираем show_progress_bar из encode_kwargs
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
                # Убрано: "show_progress_bar": True - вызывает конфликт
            }
        )
        
        # Инициализация re-ranker
        self.reranker = ReRanker(model_name=reranker_model, device=device)
        
        # Улучшенный text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        self.vector_store: Optional[FAISS] = None
        self.indexed_files: Dict[str, Dict] = {}
        self.bm25_index = None
        self.bm25_documents = []
        self.file_hashes: Dict[str, str] = {}
        
        logger.info(f"Инициализирован RAGAnalyzer с моделью {model_name} и re-ranker {reranker_model}")

    def _get_file_hash(self, file_path: str) -> str:
        """Вычисляет хэш файла для отслеживания изменений"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Ошибка вычисления хэша для {file_path}: {str(e)}")
            return ""

    def _should_reindex(self, file_path: str) -> bool:
        """Проверяет, нужно ли переиндексировать файл"""
        if file_path not in self.file_hashes:
            return True
        
        current_hash = self._get_file_hash(file_path)
        if not current_hash:
            return True
            
        return current_hash != self.file_hashes[file_path]

    def _build_bm25_index(self, documents: List[Document]):
        """Строит BM25 индекс для гибридного поиска"""
        try:
            if not documents:
                logger.warning("Нет документов для построения BM25 индекса")
                return
                
            texts = [doc.page_content for doc in documents]
            tokenized_texts = [text.lower().split() for text in texts]
            self.bm25_index = BM25Okapi(tokenized_texts)
            self.bm25_documents = documents
            logger.info(f"Построен BM25 индекс для {len(documents)} документов")
        except Exception as e:
            logger.warning(f"Не удалось построить BM25 индекс: {e}")
            self.bm25_index = None
            self.bm25_documents = []

    def _rebuild_bm25_from_vector_store(self):
        """Перестраивает BM25 индекс из векторного хранилища"""
        if not self.vector_store:
            self.bm25_index = None
            self.bm25_documents = []
            return
            
        try:
            all_documents = []
            for doc_id, doc in self.vector_store.docstore._dict.items():
                if isinstance(doc, Document):
                    all_documents.append(doc)
            
            self._build_bm25_index(all_documents)
        except Exception as e:
            logger.error(f"Ошибка перестроения BM25 индекса: {str(e)}")
            self.bm25_index = None
            self.bm25_documents = []

    def load_documents(self, file_path: str) -> List[Document]:
        """Загрузка документов с поддержкой разных форматов"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.ppt': UnstructuredPowerPointLoader,
        }
        
        loader_class = loaders.get(file_ext, TextLoader)
        
        try:
            loader = loader_class(file_path)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata.update({
                    'source_file': os.path.basename(file_path),
                    'file_path': file_path,
                    'file_type': file_ext,
                    'load_time': datetime.now().isoformat()
                })
            
            logger.info(f"Загружен {file_path}: {len(documents)} страниц")
            return documents
            
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {str(e)}")
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                for doc in documents:
                    doc.metadata.update({
                        'source_file': os.path.basename(file_path),
                        'file_path': file_path,
                        'file_type': file_ext,
                        'load_time': datetime.now().isoformat()
                    })
                return documents
            except Exception as fallback_error:
                logger.error(f"Не удалось загрузить файл {file_path}: {fallback_error}")
                return []

    def build_index(self, directory_path: str, exclude_dirs: list = None, 
                   exclude_files: list = None, use_hybrid: bool = True) -> Dict[str, Any]:
        """Построение индекса с поддержкой гибридного поиска"""
        exclude_dirs = exclude_dirs or []
        exclude_files = exclude_files or []
        
        supported_extensions = ['*.py', '*.txt', '*.md', '*.pdf', '*.docx', '*.pptx', '*.ppt']
        all_files = []
        
        for ext in supported_extensions:
            all_files.extend(glob.glob(os.path.join(directory_path, '**', ext), recursive=True))

        if not all_files:
            return {"success": False, "message": "Файлы не найдены в указанной директории!"}

        documents = []
        new_indexed_files = {}
        processed_count = 0
        
        for file_path in all_files:
            abs_path = os.path.abspath(file_path)
            filename = os.path.basename(file_path)
            
            if any(ex_dir in abs_path for ex_dir in exclude_dirs):
                continue
                
            if filename in exclude_files:
                continue
                
            if abs_path in self.indexed_files and not self._should_reindex(abs_path):
                logger.info(f"Файл не изменился, пропускаем: {filename}")
                continue

            try:
                file_docs = self.load_documents(abs_path)
                if not file_docs:
                    continue
                    
                split_docs = self.text_splitter.split_documents(file_docs)
                documents.extend(split_docs)
                
                new_indexed_files[abs_path] = {
                    'file_name': filename,
                    'chunks_count': len(split_docs),
                    'last_modified': datetime.now().isoformat(),
                    'file_size': os.path.getsize(abs_path)
                }
                
                self.file_hashes[abs_path] = self._get_file_hash(abs_path)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Ошибка при обработке {file_path}: {str(e)}")
                continue

        if not documents:
            return {"success": False, "message": "Нет документов для индексации!"}
            
        try:
            # ИСПРАВЛЕНИЕ: Добавляем прогресс-бар вручную
            logger.info("Начинаем построение векторного индекса...")
            new_vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            logger.info("Векторный индекс построен успешно")
            
            if self.vector_store:
                self.vector_store.merge_from(new_vector_store)
                self.indexed_files.update(new_indexed_files)
            else:
                self.vector_store = new_vector_store
                self.indexed_files = new_indexed_files
            
            if use_hybrid:
                self._rebuild_bm25_from_vector_store()
            
            total_chunks = self.vector_store.index.ntotal
            
            logger.info(f"Индекс успешно построен. Файлов: {len(self.indexed_files)}, Чанков: {total_chunks}")
            
            return {
                "success": True,
                "message": f"Индекс обновлён! Обработано файлов: {processed_count}, Файлов в индексе: {len(self.indexed_files)}, Фрагментов: {total_chunks}",
                "files_count": len(self.indexed_files),
                "chunks_count": total_chunks,
                "processed_files": processed_count,
                "reranker_available": self.reranker.is_available()
            }
            
        except Exception as e:
            logger.error(f"Ошибка построения индекса: {str(e)}")
            return {"success": False, "message": f"Ошибка построения индекса: {str(e)}"}

    def clear_database(self) -> Dict[str, Any]:
        """
        Полная очистка всей базы данных
        Удаляет все индексы, кэши и метаданные
        """
        try:
            # Очищаем векторное хранилище
            self.vector_store = None
            
            # Очищаем файловые индексы
            self.indexed_files.clear()
            self.file_hashes.clear()
            
            # Очищаем BM25
            self.bm25_index = None
            self.bm25_documents.clear()
            
            # Принудительный сбор мусора для освобождения памяти
            import gc
            gc.collect()
            
            logger.info("База данных полностью очищена")
            return {
                "success": True, 
                "message": "База данных полностью очищена. Все индексы и кэши удалены."
            }
            
        except Exception as e:
            logger.error(f"Ошибка при очистке базы данных: {str(e)}")
            return {"success": False, "message": f"Ошибка при очистке базы данных: {str(e)}"}

    def delete_index_files(self, storage_path: str = "./vector_store") -> Dict[str, Any]:
        """
        Удаляет физические файлы индекса с диска
        """
        try:
            if os.path.exists(storage_path):
                import shutil
                shutil.rmtree(storage_path)
                logger.info(f"Файлы индекса удалены из: {storage_path}")
                return {
                    "success": True,
                    "message": f"Файлы индекса удалены из: {storage_path}"
                }
            else:
                return {
                    "success": False,
                    "message": f"Директория {storage_path} не существует"
                }
        except Exception as e:
            logger.error(f"Ошибка удаления файлов индекса: {str(e)}")
            return {"success": False, "message": f"Ошибка удаления файлов индекса: {str(e)}"}

    def get_database_size(self) -> Dict[str, Any]:
        """
        Возвращает информацию о размере базы данных
        """
        try:
            total_size = 0
            file_count = 0
            
            for file_path, metadata in self.indexed_files.items():
                if os.path.exists(file_path):
                    total_size += metadata.get('file_size', 0)
                    file_count += 1
            
            # Размер в памяти (приблизительно)
            memory_size = 0
            if self.vector_store:
                # Приблизительный расчет размера векторов
                memory_size = self.vector_store.index.ntotal * self.vector_store.index.d * 4  # 4 байта на float32
            
            return {
                "success": True,
                "file_count": file_count,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "memory_size_mb": round(memory_size / (1024 * 1024), 2),
                "chunks_count": self.vector_store.index.ntotal if self.vector_store else 0,
                "indexed_files": len(self.indexed_files)
            }
        except Exception as e:
            return {"success": False, "message": f"Ошибка расчета размера БД: {str(e)}"}

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7, 
                     use_reranker: bool = True) -> List[Document]:
        """Гибридный поиск с опциональным re-ранкингом"""
        if not self.vector_store:
            return []
        
        initial_results = []
        
        try:
            # Семантический поиск
            semantic_results = self.vector_store.similarity_search(query, k=k*3)
            semantic_scores = {}
            
            for i, doc in enumerate(semantic_results):
                normalized_score = 1.0 - (i / (len(semantic_results) * 2))
                semantic_scores[doc.page_content] = (1 - alpha) * normalized_score
            
            # Поиск по ключевым словам (BM25)
            if self.bm25_index and self.bm25_documents:
                tokenized_query = query.lower().split()
                if tokenized_query:
                    bm25_scores = self.bm25_index.get_scores(tokenized_query)
                    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1
                    
                    normalized_bm25_scores = []
                    for score in bm25_scores:
                        if max_bm25 > 0:
                            normalized_bm25_scores.append(score / max_bm25)
                        else:
                            normalized_bm25_scores.append(0.0)
                    
                    for i, score in enumerate(normalized_bm25_scores):
                        if i < len(self.bm25_documents) and score > 0:
                            doc = self.bm25_documents[i]
                            content = doc.page_content
                            if content in semantic_scores:
                                semantic_scores[content] += alpha * score
                            else:
                                semantic_scores[content] = alpha * score
            
            # Комбинируем результаты
            scored_docs = []
            all_docs = semantic_results + (self.bm25_documents if self.bm25_documents else [])
            
            for doc in all_docs:
                score = semantic_scores.get(doc.page_content, 0)
                if score > 0:
                    scored_docs.append((doc, score))
            
            # Убираем дубликаты
            unique_docs = {}
            for doc, score in scored_docs:
                content = doc.page_content
                if content not in unique_docs or score > unique_docs[content][1]:
                    unique_docs[content] = (doc, score)
            
            sorted_results = sorted(unique_docs.values(), key=lambda x: x[1], reverse=True)
            initial_results = [doc for doc, score in sorted_results[:k*2]]
            
            # Применяем re-ranker если доступен и запрошен
            if use_reranker and self.reranker.is_available() and initial_results:
                reranked_results = self.reranker.rerank(query, initial_results, top_k=k)
                final_results = [doc for doc, score in reranked_results]
                logger.info(f"Применен re-ranker. Обработано {len(initial_results)} документов")
            else:
                final_results = initial_results[:k]
                if use_reranker and not self.reranker.is_available():
                    logger.info("Re-ranker недоступен, используется стандартное ранжирование")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Ошибка гибридного поиска: {str(e)}")
            if self.vector_store:
                return self.vector_store.similarity_search(query, k=k)
            return []

    # Остальные методы остаются без изменений...
    def search_documents(self, query: str, k: int = 5, search_type: str = "hybrid", 
                        use_reranker: bool = True) -> str:
        """Улучшенный поиск с поддержкой re-ранкинга"""
        if not self.vector_store:
            return ""

        try:
            if search_type == "hybrid":
                results = self.hybrid_search(query, k=k, use_reranker=use_reranker)
            elif search_type == "keyword" and self.bm25_index:
                results = self.keyword_search(query, k=k)
            elif search_type == "semantic":
                results = self.vector_store.similarity_search(query, k=k)
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            if not results:
                return ""

            context_parts = []
            for i, doc in enumerate(results):
                source = doc.metadata.get('source_file', 'Unknown')
                context_parts.append(f"--- Документ {i+1} ({source}) ---\n{doc.page_content}")

            context = "\n\n".join(context_parts)
            
            search_method = "гибридный поиск" if search_type == "hybrid" else search_type
            reranker_info = " + re-ranker" if use_reranker and self.reranker.is_available() else ""
            
            return f"### Найденные релевантные документы ({len(results)}) - {search_method}{reranker_info}:\n{context}"

        except Exception as e:
            logger.error(f"Ошибка поиска: {str(e)}")
            return f"Ошибка при поиске: {str(e)}"

    def get_index_stats(self) -> Dict[str, Any]:
        """Статистика индекса"""
        if not self.vector_store:
            return {"status": "Индекс не инициализирован"}
        
        total_size = sum(info.get('file_size', 0) for info in self.indexed_files.values())
        total_chunks = self.vector_store.index.ntotal
        
        return {
            "status": "Активен",
            "indexed_files": len(self.indexed_files),
            "total_chunks": total_chunks,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "embedding_dim": self.vector_store.index.d,
            "bm25_index": "Активен" if self.bm25_index else "Неактивен",
            "bm25_documents": len(self.bm25_documents) if self.bm25_documents else 0,
            "file_hashes_count": len(self.file_hashes),
            "reranker_available": self.reranker.is_available(),
            "vector_store_type": "FAISS"
        }

    # Сохранение и загрузка индекса
    def save_index(self, save_path: str) -> Dict[str, Any]:
        """Сохраняет индекс с метаданными"""
        if not self.vector_store:
            return {"success": False, "message": "Индекс не инициализирован!"}
            
        try:
            os.makedirs(save_path, exist_ok=True)
            self.vector_store.save_local(os.path.join(save_path, "faiss_index"))
            
            metadata = {
                "indexed_files": self.indexed_files,
                "file_hashes": self.file_hashes,
                "timestamp": datetime.now().isoformat(),
                "model_name": self.embeddings.model_name,
                "vector_store_class": "FAISS"
            }
            
            with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
                
            if self.bm25_index and self.bm25_documents:
                bm25_data = {
                    'documents': self.bm25_documents,
                }
                with open(os.path.join(save_path, "bm25_index.pkl"), "wb") as f:
                    pickle.dump(bm25_data, f)
            
            logger.info(f"Индекс сохранён в: {save_path}")
            return {
                "success": True, 
                "message": f"Индекс сохранён в: {save_path}",
                "files_count": len(self.indexed_files),
                "chunks_count": self.vector_store.index.ntotal,
                "bm25_saved": self.bm25_index is not None
            }
            
        except Exception as e:
            logger.error(f"Ошибка сохранения индекса: {str(e)}")
            return {"success": False, "message": f"Ошибка сохранения индекса: {str(e)}"}

    def load_index(self, load_path: str) -> Dict[str, Any]:
        """Загружает индекс с метаданными"""
        try:
            self.vector_store = FAISS.load_local(
                os.path.join(load_path, "faiss_index"),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            with open(os.path.join(load_path, "metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)
                
            self.indexed_files = metadata.get("indexed_files", {})
            self.file_hashes = metadata.get("file_hashes", {})
            
            bm25_path = os.path.join(load_path, "bm25_index.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    bm25_data = pickle.load(f)
                self.bm25_documents = bm25_data.get('documents', [])
                if self.bm25_documents:
                    self._build_bm25_index(self.bm25_documents)
            
            logger.info(f"Индекс загружен: {len(self.indexed_files)} файлов, {self.vector_store.index.ntotal} чанков")
            
            return {
                "success": True,
                "message": f"Индекс загружен! Файлов: {len(self.indexed_files)}, Фрагментов: {self.vector_store.index.ntotal}",
                "files_count": len(self.indexed_files),
                "chunks_count": self.vector_store.index.ntotal
            }
            
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса: {str(e)}")
            return {"success": False, "message": f"Ошибка загрузки индекса: {str(e)}"}

# Пример использования
def main():
    # Инициализация с re-ranker
    rag = RAGAnalyzer(
        model_name="intfloat/multilingual-e5-large",
        reranker_model="bge-reranker-v2-m3",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    rag.load_index("savedata_rag")
    print("✅ RAG инициализирован")
    
    
#    # Построение индекса
#    result = rag.build_index("data_rag")
#    print(f"✅ Построение индекса: {result['success']}")
#    
#    # Поиск
#    search_result = rag.search_documents("какая статья за мошенничество?")
#    print(f"✅ Поиск работает: {'документы найдены' if search_result else 'нет результатов'}")
#    # Статистика
#    stats = rag.get_index_stats()
#    print(f"✅ Статистика: {stats}")
#    result = rag.save_index("savedata_rag")
#    if result["success"]:
#        print("✅ Индекс успешно сохранен")
#        print(f"Результат сохранения: {result}")
#    
#    # Очистка
#    rag.clear_database()
#    print("✅ Очистка выполнена")
    
#    print("🎉 БЫСТРЫЙ ТЕСТ ПРОЙДЕН!")    

#    # Доступные методы поиска
#    methods = rag.get_search_methods()
#    print("Доступные методы:", methods)

if __name__ == "__main__":
    main()
