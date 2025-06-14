import re
import unicodedata
import json
import asyncio
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress
import logging

# Jupyter/Colab uyumluluğu için
try:
    import nest_asyncio
    NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    NEST_ASYNCIO_AVAILABLE = False

# Yapılandırma
console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Gelişmiş belge yapısı"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    chunk_id: Optional[str] = None
    source_line: int = 0

@dataclass
class SearchResult:
    """Arama sonucu yapısı"""
    document: Document
    score: float
    relevance_explanation: str
    matched_terms: List[str]
    retrieval_method: str

@dataclass
class QueryContext:
    """Sorgu bağlamı ve analizi"""
    original_query: str
    processed_query: str
    intent: str
    query_type: str
    complexity_score: float
    entities: List[str]

class QueryAnalyzer:
    """Sorgu analizi ve sınıflandırması için ajan"""

    def __init__(self):
        self.intents = {
            'factual': ['ne', 'nedir', 'kim', 'nerede', 'nasıl', 'hangi'],
            'comparison': ['fark', 'karşılaştır', 'benzer', 'farklı'],
            'temporal': ['zaman', 'tarih', 'önce', 'sonra', 'ne zaman'],
            'causal': ['neden', 'sebep', 'nedeni', 'sonucu']
        }

    def analyze_query(self, query: str) -> QueryContext:
        """Sorguyu analiz eder ve bağlam oluşturur"""
        processed = self._preprocess_query(query)
        intent = self._detect_intent(processed)
        query_type = self._classify_query_type(processed)
        complexity = self._calculate_complexity(processed)
        entities = self._extract_entities(processed)

        return QueryContext(
            original_query=query,
            processed_query=processed,
            intent=intent,
            query_type=query_type,
            complexity_score=complexity,
            entities=entities
        )

    def _preprocess_query(self, query: str) -> str:
        """Sorguyu ön işleme"""
        normalized = unicodedata.normalize('NFKC', query.lower())
        normalized = normalized.replace("İ", "i").replace("I", "ı")
        return re.sub(r'[^\w\s]', '', normalized)

    def _detect_intent(self, query: str) -> str:
        """Sorgu amacını tespit eder"""
        words = query.split()
        for intent, keywords in self.intents.items():
            if any(keyword in query for keyword in keywords):
                return intent
        return 'general'

    def _classify_query_type(self, query: str) -> str:
        """Sorgu türünü sınıflandırır"""
        if len(query.split()) <= 3:
            return 'simple'
        elif any(word in query for word in ['ve', 'veya', 'ancak']):
            return 'complex'
        else:
            return 'medium'

    def _calculate_complexity(self, query: str) -> float:
        """Sorgu karmaşıklığını hesaplar"""
        words = query.split()
        return min(len(words) / 10.0, 1.0)

    def _extract_entities(self, query: str) -> List[str]:
        """Basit varlık çıkarma"""
        # Büyük harfle başlayan kelimeler (basit NER)
        entities = re.findall(r'\b[A-ZÜĞŞIÖÇ][a-züğşıöç]+\b', query)
        return list(set(entities))

class RetrievalStrategy:
    """Farklı retrieval stratejileri için base class"""

    def __init__(self, name: str):
        self.name = name

    def retrieve(self, query_context: QueryContext, documents: List[Document], top_k: int = 3) -> List[SearchResult]:
        raise NotImplementedError

class SemanticRetrievalStrategy(RetrievalStrategy):
    """Semantic/Vector-based retrieval"""

    def __init__(self):
        super().__init__("semantic")

    def retrieve(self, query_context: QueryContext, documents: List[Document], top_k: int = 3) -> List[SearchResult]:
        results = []
        query_words = query_context.processed_query.split()

        for doc in documents:
            score = self._calculate_semantic_similarity(query_words, doc.content)
            if score > 0:
                matched_terms = self._find_matched_terms(query_words, doc.content)
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    relevance_explanation=f"Semantic similarity: {score:.3f}",
                    matched_terms=matched_terms,
                    retrieval_method=self.name
                ))

        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

    def _calculate_semantic_similarity(self, query_words: List[str], content: str) -> float:
        content_words = content.lower().split()
        query_counter = Counter(query_words)
        content_counter = Counter(content_words)

        common_words = set(query_counter.keys()) & set(content_counter.keys())
        if not common_words:
            return 0

        return sum(query_counter[word] * content_counter[word] for word in common_words) / len(query_words)

    def _find_matched_terms(self, query_words: List[str], content: str) -> List[str]:
        content_words = set(content.lower().split())
        return [word for word in query_words if word in content_words]

class HybridRetrievalStrategy(RetrievalStrategy):
    """Hibrit retrieval stratejisi (BM25 + Semantic)"""

    def __init__(self):
        super().__init__("hybrid")
        self.bm25_weight = 0.6
        self.semantic_weight = 0.4

    def retrieve(self, query_context: QueryContext, documents: List[Document], top_k: int = 3) -> List[SearchResult]:
        # BM25 ve semantic skorları hesapla
        bm25_results = self._calculate_bm25_scores(query_context, documents)
        semantic_results = self._calculate_semantic_scores(query_context, documents)

        # Skorları birleştir
        combined_results = []
        for doc in documents:
            bm25_score = next((r['score'] for r in bm25_results if r['doc_id'] == doc.id), 0)
            semantic_score = next((r['score'] for r in semantic_results if r['doc_id'] == doc.id), 0)

            combined_score = (self.bm25_weight * bm25_score) + (self.semantic_weight * semantic_score)

            if combined_score > 0:
                matched_terms = self._find_matched_terms(query_context.processed_query.split(), doc.content)
                combined_results.append(SearchResult(
                    document=doc,
                    score=combined_score,
                    relevance_explanation=f"Hybrid (BM25: {bm25_score:.3f}, Semantic: {semantic_score:.3f})",
                    matched_terms=matched_terms,
                    retrieval_method=self.name
                ))

        return sorted(combined_results, key=lambda x: x.score, reverse=True)[:top_k]

    def _calculate_bm25_scores(self, query_context: QueryContext, documents: List[Document]) -> List[Dict]:
        # BM25 implementasyonu (basitleştirilmiş)
        results = []
        query_terms = query_context.processed_query.split()

        for doc in documents:
            score = self._bm25_score(query_terms, doc.content.lower().split())
            results.append({'doc_id': doc.id, 'score': score})

        return results

    def _calculate_semantic_scores(self, query_context: QueryContext, documents: List[Document]) -> List[Dict]:
        # Semantic similarity hesaplama
        results = []
        query_words = query_context.processed_query.split()

        for doc in documents:
            score = self._semantic_similarity(query_words, doc.content.lower().split())
            results.append({'doc_id': doc.id, 'score': score})

        return results

    def _bm25_score(self, query_terms: List[str], doc_terms: List[str], k1: float = 1.5, b: float = 0.75) -> float:
        doc_len = len(doc_terms)
        avg_doc_len = doc_len  # Basitleştirme
        doc_freq = Counter(doc_terms)

        score = 0
        for term in query_terms:
            tf = doc_freq.get(term, 0)
            if tf > 0:
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                score += numerator / denominator

        return score

    def _semantic_similarity(self, query_words: List[str], doc_words: List[str]) -> float:
        query_counter = Counter(query_words)
        doc_counter = Counter(doc_words)

        common_words = set(query_counter.keys()) & set(doc_counter.keys())
        if not common_words:
            return 0

        return sum(query_counter[word] * doc_counter[word] for word in common_words) / len(query_words)

    def _find_matched_terms(self, query_words: List[str], content: str) -> List[str]:
        content_words = set(content.lower().split())
        return [word for word in query_words if word in content_words]

class RetrievalRouter:
    """Retrieval stratejilerini yönlendiren ajan"""

    def __init__(self):
        self.strategies = {
            'semantic': SemanticRetrievalStrategy(),
            'hybrid': HybridRetrievalStrategy()
        }

    def route_query(self, query_context: QueryContext) -> str:
        """Sorgu bağlamına göre en uygun stratejiyi seçer"""
        if query_context.complexity_score > 0.7:
            return 'hybrid'
        elif query_context.intent in ['comparison', 'causal']:
            return 'hybrid'
        else:
            return 'semantic'

    def retrieve(self, query_context: QueryContext, documents: List[Document], top_k: int = 3) -> List[SearchResult]:
        strategy_name = self.route_query(query_context)
        strategy = self.strategies[strategy_name]

        logger.info(f"Query routed to {strategy_name} strategy")
        return strategy.retrieve(query_context, documents, top_k)

class DocumentProcessor:
    """Gelişmiş belge işleme"""

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """Birden fazla dosyadan (txt, pdf, görsel) belgeleri yükler ve işler (görsellerde sadece dosya adı ve yolunda arama yapılır)"""
        documents = []
        import os
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            console.print("[yellow]PyPDF2 yüklü değil, PDF dosyaları okunamayacak.[/yellow]")
            PdfReader = None
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".txt":
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                    for i, line in enumerate(lines):
                        doc = Document(
                            id=f"{Path(file_path).stem}_doc_{i}",
                            content=line,
                            metadata={
                                'source': file_path,
                                'created_at': datetime.now().isoformat(),
                                'length': len(line)
                            },
                            source_line=i + 1
                        )
                        documents.append(doc)
                except FileNotFoundError:
                    console.print(f"[red]Dosya bulunamadı: {file_path}[/red]")
            elif ext == ".pdf" and PdfReader is not None:
                try:
                    reader = PdfReader(file_path)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            doc = Document(
                                id=f"{Path(file_path).stem}_pdf_{page_num}",
                                content=text.strip(),
                                metadata={
                                    'source': file_path,
                                    'created_at': datetime.now().isoformat(),
                                    'page': page_num
                                },
                                source_line=page_num + 1
                            )
                            documents.append(doc)
                except Exception as e:
                    console.print(f"[red]PDF okunamadı: {file_path} ({e})[/red]")
            elif ext in image_exts:
                # Görsel dosyası: sadece dosya adı ve yolunda arama yapılır
                doc = Document(
                    id=f"{Path(file_path).stem}_img",
                    content=os.path.basename(file_path),
                    metadata={
                        'source': file_path,
                        'created_at': datetime.now().isoformat(),
                        'type': 'image'
                    },
                    source_line=1
                )
                documents.append(doc)
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Belgeleri chunk'lara böler"""
        chunked_docs = []

        for doc in documents:
            if len(doc.content.split()) <= self.chunk_size:
                chunked_docs.append(doc)
            else:
                chunks = self._create_chunks(doc.content)
                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        id=f"{doc.id}_chunk_{i}",
                        content=chunk,
                        metadata={**doc.metadata, 'is_chunk': True, 'chunk_index': i},
                        chunk_id=f"{doc.id}_chunk_{i}",
                        source_line=doc.source_line
                    )
                    chunked_docs.append(chunk_doc)

        return chunked_docs

    def _create_chunks(self, text: str) -> List[str]:
        """Metni chunk'lara böler"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(' '.join(chunk_words))

            if i + self.chunk_size >= len(words):
                break

        return chunks

class AgenticRAGSystem:
    """Ana agentic RAG sistemi"""

    def __init__(self, file_paths: Optional[List[str]] = None):
        if file_paths is None:
            file_paths = ["belgeler.txt"]
        self.file_paths = file_paths
        self.query_analyzer = QueryAnalyzer()
        self.retrieval_router = RetrievalRouter()
        self.document_processor = DocumentProcessor()
        self.documents = []
        self._load_and_process_documents()

    def _load_and_process_documents(self):
        """Birden fazla dosyadan belgeleri yükler ve işler"""
        raw_documents = self.document_processor.load_documents(self.file_paths)
        self.documents = self.document_processor.chunk_documents(raw_documents)
        logger.info(f"Loaded {len(self.documents)} document chunks from {len(self.file_paths)} files")

    async def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Ana sorgu işleme methodu"""
        # Sorgu analizi
        query_context = self.query_analyzer.analyze_query(question)

        # Retrieval
        results = self.retrieval_router.retrieve(query_context, self.documents, top_k)

        # Cevap üretimi
        answer = self._generate_answer(query_context, results)

        return {
            'query_context': asdict(query_context),
            'results': [asdict(result) for result in results],
            'answer': answer,
            'metadata': {
                'total_documents': len(self.documents),
                'retrieval_strategy': results[0].retrieval_method if results else 'none',
                'processing_time': datetime.now().isoformat()
            }
        }

    def _generate_answer(self, query_context: QueryContext, results: List[SearchResult]) -> str:
        """Basit cevap üretimi"""
        if not results:
            return "İlgili bilgi bulunamadı."

        # En yüksek skorlu sonuçları birleştir
        top_contents = [result.document.content for result in results[:2]]
        combined_content = " ".join(top_contents)

        return f"Sorunuzla ilgili bulduğum bilgiler: {combined_content[:500]}..."

    def get_statistics(self) -> Dict[str, Any]:
        """Sistem istatistikleri"""
        total_words = sum(len(doc.content.split()) for doc in self.documents)
        avg_doc_length = total_words / len(self.documents) if self.documents else 0

        return {
            'total_documents': len(self.documents),
            'total_words': total_words,
            'average_document_length': avg_doc_length,
            'available_strategies': list(self.retrieval_router.strategies.keys())
        }

    def evaluate(self, queries: list[dict], top_k: int = 3) -> dict:
        """Evaluate retrieval with recall and MRR for a list of queries.
        Each query dict should have 'question' and 'ground_truth' (list of relevant doc ids).
        """
        from . import calculate_recall, calculate_mrr  # Ensure local import if needed
        recalls = []
        mrrs = []
        for q in queries:
            question = q['question']
            ground_truth = q['ground_truth']
            # Run retrieval
            query_context = self.query_analyzer.analyze_query(question)
            results = self.retrieval_router.retrieve(query_context, self.documents, top_k)
            predictions = [result.document.id for result in results]
            recalls.append(calculate_recall(predictions, ground_truth))
            mrrs.append(calculate_mrr(predictions, ground_truth))
        return {
            'recall': sum(recalls) / len(recalls) if recalls else 0.0,
            'mrr': sum(mrrs) / len(mrrs) if mrrs else 0.0,
            'details': list(zip(recalls, mrrs))
        }

class EnhancedCLI:
    """Gelişmiş komut satırı arayüzü"""

    def __init__(self):
        # Çoklu dosya desteği: tüm txt, pdf ve görsel dosyalarını yükle
        import glob
        file_paths = glob.glob("*.txt") + glob.glob("*.pdf") + glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png") + glob.glob("*.bmp") + glob.glob("*.gif") + glob.glob("*.webp")
        self.rag_system = AgenticRAGSystem(file_paths=file_paths)

    async def run(self):
        """Ana CLI döngüsü"""
        console.print("[bold green]🤖 Agentic RAG Sistemi Başlatıldı[/bold green]")

        # Sistem istatistikleri
        stats = self.rag_system.get_statistics()
        self._display_statistics(stats)

        while True:
            try:
                question = input("\n💭 Sorunuz (çıkmak için 'q'): ").strip()

                if question.lower() in ['q', 'quit', 'exit']:
                    console.print("[yellow]Görüşmek üzere! 👋[/yellow]")
                    break

                if not question:
                    continue

                # Sorgu işleme
                with Progress() as progress:
                    task = progress.add_task("🔍 Aranıyor...", total=100)

                    # Async sorgu
                    response = await self.rag_system.query(question)
                    progress.update(task, completed=100)

                # Sonuçları göster
                self._display_results(response)

            except KeyboardInterrupt:
                console.print("\n[yellow]Çıkılıyor...[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Hata: {str(e)}[/red]")

    def _display_statistics(self, stats: Dict[str, Any]):
        """Sistem istatistiklerini göster"""
        table = Table(title="📊 Sistem İstatistikleri")
        table.add_column("Metrik", style="cyan")
        table.add_column("Değer", style="green")

        table.add_row("Toplam Belge", str(stats['total_documents']))
        table.add_row("Toplam Kelime", str(stats['total_words']))
        table.add_row("Ortalama Belge Uzunluğu", f"{stats['average_document_length']:.1f}")
        table.add_row("Mevcut Stratejiler", ", ".join(stats['available_strategies']))

        console.print(table)

    def _display_results(self, response: Dict[str, Any]):
        """Sonuçları görüntüle (görsel ise yolunu ve küçük önizleme notunu göster)"""
        console.print(f"\n[bold blue]🧠 Sorgu Analizi[/bold blue]")
        query_ctx = response['query_context']
        console.print(f"📝 Amaç: {query_ctx['intent']}")
        console.print(f"🎯 Tür: {query_ctx['query_type']}")
        console.print(f"📊 Karmaşıklık: {query_ctx['complexity_score']:.2f}")

        if response['results']:
            console.print(f"\n[bold green]🎯 En İlgili Sonuçlar[/bold green]")

            for i, result in enumerate(response['results'][:3], 1):
                doc = result['document']
                console.print(f"\n{i}. [bold]Skor: {result['score']:.4f}[/bold] | Yöntem: {result['retrieval_method']}")
                console.print(f"📍 Kaynak: Satır {doc['source_line']}")
                console.print(f"🔤 Eşleşen terimler: {', '.join(result['matched_terms'])}")

                # Görsel ise yolunu göster
                if doc['metadata'].get('type') == 'image':
                    panel_text = f"[bold yellow]Görsel dosyası:[/bold yellow]\n{doc['metadata']['source']}\n[dim]Görselde aranan kelime dosya adında bulundu.[/dim]"
                    console.print(Panel(panel_text, border_style="magenta"))
                else:
                    # Highlighted content
                    highlighted_content = self._highlight_terms(doc['content'], result['matched_terms'])
                    console.print(Panel(highlighted_content, border_style="green"))
        else:
            console.print("[italic red]İlgili sonuç bulunamadı.[/italic red]")

        # Generated Answer
        console.print(f"\n[bold yellow]💡 Üretilen Cevap[/bold yellow]")
        console.print(Panel(response['answer'], border_style="blue"))

    def _highlight_terms(self, content: str, terms: List[str]) -> str:
        """Terimleri vurgula"""
        highlighted = content
        for term in terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"[bold red]{term}[/bold red]", highlighted)
        return highlighted

# Ana çalıştırma
async def main():
    """Ana asenkron fonksiyon"""
    cli = EnhancedCLI()
    await cli.run()

def run_system():
    """Sistem çalıştırıcı - notebook uyumlu"""
    try:
        # Mevcut event loop'u kontrol et
        loop = asyncio.get_running_loop()
        # Zaten event loop varsa, nest_asyncio kullan
        if NEST_ASYNCIO_AVAILABLE:
            nest_asyncio.apply()
            asyncio.run(main())
        else:
            console.print("[yellow]Jupyter/Colab'da çalışıyorsunuz. Senkron mod aktif.[/yellow]")
            run_sync_version()
    except RuntimeError:
        # Event loop yoksa normal çalıştır
        asyncio.run(main())
    except Exception as e:
        console.print(f"[red]Async hata: {e}[/red]")
        console.print("[yellow]Senkron versiyona geçiliyor...[/yellow]")
        run_sync_version()

def run_sync_version():
    """Senkron versiyon - uyumluluk için"""
    console.print("[bold green]🤖 Agentic RAG Sistemi Başlatıldı (Sync Mode)[/bold green]")

    rag_system = AgenticRAGSystem()
    stats = rag_system.get_statistics()

    # Basit stats gösterimi
    console.print(f"📊 Yüklenen belgeler: {stats['total_documents']}")
    console.print(f"📝 Toplam kelime: {stats['total_words']}")

    while True:
        try:
            question = input("\n💭 Sorunuz (çıkmak için 'q'): ").strip()

            if question.lower() in ['q', 'quit', 'exit']:
                console.print("[yellow]Görüşmek üzere! 👋[/yellow]")
                break

            if not question:
                continue

            # Senkron query işleme
            console.print("🔍 Aranıyor...")

            # Query context oluştur
            query_context = rag_system.query_analyzer.analyze_query(question)

            # Retrieval yap
            results = rag_system.retrieval_router.retrieve(query_context, rag_system.documents, 3)

            # Sonuçları göster
            display_sync_results(query_context, results)

        except KeyboardInterrupt:
            console.print("\n[yellow]Çıkılıyor...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Hata: {str(e)}[/red]")

def display_sync_results(query_context, results):
    """Senkron sonuç gösterimi"""
    console.print(f"\n[bold blue]🧠 Sorgu Analizi[/bold blue]")
    console.print(f"📝 Amaç: {query_context.intent}")
    console.print(f"🎯 Tür: {query_context.query_type}")
    console.print(f"📊 Karmaşıklık: {query_context.complexity_score:.2f}")

    if results:
        console.print(f"\n[bold green]🎯 En İlgili Sonuçlar[/bold green]")

        for i, result in enumerate(results[:3], 1):
            console.print(f"\n{i}. [bold]Skor: {result.score:.4f}[/bold] | Yöntem: {result.retrieval_method}")
            console.print(f"📍 Kaynak: Satır {result.document.source_line}")
            console.print(f"🔤 Eşleşen terimler: {', '.join(result.matched_terms)}")

            # Highlighted content
            highlighted = highlight_terms_simple(result.document.content, result.matched_terms)
            console.print(Panel(highlighted, border_style="green"))
    else:
        console.print("[italic red]İlgili sonuç bulunamadı.[/italic red]")

def highlight_terms_simple(content: str, terms: List[str]) -> str:
    """Basit term highlighting"""
    highlighted = content
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted = pattern.sub(f"[bold red]{term}[/bold red]", highlighted)
    return highlighted

if __name__ == "__main__":
    # Jupyter/Colab için uyumlu çalıştırma
    try:
        run_system()
    except Exception as e:
        console.print(f"[red]Sistem başlatma hatası: {e}[/red]")
        console.print("[yellow]Senkron versiyonu deniyor...[/yellow]")
        run_sync_version()