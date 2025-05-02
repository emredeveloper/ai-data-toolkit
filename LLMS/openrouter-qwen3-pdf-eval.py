import requests
import json
import time
import argparse
import PyPDF2
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import os
import numpy as np
import math
from collections import Counter
from typing import Dict, List, Tuple, Union

# Initialize Rich console
console = Console()

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            console.print(f"[bold green]Extracting text from PDF with {num_pages} pages[/bold green]")
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
        return text
    except Exception as e:
        console.print(f"[bold red]Error extracting text from PDF:[/bold red] {str(e)}")
        return ""

def generate_questions_from_pdf(pdf_text: str, num_questions: int = 2) -> List[str]:
    """Generate questions based on the PDF content."""
    # For demonstration, we'll use simple predefined questions
    # In a real application, you might want to use an LLM to generate questions
    
    questions = [
        "What is the main topic of this document?",
        "Summarize the key points in this document.",
        "What are the most important findings or conclusions?",
        "Explain any technical terms mentioned in the document.",
        "What recommendations or next steps are suggested?",
    ]
    
    return questions[:num_questions]

def get_completion(model_name: str, prompt: str) -> Tuple[dict, float]:
    """Get completion from OpenRouter API."""
    start_time = time.time()
    
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer key",
            "Content-Type": "application/json",
            "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
        },
        data=json.dumps({
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        })
    )
    
    elapsed_time = time.time() - start_time
    
    if response.status_code == 200:
        return response.json(), elapsed_time
    else:
        console.print(f"[bold red]Error:[/bold red] {response.text}")
        return None, elapsed_time

def truncate_text(text: str, max_tokens: int = 6000) -> str:
    """Truncate text to stay within token limits."""
    # Approximate token count by words (a rough estimate)
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens]) + "..."
    return text

def calculate_perplexity(text: str) -> float:
    """Calculate a simplified perplexity estimate for a text."""
    # This is a simplified perplexity calculation
    # In a real scenario, you would use a dedicated language model
    words = re.findall(r'\w+', text.lower())
    if len(words) <= 1:
        return float('inf')
    
    # Compute unigram distribution
    word_counts = Counter(words)
    total_words = len(words)
    
    # Calculate simple perplexity
    entropy = 0
    for word in words:
        prob = word_counts[word] / total_words
        entropy -= math.log2(prob)
    
    # Average entropy over all words
    avg_entropy = entropy / total_words
    perplexity = 2 ** avg_entropy
    
    return perplexity

def calculate_bleu_with_context(reference_text: str, candidate_text: str) -> Dict[str, float]:
    """Calculate BLEU score with context awareness."""
    # Tokenize the texts
    reference_sents = sent_tokenize(reference_text.lower())
    candidate_sents = sent_tokenize(candidate_text.lower())
    
    # Tokenize each sentence
    reference_tokens = [word_tokenize(sent) for sent in reference_sents]
    candidate_tokens = [word_tokenize(sent) for sent in candidate_sents]
    
    # Flatten for overall BLEU
    flat_reference = [word for sent in reference_tokens for word in sent]
    flat_candidate = [word for sent in candidate_tokens for word in sent]
    
    # Calculate BLEU scores
    bleu_1 = sentence_bleu([flat_reference], flat_candidate, weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu([flat_reference], flat_candidate, weights=(0.5, 0.5, 0, 0))
    bleu_4 = sentence_bleu([flat_reference], flat_candidate, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Calculate sentence-level BLEU and average
    sent_bleus = []
    for cand_sent in candidate_tokens:
        best_bleu = 0
        for ref_sent in reference_tokens:
            if len(ref_sent) > 3 and len(cand_sent) > 3:  # Ensure meaningful comparison
                bleu = sentence_bleu([ref_sent], cand_sent, weights=(0.25, 0.25, 0.25, 0.25))
                best_bleu = max(best_bleu, bleu)
        if best_bleu > 0:
            sent_bleus.append(best_bleu)
    
    avg_sent_bleu = sum(sent_bleus) / len(sent_bleus) if sent_bleus else 0
    
    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_4": bleu_4,
        "avg_sent_bleu": avg_sent_bleu
    }

def calculate_relevance_score(pdf_text: str, response_text: str) -> float:
    """Calculate relevance score between PDF text and response."""
    # Extract key terms from PDF (using simple frequency)
    pdf_words = re.findall(r'\w+', pdf_text.lower())
    pdf_word_counts = Counter(pdf_words)
    
    # Get top keywords (excluding common words)
    common_words = set(['the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'was', 'for', 'it', 'with', 'as', 'on'])
    keywords = [word for word, count in pdf_word_counts.most_common(50) if word not in common_words and len(word) > 2]
    
    # Count keyword occurrences in response
    response_words = re.findall(r'\w+', response_text.lower())
    keyword_matches = sum(1 for word in response_words if word in keywords)
    
    # Calculate relevance score (0-1)
    relevance = min(1.0, keyword_matches / (len(response_words) * 0.1)) if response_words else 0
    
    return relevance

def calculate_complexity_metrics(text: str) -> Dict:
    """Calculate complexity metrics for a text response."""
    words = re.findall(r'\w+', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    
    # Calculate metrics
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    unique_words = set(words)
    lexical_diversity = len(unique_words) / len(words) if words else 0
    
    # Calculate Flesch Reading Ease
    word_count = len(words)
    sentence_count = len(sentences)
    syllable_count = sum([count_syllables(word) for word in words])
    
    if sentence_count == 0 or word_count == 0:
        flesch_score = 0
    else:
        flesch_score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100
    
    # Calculate perplexity
    perplexity = calculate_perplexity(text)
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_words": len(unique_words),
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity,
        "flesch_reading_ease": flesch_score,
        "perplexity": perplexity
    }

def count_syllables(word: str) -> int:
    """Count the number of syllables in a word."""
    # This is a simple syllable counter and not fully accurate
    word = word.lower()
    # Count vowel groups
    if len(word) <= 3:
        return 1
    count = 0
    vowels = "aeiouy"
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    
    # Adjust for common patterns
    if word.endswith('e') and not word.endswith('le'):
        count -= 1
    if count == 0:
        count = 1
    return count

def evaluate_pdf_comprehension(pdf_path: str, models: List[Dict], questions: List[str] = None):
    """Evaluate how well models understand and answer questions about a PDF."""
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        console.print("[bold red]Failed to extract text from PDF. Exiting.[/bold red]")
        return
    
    # Truncate text to avoid token limits
    pdf_text = truncate_text(pdf_text)
    
    # Generate questions if not provided
    if not questions:
        questions = generate_questions_from_pdf(pdf_text)
    
    console.print(f"[bold blue]Generated {len(questions)} questions for evaluation:[/bold blue]")
    for i, q in enumerate(questions, 1):
        console.print(f"{i}. {q}")
    
    # Store results for each model
    all_results = []
    
    # Process each model
    for model in models:
        console.print(f"\n[bold cyan]Evaluating {model['display_name']}...[/bold cyan]")
        model_results = {
            "model": model["display_name"],
            "responses": [],
            "metrics": {
                "avg_response_time": 0,
                "total_tokens": 0,
                "avg_tokens_per_question": 0,
                "avg_relevance": 0,
                "avg_perplexity": 0,
                "avg_bleu": 0,
            }
        }
        
        total_time = 0
        total_tokens = 0
        total_relevance = 0
        total_perplexity = 0
        total_bleu = 0
        
        # Process each question
        for i, question in enumerate(questions, 1):
            prompt = f"""Please answer the following question based on this document:

Document content:
{pdf_text}

Question {i}: {question}

Please provide a detailed and accurate answer based solely on the information in the document. If the document doesn't contain relevant information to answer the question, please state that clearly.
"""
            
            console.print(f"[bold]Processing Question {i}...[/bold]")
            response_data, elapsed_time = get_completion(model["name"], prompt)
            
            if response_data:
                message = response_data["choices"][0]["message"]
                
                # Display response
                console.print(Panel(
                    message["content"],
                    title=f"[bold green]Question {i} - {model['display_name']} Response[/bold green]",
                    border_style="green"
                ))
                
                # Display thinking/reasoning if available
                if 'reasoning' in message and message['reasoning']:
                    console.print(Panel(
                        message['reasoning'],
                        title=f"[bold yellow]Question {i} - {model['display_name']} Thinking Process[/bold yellow]",
                        box=box.ROUNDED,
                        border_style="yellow"
                    ))
                
                # Get token usage information safely
                tokens = 0
                completion_tokens = 0
                
                # Safely extract token usage information
                if "usage" in response_data:
                    tokens = response_data["usage"].get("total_tokens", 0)
                    completion_tokens = response_data["usage"].get("completion_tokens", 0)
                else:
                    # Estimate tokens if not provided by API
                    # Rough estimate: 1 token ≈ 4 characters for English text
                    tokens = len(prompt) // 4 + len(message["content"]) // 4
                    completion_tokens = len(message["content"]) // 4
                    console.print(f"[yellow]Warning: Token usage information not available. Using rough estimates.[/yellow]")
                
                # Calculate additional metrics
                response_text = message["content"]
                complexity_metrics = calculate_complexity_metrics(response_text)
                relevance_score = calculate_relevance_score(pdf_text, response_text)
                bleu_metrics = calculate_bleu_with_context(pdf_text, response_text)
                
                # Store response metrics
                model_results["responses"].append({
                    "question": question,
                    "response": response_text,
                    "response_time": elapsed_time,
                    "tokens": tokens,
                    "completion_tokens": completion_tokens,
                    "complexity_metrics": complexity_metrics,
                    "relevance_score": relevance_score,
                    "bleu_metrics": bleu_metrics
                })
                
                total_time += elapsed_time
                total_tokens += tokens
                total_relevance += relevance_score
                total_perplexity += complexity_metrics["perplexity"]
                total_bleu += bleu_metrics["avg_sent_bleu"]
            
        # Calculate aggregate metrics
        num_questions = len(questions)
        if num_questions > 0:
            model_results["metrics"]["avg_response_time"] = total_time / num_questions
            model_results["metrics"]["total_tokens"] = total_tokens
            model_results["metrics"]["avg_tokens_per_question"] = total_tokens / num_questions
            model_results["metrics"]["avg_relevance"] = total_relevance / num_questions
            model_results["metrics"]["avg_perplexity"] = total_perplexity / num_questions
            model_results["metrics"]["avg_bleu"] = total_bleu / num_questions
        
        all_results.append(model_results)
    
    # Compare and display results
    if all_results:
        display_comparison_results(all_results, questions)

def display_comparison_results(results: List[Dict], questions: List[str]):
    """Display comparison of model results."""
    # Performance metrics table
    console.print("\n[bold blue]Performance Metrics[/bold blue]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    for result in results:
        table.add_column(result["model"])
    
    # Add rows for performance metrics
    table.add_row("Avg. Response Time (s)", 
                 *[f"{r['metrics']['avg_response_time']:.2f}" for r in results])
    table.add_row("Total Tokens Used", 
                 *[f"{r['metrics']['total_tokens']}" for r in results])
    table.add_row("Avg. Tokens per Question", 
                 *[f"{r['metrics']['avg_tokens_per_question']:.1f}" for r in results])
    table.add_row("Avg. PDF Relevance Score", 
                 *[f"{r['metrics']['avg_relevance']:.4f}" for r in results])
    table.add_row("Avg. BLEU Score", 
                 *[f"{r['metrics']['avg_bleu']:.4f}" for r in results])
    table.add_row("Avg. Perplexity", 
                 *[f"{r['metrics']['avg_perplexity']:.2f}" for r in results])
    
    console.print(table)
    
    # Calculate and display response quality metrics
    console.print("\n[bold blue]Response Quality Metrics[/bold blue]")
    
    for i, question in enumerate(questions):
        console.print(f"\n[bold]Question {i+1}:[/bold] {question}")
        
        quality_table = Table(show_header=True, header_style="bold cyan")
        quality_table.add_column("Metric")
        for result in results:
            quality_table.add_column(result["model"])
        
        # Get the metrics for each response
        response_metrics = []
        for result in results:
            metrics = result["responses"][i]
            response_metrics.append(metrics)
        
        # Add rows for complexity metrics
        quality_table.add_row("Word Count", 
                             *[f"{m['complexity_metrics']['word_count']}" for m in response_metrics])
        quality_table.add_row("Sentence Count", 
                             *[f"{m['complexity_metrics']['sentence_count']}" for m in response_metrics])
        quality_table.add_row("Unique Words", 
                             *[f"{m['complexity_metrics']['unique_words']}" for m in response_metrics])
        quality_table.add_row("Lexical Diversity", 
                             *[f"{m['complexity_metrics']['lexical_diversity']:.3f}" for m in response_metrics])
        quality_table.add_row("Reading Ease", 
                             *[f"{m['complexity_metrics']['flesch_reading_ease']:.1f}" for m in response_metrics])
        quality_table.add_row("Perplexity", 
                             *[f"{m['complexity_metrics']['perplexity']:.2f}" for m in response_metrics])
        quality_table.add_row("PDF Relevance", 
                             *[f"{m['relevance_score']:.4f}" for m in response_metrics])
        quality_table.add_row("BLEU-1 Score", 
                             *[f"{m['bleu_metrics']['bleu_1']:.4f}" for m in response_metrics])
        quality_table.add_row("BLEU-2 Score", 
                             *[f"{m['bleu_metrics']['bleu_2']:.4f}" for m in response_metrics])
        quality_table.add_row("BLEU-4 Score", 
                             *[f"{m['bleu_metrics']['bleu_4']:.4f}" for m in response_metrics])
        quality_table.add_row("Sent-BLEU Average", 
                             *[f"{m['bleu_metrics']['avg_sent_bleu']:.4f}" for m in response_metrics])
        
        console.print(quality_table)
    
    # Overall analysis
    console.print("\n[bold blue]Overall Analysis[/bold blue]")
    
    # Find best models for different metrics
    faster_model = min(results, key=lambda r: r["metrics"]["avg_response_time"])
    most_relevant_model = max(results, key=lambda r: r["metrics"]["avg_relevance"])
    best_bleu_model = max(results, key=lambda r: r["metrics"]["avg_bleu"])
    lowest_perplexity_model = min(results, key=lambda r: r["metrics"]["avg_perplexity"])
    
    analysis = f"""
    **Efficiency Metrics:**
    - {faster_model['model']} is faster, with an average response time of {faster_model['metrics']['avg_response_time']:.2f}s
    
    **PDF Understanding Metrics:**
    - {most_relevant_model['model']} has the highest PDF relevance score ({most_relevant_model['metrics']['avg_relevance']:.4f})
    - {best_bleu_model['model']} has the highest average BLEU score ({best_bleu_model['metrics']['avg_bleu']:.4f})
    
    **Output Quality Metrics:**
    - {lowest_perplexity_model['model']} has the lowest perplexity ({lowest_perplexity_model['metrics']['avg_perplexity']:.2f}), indicating more fluent responses
    
    **Recommendations:**
    - For most accurate PDF information extraction: {most_relevant_model['model']}
    - For most fluent outputs: {lowest_perplexity_model['model']}
    - For best overall performance: {best_bleu_model['model'] if best_bleu_model['metrics']['avg_bleu'] > 0.1 else most_relevant_model['model']}
    """
    
    console.print(Markdown(analysis))
    
    console.print("\n[bold blue]Metrics Explanation[/bold blue]")
    metrics_explanation = """
    **PDF Understanding Metrics:**
    - **PDF Relevance Score**: Measures how well the response contains key terms from the PDF (0-1, higher is better)
    - **BLEU Scores**: Compares n-grams between the PDF and response (0-1, higher indicates better representation)
    - **Sent-BLEU Average**: BLEU scores at the sentence level, showing finer-grained similarity
    
    **Output Quality Metrics:**
    - **Perplexity**: Measures how predictable the text is (lower is better, usually between 5-50 for fluent text)
    - **Lexical Diversity**: Ratio of unique words to total words (higher indicates more diverse vocabulary)
    - **Reading Ease**: Flesch Reading Ease score (0-100, higher is easier to read)
    
    **Additional Metrics:**
    - **Word/Sentence Count**: Raw volume of the response
    - **Avg Sentence Length**: Average words per sentence (indicator of complexity)
    """
    console.print(Markdown(metrics_explanation))

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM comprehension of PDF documents")
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["qwen/qwen3-0.6b-04-28:free", "qwen/qwen3-1.7b:free"],
                       help="Models to evaluate (default: qwen3-0.6b and qwen3-1.7b)")
    parser.add_argument("--questions", type=str, nargs="+", help="Custom questions to ask about the PDF")
    
    args = parser.parse_args()
    
    # Check if PDF exists
    if not os.path.exists(args.pdf):
        console.print(f"[bold red]Error: PDF file not found at {args.pdf}[/bold red]")
        return
    
    # Format models for processing
    models = []
    for model_name in args.models:
        display_name = model_name.split('/')[-1].split(':')[0].upper()
        models.append({"name": model_name, "display_name": display_name})
    
    # Run evaluation
    evaluate_pdf_comprehension(args.pdf, models, args.questions)

if __name__ == "__main__":
    main()
