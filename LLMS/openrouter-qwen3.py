import requests
import json
import time
from rich.console import Console
from rich.panel import Panel
from rich import box
from rich.table import Table
from rich.markdown import Markdown
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import re
import argparse
from typing import Dict, List

# Initialize Rich console
console = Console()

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_completion(model_name, prompt):
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
    
    return response, elapsed_time

def analyze_results(results):
    """Analyze and provide insights about the model comparison results."""
    if len(results) < 2:
        return "Not enough models to compare."
        
    analysis = []
    
    # Compare response times
    fastest_model = min(results, key=lambda x: x['response_time'])
    slowest_model = max(results, key=lambda x: x['response_time'])
    time_diff = slowest_model['response_time'] - fastest_model['response_time']
    analysis.append(f"- {fastest_model['model']} is {time_diff:.2f}s faster than {slowest_model['model']} ({(time_diff/slowest_model['response_time']*100):.1f}% speedup)")
    
    # Compare token efficiency
    most_tokens = max(results, key=lambda x: x['completion_tokens'])
    least_tokens = min(results, key=lambda x: x['completion_tokens'])
    token_ratio = most_tokens['completion_tokens'] / least_tokens['completion_tokens'] if least_tokens['completion_tokens'] > 0 else 0
    analysis.append(f"- {most_tokens['model']} produced {token_ratio:.1f}x more tokens than {least_tokens['model']}")
    
    # Compare BLEU scores
    best_bleu = max(results, key=lambda x: x['bleu_1'])
    worst_bleu = min(results, key=lambda x: x['bleu_1'])
    analysis.append(f"- {best_bleu['model']} has {best_bleu['bleu_1']:.4f} BLEU-1 score vs {worst_bleu['model']}'s {worst_bleu['bleu_1']:.4f}")
    
    # Compare lexical diversity
    most_diverse = max(results, key=lambda x: x['lexical_diversity'])
    least_diverse = min(results, key=lambda x: x['lexical_diversity'])
    analysis.append(f"- {most_diverse['model']} has higher lexical diversity ({most_diverse['lexical_diversity']:.4f}) than {least_diverse['model']} ({least_diverse['lexical_diversity']:.4f})")
    
    # Provide overall assessment
    analysis.append("\n**Overall Assessment:**")
    
    # Assess which model is better for different use cases
    concise_model = min(results, key=lambda x: x['response_length'])
    detailed_model = max(results, key=lambda x: x['response_length'])
    analysis.append(f"- {concise_model['model']} provides more concise answers, suitable for quick responses")
    analysis.append(f"- {detailed_model['model']} gives more detailed explanations, better for comprehensive information")
    
    return "\n".join(analysis)

def generate_prompts():
    """Generate a list of prompts for testing."""
    return [
        "What is the meaning of life?",
        "Explain quantum computing in simple terms",
        "Write a short poem about nature",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis"
    ]

# Main function
def main():
    parser = argparse.ArgumentParser(description='Compare language models using OpenRouter API')
    parser.add_argument('--prompt', type=str, help='Custom prompt to use')
    parser.add_argument('--models', type=str, nargs='+', default=['qwen/qwen3-0.6b-04-28:free', 'qwen/qwen3-1.7b:free'], 
                        help='Models to compare (default: qwen3-0.6b and qwen3-1.7b)')
    args = parser.parse_args()
    
    # User prompt
    user_prompt = args.prompt if args.prompt else "What is the meaning of life?"
    
    # Reference answer (ideally from a high-quality source)
    reference_answer = """The meaning of life is a philosophical question concerning the purpose and significance of life or existence in general. 
    Different perspectives include pursuing happiness, wisdom, love, or fulfillment of one's potential. 
    Some emphasize relationships, serving others, personal growth, or contributing to society. 
    Various philosophical and religious traditions offer different answers, and many individuals create their own meaning through personal choices and experiences."""
    
    # Models to compare
    models = []
    for model_name in args.models:
        display_name = model_name.split('/')[-1].split(':')[0].upper()
        models.append({"name": model_name, "display_name": display_name})
    
    # Collect results
    results = []
    
    # Make requests to both models
    for model in models:
        console.print(f"\n[bold blue]Requesting response from {model['display_name']}...[/bold blue]")
        response, elapsed_time = get_completion(model["name"], user_prompt)
        
        if response.status_code == 200:
            response_data = response.json()
            message = response_data["choices"][0]["message"]
            
            # Display normal response in a panel
            console.print(Panel(
                message['content'],
                title=f"[bold green]{model['display_name']} Response[/bold green]",
                subtitle=f"Role: {message['role']}",
                box=box.ROUNDED,
                border_style="green"
            ))
            
            # Display thinking/reasoning if available
            if 'reasoning' in message and message['reasoning']:
                console.print(Panel(
                    message['reasoning'],
                    title=f"[bold yellow]{model['display_name']} Thinking Process[/bold yellow]",
                    box=box.ROUNDED,
                    border_style="yellow"
                ))
            
            # Calculate BLEU score
            response_text = message['content']
            
            # Tokenize reference and candidate text
            reference_tokens = [word_tokenize(reference_answer.lower())]
            candidate_tokens = word_tokenize(response_text.lower())
            
            # Calculate BLEU scores with different n-gram weights
            bleu_1 = sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0))
            bleu_2 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0))
            bleu_4 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25))
            
            # Calculate response length and uniqueness metrics
            words = re.findall(r'\w+', response_text.lower())
            unique_words = set(words)
            lexical_diversity = len(unique_words) / len(words) if words else 0
            
            # Store metrics
            results.append({
                "model": model["display_name"],
                "response_time": elapsed_time,
                "token_count": response_data["usage"]["total_tokens"],
                "prompt_tokens": response_data["usage"]["prompt_tokens"],
                "completion_tokens": response_data["usage"]["completion_tokens"],
                "finish_reason": response_data["choices"][0]["finish_reason"],
                "bleu_1": bleu_1,
                "bleu_2": bleu_2,
                "bleu_4": bleu_4,
                "response_length": len(words),
                "unique_words": len(unique_words),
                "lexical_diversity": lexical_diversity
            })
        else:
            console.print(f"[bold red]Error with {model['display_name']}:[/bold red] {response.text}")

    # Compare metrics
    if len(results) > 0:
        console.print("\n[bold cyan]Performance Comparison[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        for result in results:
            table.add_column(result["model"])
        
        # Add rows for each metric
        table.add_row("Response Time (s)", *[f"{r['response_time']:.2f}" for r in results])
        table.add_row("Total Tokens", *[f"{r['token_count']}" for r in results])
        table.add_row("Prompt Tokens", *[f"{r['prompt_tokens']}" for r in results])
        table.add_row("Completion Tokens", *[f"{r['completion_tokens']}" for r in results])
        table.add_row("Finish Reason", *[f"{r['finish_reason']}" for r in results])
        
        # Add BLEU score metrics
        table.add_row("BLEU-1 Score", *[f"{r['bleu_1']:.4f}" for r in results])
        table.add_row("BLEU-2 Score", *[f"{r['bleu_2']:.4f}" for r in results])
        table.add_row("BLEU-4 Score", *[f"{r['bleu_4']:.4f}" for r in results])
        
        # Add lexical metrics
        table.add_row("Response Length (words)", *[f"{r['response_length']}" for r in results])
        table.add_row("Unique Words", *[f"{r['unique_words']}" for r in results])
        table.add_row("Lexical Diversity", *[f"{r['lexical_diversity']:.4f}" for r in results])
        
        console.print(table)
        
        # Show analysis
        analysis = analyze_results(results)
        console.print("\n[bold blue]Analysis[/bold blue]")
        console.print(Markdown(analysis))
        
        console.print("\n[bold blue]Metrics Explanation[/bold blue]")
        metrics_explanation = """
        - **BLEU Scores**: Measure similarity between model output and reference text (higher is better)
        - **BLEU-1**: Compares individual words
        - **BLEU-2**: Compares pairs of adjacent words
        - **BLEU-4**: Compares sequences of up to 4 adjacent words
        - **Response Length**: Total number of words in the response
        - **Unique Words**: Number of distinct words used
        - **Lexical Diversity**: Ratio of unique words to total words (higher means more diverse vocabulary)
        """
        console.print(Markdown(metrics_explanation))

if __name__ == "__main__":
    main()