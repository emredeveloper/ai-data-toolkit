import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import re

endpoint = "https://models.github.ai/inference"
model = "microsoft/Phi-4-reasoning"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

console = Console()

def get_model_response(prompt, with_thinking=True):
    messages = []
    
    if with_thinking:
        # Simplified system message
        messages.append(SystemMessage(
            "Put your thinking inside <think> tags and your final answer after the tags.\n"
            "Example: <think>This is my reasoning...</think>\nThis is my answer."
        ))
    
    messages.append(UserMessage(prompt))
    
    response = client.complete(
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model=model
    )
    
    content = response.choices[0].message.content
    
    # If thinking tags aren't present but with_thinking was requested, 
    # artificially add them to ensure proper display
    if with_thinking and ("<think>" not in content or "</think>" not in content):
        # Check if there's likely a thinking part by looking for reasoning patterns
        if len(content.split("\n\n")) > 1:
            # Assume first paragraphs are thinking and last paragraph is answer
            parts = content.split("\n\n")
            thinking_part = "\n\n".join(parts[:-1])
            answer_part = parts[-1]
            content = f"<think>{thinking_part}</think>\n\n{answer_part}"
        else:
            # If we can't split nicely, just wrap everything as thinking with a generic answer
            content = f"<think>{content}</think>\n\nBased on my analysis, {prompt}"
    
    return content

def display_response(response_text):
    # Improved regex pattern to reliably extract thinking part
    thinking_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    match = thinking_pattern.search(response_text)
    
    if match:
        thinking_part = match.group(1).strip()
        # Get everything after the last </think> tag
        answer_part = re.sub(r'.*?</think>', '', response_text, flags=re.DOTALL).strip()
        
        console.print(Panel(
            Markdown(thinking_part),
            title="Thinking Process",
            border_style="yellow"
        ))
        
        console.print(Panel(
            Markdown(answer_part),
            title="Final Answer",
            border_style="green"
        ))
    else:
        # If no think tags, show the whole response
        console.print(Panel(
            Markdown(response_text),
            title="Response",
            border_style="blue"
        ))

# Example usage - you can modify this prompt
prompt = "What is the capital of France?"
response_text = get_model_response(prompt)
display_response(response_text)