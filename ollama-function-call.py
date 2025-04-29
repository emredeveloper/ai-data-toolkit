import ollama
from typing import Dict, Any, Callable, List
import json
import logging
from pydantic import BaseModel, ValidationError
from datetime import datetime
import requests
import xml.etree.ElementTree as ET

# Loglama ayarları
logging.basicConfig(
    filename='ollama_tools.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Araçları saklamak için global kayıt
TOOLS: Dict[str, Dict[str, Any]] = {}

class ToolSchema(BaseModel):
    """Base Pydantic model for validating tool parameters."""
    pass

class AddNumbersSchema(ToolSchema):
    a: int
    b: int

class MultiplyNumbersSchema(ToolSchema):
    a: int
    b: int

# Yeni tool şemaları
class ConcatTextSchema(ToolSchema):
    text1: str
    text2: str

class SumListSchema(ToolSchema):
    numbers: List[int]

class ReverseTextSchema(ToolSchema):
    text: str

class GetCurrentTimeSchema(ToolSchema):
    format: str = "%Y-%m-%d %H:%M:%S"

class RecentArxivSchema(ToolSchema):
    """Schema for fetching recent arXiv papers."""
    query: str
    max_results: int = 10
    category: str = "cs.AI"  # Default to AI/ML papers

def register_tool(
    name: str,
    description: str,
    func: Callable,
    schema: BaseModel,
    parameters: Dict[str, Any]
) -> None:
    """Register a new tool with its function, schema, and parameters."""
    TOOLS[name] = {
        "func": func,
        "schema": schema,
        "tool": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
    }
    logging.info(f"Registered tool: {name}")

# Fonksiyon tanımları
def add_two_numbers(params: AddNumbersSchema) -> int:
    """Add two numbers."""
    return params.a + params.b

def multiply_two_numbers(params: MultiplyNumbersSchema) -> int:
    """Multiply two numbers."""
    return params.a * params.b

# Yeni tool fonksiyonları
def concat_text(params: ConcatTextSchema) -> str:
    """Concatenate two texts."""
    return params.text1 + params.text2

def sum_list(params: SumListSchema) -> int:
    """Sum a list of numbers."""
    return sum(params.numbers)

def reverse_text(params: ReverseTextSchema) -> str:
    """Reverse the given text."""
    return params.text[::-1]

def get_current_time(params: GetCurrentTimeSchema) -> str:
    """Get the current date and time in the specified format."""
    return datetime.now().strftime(params.format)

def fetch_recent_arxiv(params: RecentArxivSchema) -> List[dict]:
    """
    Fetch recent arXiv papers using the official API.
    """
    base_url = "http://export.arxiv.org/api/query"
    # Search in specific category and sort by submission date
    search_query = f"cat:{params.category}"
    if params.query:
        search_query += f" AND all:{params.query}"
    
    payload = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": params.max_results
    }
    
    try:
        response = requests.get(base_url, params=payload, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        entries = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            summary = entry.find('atom:summary', ns).text.strip()
            published = entry.find('atom:published', ns).text.strip()
            
            # Get the proper arxiv link
            links = entry.findall('atom:link', ns)
            pdf_link = next((l.get('href') for l in links if l.get('title') == 'pdf'), '')
            html_link = next((l.get('href') for l in links if l.get('rel') == 'alternate'), '')
            
            # Get categories/topics
            categories = [cat.get('term') for cat in entry.findall('arxiv:primary_category', ns)]
            
            entries.append({
                "title": title,
                "summary": summary,
                "published": published,
                "pdf_link": pdf_link,
                "html_link": html_link,
                "categories": categories
            })
        
        if not entries:
            return [{"status": "no_results", 
                    "message": f"No recent papers found for query: '{params.query}' in category {params.category}",
                    "query": params.query}]
        
        return entries
        
    except Exception as e:
        logging.error(f"arXiv API error: {str(e)}")
        return [{"status": "error", "message": str(e)}]

# Araçları kaydet
register_tool(
    name="add_two_numbers",
    description="Add two numbers",
    func=add_two_numbers,
    schema=AddNumbersSchema,
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "The first number"},
            "b": {"type": "integer", "description": "The second number"},
        },
        "required": ["a", "b"],
    }
)

register_tool(
    name="multiply_two_numbers",
    description="Multiply two numbers",
    func=multiply_two_numbers,
    schema=MultiplyNumbersSchema,
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "The first number"},
            "b": {"type": "integer", "description": "The second number"},
        },
        "required": ["a", "b"],
    }
)

# Yeni araçları kaydet
register_tool(
    name="concat_text",
    description="Concatenate two texts",
    func=concat_text,
    schema=ConcatTextSchema,
    parameters={
        "type": "object",
        "properties": {
            "text1": {"type": "string", "description": "First text"},
            "text2": {"type": "string", "description": "Second text"},
        },
        "required": ["text1", "text2"],
    }
)

register_tool(
    name="sum_list",
    description="Sum a list of numbers",
    func=sum_list,
    schema=SumListSchema,
    parameters={
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of numbers to sum"
            }
        },
        "required": ["numbers"],
    }
)

register_tool(
    name="reverse_text",
    description="Reverse the given text",
    func=reverse_text,
    schema=ReverseTextSchema,
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to reverse"},
        },
        "required": ["text"],
    }
)

register_tool(
    name="get_current_time",
    description="Get the current date and time in the specified format",
    func=get_current_time,
    schema=GetCurrentTimeSchema,
    parameters={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "Datetime format string (default: '%Y-%m-%d %H:%M:%S')"
            }
        },
        "required": [],
    }
)

register_tool(
    name="fetch_recent_arxiv",
    description="Fetch recent arXiv papers with optional keyword filter.",
    func=fetch_recent_arxiv,
    schema=RecentArxivSchema,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional keyword to filter papers (leave empty for all recent papers)"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10,
                "minimum": 1,
                "maximum": 100
            },
            "category": {
                "type": "string",
                "description": "arXiv category (e.g., cs.AI, cs.ML)",
                "default": "cs.AI"
            }
        },
        "required": ["query"]
    }
)

def execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool call and validate parameters using Pydantic."""
    function_name = tool_call["function"]["name"]
    function_args = tool_call["function"]["arguments"]
    
    if isinstance(function_args, str):
        try:
            function_args = json.loads(function_args)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON in arguments"}
    
    # Convert any 'null' strings to None
    if isinstance(function_args, dict):
        function_args = {k: None if v == "null" else v for k, v in function_args.items()}
    
    if function_name not in TOOLS:
        logging.error(f"Tool not found: {function_name}")
        return {"status": "error", "message": f"Tool '{function_name}' not found."}
    
    try:
        schema = TOOLS[function_name]["schema"]
        validated_params = schema(**function_args)
        func = TOOLS[function_name]["func"]
        result = func(validated_params)
        
        logging.info(f"Executed {function_name} with args {function_args}, result: {result}")
        return {"status": "success", "result": result}
    
    except ValidationError as e:
        error_msg = f"Invalid parameters for '{function_name}': {str(e)}"
        logging.error(error_msg)
        return {"status": "error", "message": error_msg}
    except Exception as e:
        error_msg = f"Failed to execute '{function_name}': {str(e)}"
        logging.error(error_msg)
        return {"status": "error", "message": error_msg}

def process_tool_calls(tool_calls: List[Dict[str, Any]], messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process multiple tool calls and append results to messages.
    
    Args:
        tool_calls: List of tool calls from the model
        messages: Current message history
    Returns:
        Updated message history
    """
    for tool_call in tool_calls:
        print(f"\n- Function: {tool_call['function']['name']}")
        print(f"  Arguments: {json.dumps(tool_call['function']['arguments'], indent=2)}")
        
        result = execute_tool_call(tool_call)
        
        if result["status"] == "success":
            print(f"  Result: {result['result']}")
            messages.append({
                "role": "tool",
                "content": json.dumps({"tool": tool_call["function"]["name"], "result": result["result"]})
            })
        else:
            print(f"  Error: {result['message']}")
            messages.append({
                "role": "tool",
                "content": json.dumps({"tool": tool_call["function"]["name"], "error": result["message"]})
            })
    
    return messages

def format_arxiv_results(results: List[dict]) -> str:
    """Format arXiv results in a readable way."""
    if not results:
        return "An error occurred while searching arXiv."
        
    if "status" in results[0]:
        if results[0]["status"] == "no_results":
            return results[0]["message"]
        elif results[0]["status"] == "error":
            return f"Error: {results[0]['message']}"
    
    formatted = "Found the following recent arXiv papers:\n\n"
    for i, paper in enumerate(results, 1):
        formatted += f"{i}. Title: {paper['title']}\n"
        formatted += f"   Published: {paper['published']}\n"
        formatted += f"   Categories: {', '.join(paper['categories'])}\n"
        formatted += f"   PDF: {paper['pdf_link']}\n"
        formatted += f"   Web: {paper['html_link']}\n"
        formatted += f"   Summary: {paper['summary'][:200]}...\n\n"
    return formatted

def explain_result(messages: List[Dict[str, Any]]) -> str:
    """Ask the model to explain the tool results in natural language."""
    last_tool_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            last_tool_msg = msg
            break

    if last_tool_msg:
        try:
            content = json.loads(last_tool_msg["content"])
            tool_name = content.get("tool")
            result = content.get("result")

            # Handle arXiv results specially
            if tool_name == "fetch_recent_arxiv" and isinstance(result, list):
                formatted_results = format_arxiv_results(result)
                messages.append({"role": "assistant", "content": formatted_results})
                return formatted_results

        except Exception:
            pass

    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=messages + [
                {"role": "user", "content": "Please explain these research papers in simple terms."}
            ]
        )
        return response.get("message", {}).get("content", "No explanation provided.")
    except ollama.ResponseError as e:
        logging.error(f"Explanation error: {str(e)}")
        return f"Error generating explanation: {str(e)}"

def main():
    """Main function for interactive tool calling demo."""
    print("Advanced Ollama Tool Calling Demo (Type 'exit' to quit)")
    print(f"Available tools: {', '.join(TOOLS.keys())}")
    
    messages = []
    
    while True:
        user_input = input("\nYour query: ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = ollama.chat(
                model="llama3.2:3b",
                messages=messages,
                tools=[tool["tool"] for tool in TOOLS.values()],
            )
            
            tool_calls = response.get("message", {}).get("tool_calls", [])
            
            if tool_calls:
                print("\nTool Calls:")
                messages = process_tool_calls(tool_calls, messages)
                
                # Sonuçları doğal dilde açıkla
                explanation = explain_result(messages)
                print("\nExplanation:")
                print(explanation)
            else:
                content = response.get("message", {}).get("content", "No response from model.")
                print("\nModel Response:")
                print(content)
                messages.append({"role": "assistant", "content": content})
                
        except ollama.ResponseError as e:
            error_msg = f"Ollama API error: {str(e)}"
            logging.error(error_msg)
            print(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logging.error(error_msg)
            print(error_msg)

# Örnek prompt girdileri:
# 1. İki sayıyı topla
# Your query: Lütfen 12 ile 34'ü topla.

# 2. İki sayıyı çarp
# Your query: 7 ile 8'i çarpar mısın?

# 3. İki metni birleştir
# Your query: "Merhaba" ve "Dünya" kelimelerini birleştir.

# 4. Bir sayı listesinin toplamı
# Your query: 3, 5, 7 ve 11 sayılarını topla.

# 5. Metni ters çevir
# Your query: "Ollama harika" cümlesini ters çevir.

# 6. Şu anki tarihi ve saati ver
# Your query: Şu anki tarihi ve saati bana söyler misin?

# 7. Farklı formatta tarih/saat
# Your query: Tarihi sadece yıl-ay-gün olarak ver (format: %Y-%m-%d).

# 8. Son arxiv makaleleri
# Your query: Son arxiv'de "quantum computing" ile ilgili çıkan makaleleri getirir misin?

if __name__ == "__main__":
    main()