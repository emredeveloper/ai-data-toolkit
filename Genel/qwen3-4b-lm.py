"""
LM Studio Tool Use Demo: Wikipedia Querying Chatbot
Demonstrates how an LM Studio model can query Wikipedia
"""

# Standard library imports
import itertools
import json
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request

# Third-party imports
from openai import OpenAI

# Initialize LM Studio client
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL = "qwen3-4b"


def fetch_wikipedia_content(search_query: str) -> dict:
    """Fetches wikipedia content for a given search_query"""
    try:
        # Search for most relevant article
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": search_query,
            "srlimit": 1,
        }

        url = f"{search_url}?{urllib.parse.urlencode(search_params)}"
        with urllib.request.urlopen(url) as response:
            search_data = json.loads(response.read().decode())

        if not search_data["query"]["search"]:
            return {
                "status": "error",
                "message": f"No Wikipedia article found for '{search_query}'",
            }

        # Get the normalized title from search results
        normalized_title = search_data["query"]["search"][0]["title"]

        # Now fetch the actual content with the normalized title
        content_params = {
            "action": "query",
            "format": "json",
            "titles": normalized_title,
            "prop": "extracts",
            "exintro": "true",
            "explaintext": "true",
            "redirects": 1,
        }

        url = f"{search_url}?{urllib.parse.urlencode(content_params)}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())

        pages = data["query"]["pages"]
        page_id = list(pages.keys())[0]

        if page_id == "-1":
            return {
                "status": "error",
                "message": f"No Wikipedia article found for '{search_query}'",
            }

        content = pages[page_id]["extract"].strip()
        return {
            "status": "success",
            "content": content,
            "title": pages[page_id]["title"],
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# Define tool for LM Studio
WIKI_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_wikipedia_content",
        "description": (
            "Search Wikipedia and fetch the introduction of the most relevant article. "
            "Always use this if the user is asking for something that is likely on wikipedia. "
            "If the user has a typo in their search query, correct it before searching."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Search query for finding the Wikipedia article",
                },
            },
            "required": ["search_query"],
        },
    },
}


# Class for displaying the state of model processing
class Spinner:
    def __init__(self, message="Processing..."):
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
        self.busy = False
        self.delay = 0.1
        self.message = message
        self.thread = None

    def write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def _spin(self):
        while self.busy:
            self.write(f"\r{self.message} {next(self.spinner)}")
            time.sleep(self.delay)
        self.write("\r\033[K")  # Clear the line

    def __enter__(self):
        self.busy = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.busy = False
        time.sleep(self.delay)
        if self.thread:
            self.thread.join()
        self.write("\r")  # Move cursor to beginning of line


def chat_loop():
    """
    Main chat loop that processes user input and handles tool calls.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that can retrieve Wikipedia articles. "
                "When asked about a topic, you can retrieve Wikipedia articles "
                "and cite information from them."
            ),
        }
    ]

    print(
        "Assistant: "
        "Hi! I can access Wikipedia to help answer your questions about history, "
        "science, people, places, or concepts - or we can just chat about "
        "anything else!"
    )
    print("(Type 'quit' to exit)")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})
        try:
            with Spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=[WIKI_TOOL],
                )

            if response.choices[0].message.tool_calls:
                # Handle all tool calls
                tool_calls = response.choices[0].message.tool_calls

                # Add all tool calls to messages
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": tool_call.function,
                            }
                            for tool_call in tool_calls
                        ],
                    }
                )

                # Process each tool call and add results
                for tool_call in tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    result = fetch_wikipedia_content(args["search_query"])

                    # Print the Wikipedia content in a formatted way
                    terminal_width = shutil.get_terminal_size().columns
                    print("\n" + "=" * terminal_width)
                    if result["status"] == "success":
                        print(f"\nWikipedia article: {result['title']}")
                        print("-" * terminal_width)
                        print(result["content"])
                    else:
                        print(
                            f"\nError fetching Wikipedia content: {result['message']}"
                        )
                    print("=" * terminal_width + "\n")

                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(result),
                            "tool_call_id": tool_call.id,
                        }
                    )

                # Stream the post-tool-call response
                print("\nAssistant:", end=" ", flush=True)
                stream_response = client.chat.completions.create(
                    model=MODEL, messages=messages, stream=True
                )
                collected_content = ""
                for chunk in stream_response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        collected_content += content
                print()  # New line after streaming completes
                messages.append(
                    {
                        "role": "assistant",
                        "content": collected_content,
                    }
                )
            else:
                # Handle regular response
                print("\nAssistant:", response.choices[0].message.content)
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                )

        except Exception as e:
            print(
                f"\nError chatting with the LM Studio server!\n\n"
                f"Please ensure:\n"
                f"1. LM Studio server is running at 127.0.0.1:1234 (hostname:port)\n"
                f"2. Model '{MODEL}' is downloaded\n"
                f"3. Model '{MODEL}' is loaded, or that just-in-time model loading is enabled\n\n"
                f"Error details: {str(e)}\n"
                "See https://lmstudio.ai/docs/basics/server for more information"
            )
            exit(1)


if __name__ == "__main__":
    chat_loop()