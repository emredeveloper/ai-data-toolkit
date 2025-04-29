from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
import asyncio
from datetime import datetime
import signal
import sys

# Function to handle proper cleanup on exit
def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

mcp_fetch = MCPServerStdio(
    command="uvx",
    args=["mcp-server-fetch"],
)


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Configure Ollama through OpenAI compatibility API
ollama_model = OpenAIModel(
    model_name='granite3.3:8b', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

agent = Agent(
    ollama_model,
    system_prompt=(
        "You are a helpful assistant. Use tools to achieve the user's goal."),
    mcp_servers=[mcp_fetch],
    tools=[Tool(get_time)])


async def main():
    try:
        async with agent.run_mcp_servers():
            prompt = """
                Please get the content of docs.replit.com/updates and summarize them. 
                Return the summary as well as the time you got the content.
                """
            result = await agent.run(prompt)
            print(result.output)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Ensure proper cleanup
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
