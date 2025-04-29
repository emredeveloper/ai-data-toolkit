import argparse
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
import asyncio
import sys
import os

def create_mcp_server(mode: str, db_path: str):
    if mode not in ['stdio', 'sse', 'warmup']:
        raise ValueError(f"Invalid mode: {mode}. Supported modes are 'stdio', 'sse', or 'warmup'.")
    
    if not os.path.isabs(db_path):
        raise ValueError(f"Database path must be absolute: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Warning: Database file {db_path} does not exist. It will be created by the SQLite MCP server.")

    # Use npx to run @modelcontextprotocol/server-sqlite
    args = ['-y', '@modelcontextprotocol/server-sqlite', db_path]
    if mode != 'stdio':
        args.append(mode)  # Append mode for sse or warmup (if supported)
        print(f"Warning: Mode '{mode}' may not be fully supported by @modelcontextprotocol/server-sqlite. Using as argument.")

    return MCPServerStdio(
        'npx',
        args=args
    )

async def main(mode: str, db_path: str):
    try:
        server = create_mcp_server(mode, db_path)
        agent = Agent('ollama:llama3', mcp_servers=[server])
        
        async with agent.run_mcp_servers():
            result = await agent.run('2000-01-01 ile 2025-03-18 arasındaki gün sayısı kaçtır?')
            print(result.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Pydantic AI agent with SQLite MCP server in specified mode.')
    parser.add_argument('--mode', type=str, default='stdio', choices=['stdio', 'sse', 'warmup'],
                        help='MCP server mode (stdio, sse, or warmup)')
    parser.add_argument('--db-path', type=str, required=True,
                        help='Absolute path to the SQLite database file (e.g., C:\\Users\\emreq\\Desktop\\test.db)')
    args = parser.parse_args()
    
    asyncio.run(main(args.mode, args.db_path))