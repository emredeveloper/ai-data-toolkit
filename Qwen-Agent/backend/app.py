import re
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from qwen_agent.agents import Assistant
import logging


app = FastAPI()
class RequestData(BaseModel):
    content: str
bot = Assistant(
    llm={
        "model": "qwen3:4b",
        "model_server": "http://localhost:11434/v1",
        "api_key": "EMPTY"
    },
    function_list=["code_interpreter"],
    system_message="You are a summarization assistant. Directly output the cleaned summary of the given text without any reasoning, self-talk, thoughts, or internal planning steps. Do not include phrases like 'I think', 'maybe', 'let's', 'the user wants', or anything not part of the final summary. Your output must look like it was written by an editor, not a model."
)

@app.post("/summarize_stream_status")
async def summarize_stream_status(data: RequestData):
    user_input = data.content
    def stream():
        try:
            yield "🔍 Reading content on website...\n"
            print(" Received text:", user_input[:200])
            messages = [
            {"role": "system", "content": "You are a summarization assistant. Directly output the cleaned summary of the given text without any reasoning, self-talk, thoughts, or internal planning steps. Do not include phrases like 'I think', 'maybe', 'let's', 'the user wants', or anything not part of the final summary. Your output must look like it was written by an editor, not a model."},
            {"role": "user", "content": "<nothink>\nSummarize the following text clearly and concisely. Do not include any internal thoughts, planning, or reasoning. Just return the final summary:\n\n" + user_input + "\n</nothink>"}
            ]
            yield "🧠 Generating summary...\n"
            result = bot.run(messages)
            result_list = list(result)
            print(" Raw result:", result_list)
            # Extract summary
            last_content = None
            for item in reversed(result_list):
                if isinstance(item, list):
                    for subitem in reversed(item):
                        if isinstance(subitem, dict) and "content" in subitem:
                            last_content = subitem["content"]
                            break
                if last_content:
                    break
            if not last_content:
                yield " No valid summary found.\n"
                return
            summary = re.sub(r"</?think>", "", last_content)
            summary = re.sub(
                r"(?s)^.*?(Summary:|Here's a summary|The key points are|Your tutorial|This tutorial|To summarize|Final summary:)", 
                "", 
                summary, 
                flags=re.IGNORECASE
            )
            summary = re.sub(r"\n{3,}", "\n\n", summary)
            summary = summary.strip()
            yield "\n📄 Summary:\n" + summary + "\n"
        except Exception as e:
            print(" Error:", e)
            yield f"\n Error: {str(e)}\n"
    return StreamingResponse(stream(), media_type="text/plain")