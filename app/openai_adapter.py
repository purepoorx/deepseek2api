import json
import time

async def convert_to_openai_stream(deepseek_stream, model_id: str):
    completion_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())
    
    is_thinking = False
    
    async for chunk in deepseek_stream:
        chunk_type = chunk.get("type")
        content = chunk.get("content")
        
        if not chunk_type or not content:
            continue

        delta = {"role": "assistant", "content": None, "reasoning_content": None}
        
        if chunk_type == "thinking_start":
            is_thinking = True
            delta["reasoning_content"] = "[Thinking] "
        elif chunk_type == "answer_start":
            is_thinking = False
            delta["content"] = "" # Start of answer
        elif chunk_type == "thinking":
            delta["reasoning_content"] = content
        elif chunk_type == "content":
            delta["content"] = content
        
        openai_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_id,
            "choices": [{"delta": delta, "index": 0, "finish_reason": None}]
        }
        yield "data: " + json.dumps(openai_chunk) + "\n\n"

    # Send final chunk
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_id,
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
    }
    yield "data: " + json.dumps(final_chunk) + "\n\n"
    yield "data: [DONE]\n\n"
