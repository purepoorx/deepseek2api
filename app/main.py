from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import time

from .config import config
from .accounts import account_manager
from .deepseek import DeepSeekClient
from .openai_adapter import convert_to_openai_stream

app = FastAPI()
http_client = httpx.AsyncClient(verify=False)

@app.on_event("startup")
async def startup_event():
    await account_manager.initialize_tokens(http_client)

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

# --- API Key Authentication ---
async def verify_api_key(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    
    token = auth_header.split(" ")[1]
    if not config.api_keys:
        print("Warning: No api_keys configured. Allowing all requests.")
        return
    if token not in config.api_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key.")

# --- OpenAI Compatible Endpoints ---
@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-thinking", "object": "model", "owned_by": "deepseek"}
        ]
    }

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def stream_generator(request_data: dict):
    """A unified generator to handle the entire streaming process with proper resource management."""
    model = request_data.get("model", "deepseek-chat")
    prompt = request_data.get("messages", [{}])[-1].get("content")
    
    if "thinking_enabled" in request_data:
        thinking_enabled = request_data["thinking_enabled"]
    else:
        thinking_enabled = "thinking" in model

    async with account_manager.managed_account() as account:
        client = DeepSeekClient(token=account.token, session=http_client)
        async with client.managed_chat_session() as session_id:
            if not session_id:
                # In streaming mode, we yield an error. Here we adapt for non-stream.
                # This part is tricky for non-stream, let's assume session_id is always created for now.
                # A more robust solution would handle this failure case.
                yield {"type": "error", "content": "Failed to create chat session."}
                return

            stream = client.chat_stream(session_id, prompt, thinking_enabled)
            async for chunk in stream: # We'll convert to openai format outside
                yield chunk

async def get_full_response(request_data: dict):
    """Handles the non-streaming case, aggregating content."""
    full_content = ""
    async for chunk in stream_generator(request_data):
        if chunk.get("type") == "content":
            full_content += chunk.get("content", "")
        elif chunk.get("type") == "error":
            # If the generator yields an error, we can propagate it
            raise HTTPException(status_code=500, detail=chunk.get("content"))
    return full_content


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request):
    try:
        request_data = await request.json()
    except Exception:
        print("Client disconnected before request body was received.")
        return JSONResponse(status_code=400, content={"error": "Client disconnected."})

    stream = request_data.get("stream", False)
    model = request_data.get("model", "deepseek-chat")

    if stream:
        # The generator now yields raw chunks, convert them for the stream
        openai_stream = convert_to_openai_stream(stream_generator(request_data), model)
        return StreamingResponse(openai_stream, media_type="text/event-stream")
    else:
        # Handle non-streaming case
        full_content = await get_full_response(request_data)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": full_content}, "index": 0, "finish_reason": "stop"}]
        }

if __name__ == "__main__":
    import uvicorn
    print("This file is not meant to be run directly. Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000")