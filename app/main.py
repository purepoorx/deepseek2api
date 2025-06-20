from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import time

from .config import config
from .accounts import account_manager
from .deepseek import DeepSeekClient
from .openai_adapter import convert_to_openai_stream

app = FastAPI()

# CORS Middleware Configuration
origins = ["*"]  # In production, you should restrict this to specific domains.

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def chat_completions(request: Request):
    try:
        request_data = await request.json()
    except Exception:
        print("Client disconnected before request body was received.")
        return JSONResponse(status_code=400, content={"error": "Client disconnected."})

    model = request_data.get("model", "deepseek-chat")
    
    messages = request_data.get("messages", [])
    prompt_parts = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role and content:
            prompt_parts.append(f"{role}: {content}")
    prompt = "\n".join(prompt_parts)

    thinking_enabled = "thinking" in model if "thinking_enabled" not in request_data else request_data["thinking_enabled"]

    if request_data.get("stream", False):
        async def stream_logic():
            async with account_manager.managed_account() as account:
                client = DeepSeekClient(token=account.token, session=http_client)
                async with client.managed_chat_session() as session_id:
                    if not session_id:
                        return
                    raw_stream = client.chat_stream(session_id, prompt, thinking_enabled)
                    openai_stream = convert_to_openai_stream(raw_stream, model)
                    async for chunk in openai_stream:
                        yield chunk
        return StreamingResponse(stream_logic(), media_type="text/event-stream")
    else:
        full_content = ""
        async with account_manager.managed_account() as account:
            client = DeepSeekClient(token=account.token, session=http_client)
            async with client.managed_chat_session() as session_id:
                if not session_id:
                    raise HTTPException(status_code=500, detail="Failed to create chat session.")
                
                async for chunk in client.chat_stream(session_id, prompt, thinking_enabled):
                    if chunk.get("type") == "content":
                        full_content += chunk.get("content", "")
        
        response_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": full_content}, "index": 0, "finish_reason": "stop"}]
        }
        return JSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    print("This file is not meant to be run directly. Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000")