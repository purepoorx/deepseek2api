import ctypes
import struct
import os
import json
import base64
from wasmtime import Store, Module, Linker
import httpx
from contextlib import asynccontextmanager

# --- API Configuration ---
DEEPSEEK_API_BASE = "https://chat.deepseek.com/api/v0"
WASM_FILE_PATH = "sha3_wasm_bg.7b9ca65ddd.wasm"
BASE_HEADERS = {
    "Host": "chat.deepseek.com",
    "User-Agent": "DeepSeek/1.0.13 Android/35",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "x-client-platform": "web",
    "x-client-version": "1.2.0-sse-hint",
}

class PoWGenerator:
    # This class remains synchronous as it's CPU-bound
    def __init__(self, wasm_path: str):
        if not os.path.exists(wasm_path):
            raise FileNotFoundError(f"WASM file not found at: {wasm_path}")
        self.store = Store()
        self.linker = Linker(self.store.engine)
        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()
        module = Module(self.store.engine, wasm_bytes)
        self.instance = self.linker.instantiate(self.store, module)
        exports = self.instance.exports(self.store)
        try:
            self.memory = exports["memory"]
            self.add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
            self.alloc = exports["__wbindgen_export_0"]
            self.wasm_solve = exports["wasm_solve"]
        except KeyError as e:
            raise RuntimeError(f"Could not find required export in WASM module: {e}")

    def _write_memory(self, offset: int, data: bytes):
        base_addr = ctypes.cast(self.memory.data_ptr(self.store), ctypes.c_void_p).value
        ctypes.memmove(base_addr + offset, data, len(data))

    def _read_memory(self, offset: int, size: int) -> bytes:
        base_addr = ctypes.cast(self.memory.data_ptr(self.store), ctypes.c_void_p).value
        return ctypes.string_at(base_addr + offset, size)

    def _encode_string(self, text: str) -> tuple[int, int]:
        data = text.encode("utf-8")
        length = len(data)
        ptr_val = self.alloc(self.store, length, 1)
        ptr = int(ptr_val.value) if hasattr(ptr_val, "value") else int(ptr_val)
        self._write_memory(ptr, data)
        return ptr, length

    def calculate_answer(self, challenge_data: dict) -> int | None:
        challenge_str = challenge_data["challenge"]
        difficulty = challenge_data["difficulty"]
        salt = challenge_data["salt"]
        expire_at = challenge_data["expire_at"]
        
        prefix = f"{salt}_{expire_at}_"
        retptr = self.add_to_stack(self.store, -16)
        try:
            ptr_challenge, len_challenge = self._encode_string(challenge_str)
            ptr_prefix, len_prefix = self._encode_string(prefix)
            self.wasm_solve(
                self.store, retptr, ptr_challenge, len_challenge,
                ptr_prefix, len_prefix, float(difficulty)
            )
            status_bytes = self._read_memory(retptr, 4)
            if len(status_bytes) != 4: return None
            status = struct.unpack("<i", status_bytes)[0]
            if status == 0: return None
            value_bytes = self._read_memory(retptr + 8, 8)
            if len(value_bytes) != 8: return None
            value = struct.unpack("<d", value_bytes)[0]
            return int(value)
        finally:
            self.add_to_stack(self.store, 16)

class DeepSeekClient:
    def __init__(self, token: str, session: httpx.AsyncClient):
        if not token:
            raise ValueError("An authorization token is required.")
        self.token = token
        self.headers = {**BASE_HEADERS, "authorization": f"Bearer {self.token}"}
        self.pow_generator = PoWGenerator(WASM_FILE_PATH)
        self.session = session
        print("DeepSeekClient initialized.")

    async def _create_session(self) -> str | None:
        print("Creating new chat session...")
        try:
            resp = await self.session.post(
                f"{DEEPSEEK_API_BASE}/chat_session/create",
                headers=self.headers,
                json={"character_id": None},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == 0:
                session_id = data["data"]["biz_data"]["id"]
                print(f"Session created: {session_id}")
                return session_id
            else:
                print(f"Failed to create session: {data.get('msg')}")
                return None
        except Exception as e:
            print(f"Error creating session: {e}")
            return None

    async def _get_and_solve_pow(self) -> str | None:
        print("Getting and solving PoW challenge...")
        try:
            resp = await self.session.post(
                f"{DEEPSEEK_API_BASE}/chat/create_pow_challenge",
                headers=self.headers,
                json={"target_path": "/api/v0/chat/completion"},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                print(f"Failed to get PoW challenge: {data.get('msg')}")
                return None
            
            challenge_data = data["data"]["biz_data"]["challenge"]
            answer = self.pow_generator.calculate_answer(challenge_data)
            if answer is None:
                print("Failed to solve PoW challenge.")
                return None
            
            print(f"PoW challenge solved. Answer: {answer}")

            pow_dict = {
                "algorithm": challenge_data["algorithm"],
                "challenge": challenge_data["challenge"],
                "salt": challenge_data["salt"],
                "answer": answer,
                "signature": challenge_data["signature"],
                "target_path": challenge_data["target_path"],
            }
            pow_str = json.dumps(pow_dict, separators=(",", ":"))
            encoded_pow = base64.b64encode(pow_str.encode("utf-8")).decode("utf-8")
            return encoded_pow

        except Exception as e:
            print(f"Error during PoW process: {e}")
            return None

    async def _delete_session(self, session_id: str):
        print(f"\nDeleting session: {session_id}")
        try:
            await self.session.post(
                f"{DEEPSEEK_API_BASE}/chat_session/delete",
                headers=self.headers,
                json={"chat_session_id": session_id},
            )
            print("Session deleted successfully.")
        except Exception as e:
            print(f"Error deleting session: {e}")

    @asynccontextmanager
    async def managed_chat_session(self):
        session_id = await self._create_session()
        if not session_id:
            # If session creation fails, yield None and let the caller handle it
            yield None
            return
        try:
            yield session_id
        finally:
            # This ensures deletion happens even if the stream breaks
            await self._delete_session(session_id)

    async def chat_stream(self, session_id: str, prompt: str, thinking_enabled: bool = True):
        pow_response = await self._get_and_solve_pow()
        if not pow_response:
            return
            
        chat_headers = {**self.headers, "x-ds-pow-response": pow_response}
        payload = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": False,
        }

        print(f"\n--- Sending prompt (Thinking: {thinking_enabled}) ---")
        
        try:
            async with self.session.stream(
                "POST",
                f"{DEEPSEEK_API_BASE}/chat/completion",
                headers=chat_headers,
                json=payload,
                timeout=None
            ) as resp:
                resp.raise_for_status()
                
                is_thinking = False
                buffer = ""
                async for content_chunk in resp.aiter_bytes():
                    buffer += content_chunk.decode('utf-8', errors='ignore')
                    while '\n\n' in buffer:
                        line, buffer = buffer.split('\n\n', 1)
                        if not line.startswith("data:"):
                            continue
                            
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            return
                            
                        try:
                            chunk = json.loads(data_str)
                            path = chunk.get("p")
                            value = chunk.get("v")

                            if path == "response/thinking_content":
                                if not is_thinking:
                                    is_thinking = True
                                    yield {"type": "thinking_start"}
                                yield {"type": "thinking", "content": value}
                            elif path == "response/content":
                                if is_thinking:
                                    is_thinking = False
                                    yield {"type": "answer_start"}
                                yield {"type": "content", "content": value}
                            elif value is not None and isinstance(value, str) and "p" not in chunk:
                                if is_thinking:
                                    yield {"type": "thinking", "content": value}
                                else:
                                    yield {"type": "content", "content": value}

                        except (json.JSONDecodeError, AttributeError):
                            continue
        except Exception as e:
            print(f"\nError during chat completion: {e}")
            yield {"type": "error", "content": str(e)}
