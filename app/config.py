import json
from pydantic import BaseModel, Field
from typing import List

class Account(BaseModel):
    email: str = None
    mobile: str = None
    password: str
    token: str = None

class Settings(BaseModel):
    accounts: List[Account] = Field(default_factory=list)
    api_keys: List[str] = Field(default_factory=list)

def load_config(path: str = "config.json") -> Settings:
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return Settings(**data)
    except FileNotFoundError:
        print(f"Warning: {path} not found. Using default settings.")
        return Settings()
    except Exception as e:
        print(f"Error loading config from {path}: {e}")
        return Settings()

config = load_config()
