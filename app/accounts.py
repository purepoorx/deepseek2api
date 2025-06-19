import asyncio
import httpx
from contextlib import asynccontextmanager
from .config import config

class AccountManager:
    def __init__(self):
        self.account_queue = asyncio.Queue()
        self.accounts = config.accounts
        print(f"AccountManager loaded with {len(self.accounts)} accounts from config.")

    async def _login_and_get_token(self, session: httpx.AsyncClient, account):
        print(f"Attempting to log in for {account.email or account.mobile}...")
        payload = {
            "password": account.password,
            "device_id": "deepseek_to_api_accounts",
            "os": "android",
        }
        if account.email:
            payload["email"] = account.email
        else:
            payload["mobile"] = account.mobile
        
        try:
            resp = await session.post(
                "https://chat.deepseek.com/api/v0/users/login",
                json=payload,
                # httpx does not support impersonate, but we can set headers
                headers={"User-Agent": "DeepSeek/1.0.13 Android/35"}
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == 0:
                token = data["data"]["biz_data"]["user"]["token"]
                print(f"Token acquired for {account.email or account.mobile}.")
                return token
            print(f"Login failed for {account.email or account.mobile}: {data.get('msg')}")
            return None
        except Exception as e:
            print(f"Login request failed for {account.email or account.mobile}: {e}")
            return None

    async def initialize_tokens(self, session: httpx.AsyncClient):
        tasks = []
        for acc in self.accounts:
            if not acc.token:
                tasks.append(self._login_and_get_token(session, acc))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=acc.token)))

        results = await asyncio.gather(*tasks)
        
        for i, token in enumerate(results):
            if token:
                self.accounts[i].token = token
                await self.account_queue.put(self.accounts[i])
        
        print(f"AccountManager initialized with {self.account_queue.qsize()} active accounts.")

    async def get_account(self):
        print("Waiting for an available account...")
        account = await self.account_queue.get()
        print("Account acquired.")
        return account

    def release_account(self, account):
        self.account_queue.put_nowait(account)
        print("Account released.")

    @asynccontextmanager
    async def managed_account(self):
        account = await self.get_account()
        try:
            yield account
        finally:
            self.release_account(account)

account_manager = AccountManager()
