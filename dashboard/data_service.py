import json
import os
from types import SimpleNamespace
from urllib.request import Request, urlopen
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from solders.pubkey import Pubkey

try:
    # solana-py versions before 0.40 exposed this synchronous client.
    from solana.rpc.api import Client as _SolanaClient
except ModuleNotFoundError:
    _SolanaClient = None


class _JsonRpcClient:
    """Small synchronous fallback for solana-py versions without rpc.api."""

    def __init__(self, endpoint):
        self.endpoint = endpoint

    def get_balance(self, pubkey):
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBalance",
            "params": [str(pubkey)],
        }).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return SimpleNamespace(value=result["result"]["value"])

load_dotenv()

class DashboardService:
    def __init__(self):
        db_user = os.getenv("DB_USER", "postgres")
        db_pass = os.getenv("DB_PASSWORD", "password")
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "crypto_quant")
        self.engine = sqlalchemy.create_engine(f"postgresql://{db_user}:{db_pass}@{db_host}:5432/{db_name}")
        rpc_url = os.getenv("QUICKNODE_RPC_URL", "https://api.mainnet-beta.solana.com")
        self.rpc = _SolanaClient(rpc_url) if _SolanaClient else _JsonRpcClient(rpc_url)
        self.wallet_addr = self._get_wallet_address()

    def _get_wallet_address(self):
        try:
            from solders.keypair import Keypair
            pk_str = os.getenv("SOLANA_PRIVATE_KEY", "")
            if "[" in pk_str:
                kp = Keypair.from_bytes(json.loads(pk_str))
            else:
                kp = Keypair.from_base58_string(pk_str)
            return str(kp.pubkey())
        except Exception:
            return "Unknown"

    def get_wallet_balance(self):
        try:
            resp = self.rpc.get_balance(Pubkey.from_string(self.wallet_addr))
            return resp.value / 1e9
        except Exception as e:
            return 0.0

    def load_portfolio(self):
        try:
            with open("portfolio_state.json", "r") as f:
                data = json.load(f)
                if not data: return pd.DataFrame()
                
                df = pd.DataFrame(data.values())
                # 计算当前预估 PnL
                if 'highest_price' in df.columns and 'entry_price' in df.columns:
                    df['pnl_pct'] = (df['highest_price'] - df['entry_price']) / df['entry_price']
                return df
        except FileNotFoundError:
            return pd.DataFrame()

    def load_strategy_info(self):
        try:
            with open("best_meme_strategy.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"formula": "Not Trained Yet"}

    def get_market_overview(self, limit=50):
        query = f"""
        SELECT t.symbol, o.address, o.close, o.volume, o.liquidity, o.fdv, o.time
        FROM ohlcv o
        JOIN tokens t ON o.address = t.address
        WHERE o.time = (
            SELECT MAX(latest.time)
            FROM ohlcv AS latest
            WHERE latest.address = o.address
        )
        ORDER BY o.liquidity DESC
        LIMIT {limit}
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception:
            return pd.DataFrame()
    
    def get_recent_logs(self, n=50):
        log_file = "strategy.log"
        if not os.path.exists(log_file): return []
        
        with open(log_file, "r") as f:
            lines = f.readlines()
            return lines[-n:]
