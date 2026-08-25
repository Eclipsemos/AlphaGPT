import asyncio
import aiohttp
from datetime import datetime
import json
from loguru import logger
from .config import Config
from .db_manager import DBManager
from .providers.birdeye import BirdeyeProvider
from .providers.dexscreener import DexScreenerProvider

class DataManager:
    def __init__(self):
        self.db = DBManager()
        self.birdeye = BirdeyeProvider()
        self.dexscreener = DexScreenerProvider()
        
    async def initialize(self):
        await self.db.connect()
        await self.db.init_schema()

    async def close(self):
        await self.db.close()

    def _write_status(self, **values):
        status = {
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "birdeye_requests": self.birdeye.request_count,
            "birdeye_rate_limits": self.birdeye.rate_limit_count,
            "birdeye_last_status": self.birdeye.last_status,
            **values,
        }
        with open("data_pipeline_status.json", "w") as handle:
            json.dump(status, handle, indent=2)

    async def pipeline_sync_daily(self):
        logger.info("Step 1: Discovering trending tokens...")
        # Birdeye's trending endpoint currently accepts at most 50 tokens.
        limit = min(50, 500 if Config.BIRDEYE_IS_PAID else 100)
        candidates = await self.birdeye.get_trending_tokens(limit=limit)
        
        logger.info(f"Raw candidates found: {len(candidates)}")

        selected_tokens = []
        for t in candidates:
            liq = t.get('liquidity', 0)
            fdv = t.get('fdv', 0)
            
            if liq < Config.MIN_LIQUIDITY_USD: continue
            if fdv < Config.MIN_FDV: continue
            if fdv > Config.MAX_FDV: continue # 剔除像 WIF/BONK 这种巨无霸，专注于早期高成长
            
            selected_tokens.append(t)
            
        logger.info(f"Tokens selected after filtering: {len(selected_tokens)}")
        
        if not selected_tokens:
            self._write_status(candidate_count=len(candidates), selected_count=0, candle_count=0)
            logger.warning("No tokens passed the filter. Relax constraints in Config.")
            return

        db_tokens = [(t['address'], t['symbol'], t['name'], t['decimals'], Config.CHAIN) for t in selected_tokens]
        await self.db.upsert_tokens(db_tokens)
        snapshot_time = datetime.utcnow().replace(second=0, microsecond=0)
        await self.db.insert_token_snapshot(snapshot_time, selected_tokens)

        logger.info(f"Step 4: Fetching OHLCV for {len(selected_tokens)} tokens...")
        
        async with aiohttp.ClientSession(headers=self.birdeye.headers, trust_env=True) as session:
            tasks = []
            for t in selected_tokens:
                tasks.append(self.birdeye.get_token_history(
                    session,
                    t['address'],
                    liquidity=t.get('liquidity'),
                    fdv=t.get('fdv'),
                ))
            
            batch_size = 20
            total_candles = 0
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                results = await asyncio.gather(*batch)
                
                records = [item for sublist in results if sublist for item in sublist]
                
                # 批量写入
                await self.db.batch_insert_ohlcv(records)
                total_candles += len(records)
                logger.info(f"Processed batch {i}/{len(tasks)}. Inserted {len(records)} candles.")
                
        logger.success(f"Pipeline complete. Total candles stored: {total_candles}")
        self._write_status(candidate_count=len(candidates), selected_count=len(selected_tokens), candle_count=total_candles)
