"""Rate limiting for the real (workspace) endpoints. Redis (Upstash) when REDIS_URL is
set - required so limits hold across multiple API instances, unlike in-memory state.
Falls back to an in-memory async limiter if REDIS_URL is unset, so the app still runs
without Redis configured - same degrade-gracefully approach as the rest of the codebase
(fastembed's hash fallback, the generator's extractive fallback).

ponytail: the Redis path is a fixed window (INCR + EXPIRE NX), not a true sliding window.
Upgrade path: a small Lua script if burst-at-window-boundary abuse becomes a real problem.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Optional


class InMemoryLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict = defaultdict(deque)

    async def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


class RedisLimiter:
    def __init__(self, redis_client, per_minute: int):
        self.redis = redis_client
        self.per_minute = per_minute

    async def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        redis_key = f"contexta:ratelimit:{key}"
        count = await self.redis.incr(redis_key)
        if count == 1:
            await self.redis.expire(redis_key, 60)
        return count <= self.per_minute


def make_limiter(redis_url: Optional[str], per_minute: int):
    if redis_url:
        import redis.asyncio as aioredis

        return RedisLimiter(aioredis.from_url(redis_url), per_minute)
    return InMemoryLimiter(per_minute)
