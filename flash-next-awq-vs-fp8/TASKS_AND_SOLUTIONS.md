# Coding benchmark: tasks and reference solutions

[Back to article](README.md) · [Русская версия](TASKS_AND_SOLUTIONS_RU.md)

These are the eight tasks used to compare Qwen3.8 Flash-Next AWQ W4A16 and FP8. The solutions below are compact reference implementations for checking the requirements; they are not verbatim copies of either model's answer.

---

## Task 1 — Python session grouping / correctness

Group user events into sessions. A new session begins when the gap between adjacent events of the same user is **strictly greater than 30 minutes**. Input may be unsorted and must not be modified. Output must be ordered by `user_id` and then session start time. A gap of exactly 30 minutes stays in one session. Complexity must be no worse than `O(n log n)`.

```python
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Event:
    user_id: int
    ts: datetime
    name: str


def build_sessions(events: list[Event]) -> list[list[Event]]:
    if not events:
        return []

    ordered = sorted(events, key=lambda e: (e.user_id, e.ts))
    sessions: list[list[Event]] = []
    current: list[Event] = [ordered[0]]

    for event in ordered[1:]:
        prev = current[-1]
        if (
            event.user_id == prev.user_id
            and event.ts - prev.ts <= timedelta(minutes=30)
        ):
            current.append(event)
        else:
            sessions.append(current)
            current = [event]

    sessions.append(current)
    return sessions
```

The original version assumed pre-sorted input and incorrectly split on `>= 30 minutes` instead of `> 30 minutes`.

---

## Task 2 — Async cache / shared in-flight computation

For one key, concurrent callers must share one expensive `factory()` computation. Different keys must run in parallel. Failures must allow retry. Cancelling one waiter must not cancel the shared computation.

```python
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class AsyncCache:
    def __init__(self) -> None:
        self._cache: dict[Any, Any] = {}
        self._inflight: dict[Any, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key, factory: Callable[[], Awaitable[Any]]):
        async with self._lock:
            if key in self._cache:
                return self._cache[key]

            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(self._compute(key, factory))
                self._inflight[key] = task

        return await asyncio.shield(task)

    async def _compute(self, key, factory):
        this_task = asyncio.current_task()
        try:
            value = await factory()
            async with self._lock:
                self._cache[key] = value
            return value
        finally:
            async with self._lock:
                if self._inflight.get(key) is this_task:
                    self._inflight.pop(key, None)
```

The global lock protects only the dictionaries; it is not held while `factory()` runs.

---

## Task 3 — Smallest covering range

Given up to 1000 sorted lists and up to 1,000,000 values total, find the shortest inclusive range `[L, R]` containing at least one value from every list. Break ties by the smaller `L`. Do not flatten all values into one giant sorted list.

```python
import heapq


def smallest_covering_range(
    lists: list[list[int]],
) -> tuple[int, int] | None:
    if not lists or any(not xs for xs in lists):
        return None

    heap: list[tuple[int, int, int]] = []
    current_max = -10**30

    for list_idx, xs in enumerate(lists):
        value = xs[0]
        heap.append((value, list_idx, 0))
        current_max = max(current_max, value)

    heapq.heapify(heap)
    best_l = heap[0][0]
    best_r = current_max

    while True:
        current_min, list_idx, elem_idx = heapq.heappop(heap)

        if (
            current_max - current_min < best_r - best_l
            or (
                current_max - current_min == best_r - best_l
                and current_min < best_l
            )
        ):
            best_l, best_r = current_min, current_max

        next_idx = elem_idx + 1
        if next_idx == len(lists[list_idx]):
            break

        next_value = lists[list_idx][next_idx]
        current_max = max(current_max, next_value)
        heapq.heappush(heap, (next_value, list_idx, next_idx))

    return best_l, best_r
```

Time complexity is `O(N log K)` and extra memory is `O(K)`.

---

## Task 4 — PostgreSQL aggregation and latest row

Return every customer, count/sum of `COMPLETED` orders from the last 30 days, and the latest matching order by `created_at DESC, id DESC`.

```sql
WITH recent_completed AS (
    SELECT
        o.*,
        row_number() OVER (
            PARTITION BY o.customer_id
            ORDER BY o.created_at DESC, o.id DESC
        ) AS rn
    FROM orders AS o
    WHERE o.status = 'COMPLETED'
      AND o.created_at >= now() - interval '30 days'
),
agg AS (
    SELECT
        customer_id,
        count(*) AS completed_count,
        sum(total) AS completed_total,
        max(id) FILTER (WHERE rn = 1) AS latest_order_id,
        max(created_at) FILTER (WHERE rn = 1) AS latest_order_created_at
    FROM recent_completed
    GROUP BY customer_id
)
SELECT
    c.id AS customer_id,
    c.name,
    coalesce(a.completed_count, 0) AS completed_count,
    coalesce(a.completed_total, 0) AS completed_total,
    a.latest_order_id,
    a.latest_order_created_at
FROM customers AS c
LEFT JOIN agg AS a
    ON a.customer_id = c.id
ORDER BY c.id;
```

Useful index:

```sql
CREATE INDEX orders_completed_customer_created_id_idx
ON orders (customer_id, created_at DESC, id DESC)
INCLUDE (total)
WHERE status = 'COMPLETED';
```

---

## Task 5 — Strict TypeScript `parsePort`

Accept only valid integer ports and strings consisting solely of a decimal integer plus optional surrounding whitespace. Reject fractional values, booleans, arrays, objects, junk suffixes, infinities and out-of-range values. Validate fallback as well.

```typescript
export function parsePort(value: unknown, fallback = 3000): number {
    if (
        typeof fallback !== "number" ||
        !Number.isFinite(fallback) ||
        !Number.isInteger(fallback) ||
        fallback < 1 ||
        fallback > 65535
    ) {
        throw new RangeError("fallback must be an integer in range 1..65535");
    }

    if (value === undefined || value === null) {
        return fallback;
    }

    let port: number;

    if (typeof value === "number") {
        if (!Number.isFinite(value) || !Number.isInteger(value)) {
            return fallback;
        }
        port = value;
    } else if (typeof value === "string") {
        if (value.trim() === "") {
            return fallback;
        }
        if (!/^\s*\d+\s*$/.test(value)) {
            return fallback;
        }
        port = Number(value.trim());
    } else {
        return fallback;
    }

    return port >= 1 && port <= 65535 ? port : fallback;
}
```

---

## Task 6 — Multi-file storage semantics

Email lookup is case-insensitive while the original email string must be preserved. Storage must not expose mutable references to its internal objects.

### `storage.py`

```python
USERS = {}


def _key(email: str) -> str:
    return email.lower()


def save_user(user):
    USERS[_key(user["email"])] = dict(user)


def get_user(email):
    user = USERS.get(_key(email))
    return None if user is None else dict(user)
```

### `service.py`

```python
from storage import get_user, save_user


def register(email, name):
    if get_user(email) is not None:
        raise ValueError("already exists")

    user = {
        "email": email,
        "name": name,
    }

    save_user(user)
    return dict(user)


def login(email):
    user = get_user(email)
    if user is None:
        raise ValueError("not found")
    return user
```

---

## Task 7 — LRU + TTL with amortized O(1)

TTL is measured from the last `put()`, not `get()`. `get()` updates LRU position only. Expired entries may be removed lazily. Operations must avoid scanning the whole cache.

```python
import time
from collections import OrderedDict, deque


class TTLCache:
    def __init__(self, capacity: int, ttl_seconds: float):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")

        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._data = OrderedDict()
        self._expiry = deque()
        self._generation = 0

    def _purge_expired(self, now: float) -> None:
        while self._expiry and self._expiry[0][0] <= now:
            expires_at, key, generation = self._expiry.popleft()
            item = self._data.get(key)
            if item is None:
                continue
            _, current_expires, current_generation = item
            if (
                current_generation == generation
                and current_expires == expires_at
            ):
                del self._data[key]

    def get(self, key):
        now = time.monotonic()
        self._purge_expired(now)

        item = self._data.get(key)
        if item is None:
            return None

        value, _, _ = item
        self._data.move_to_end(key)
        return value

    def put(self, key, value):
        now = time.monotonic()
        self._purge_expired(now)

        self._generation += 1
        generation = self._generation
        expires_at = now + self.ttl_seconds

        if key in self._data:
            del self._data[key]

        self._data[key] = (value, expires_at, generation)
        self._expiry.append((expires_at, key, generation))

        if len(self._data) > self.capacity:
            self._data.popitem(last=False)
```

With a fixed TTL and monotonic time, expiry timestamps appended by `put()` are nondecreasing. Each expiry record is appended and popped at most once, giving amortized O(1) cleanup.

---

## Task 8 — Bounded-concurrency async runner

At most `limit` jobs may execute concurrently. Results must preserve input order. One job failure must not cancel the others. A very large generator must not cause one task per job to be allocated. External cancellation of `run_jobs()` must cancel and await all workers.

```python
import asyncio
from collections.abc import Iterable, Callable, Awaitable
from typing import Any


async def run_jobs(
    jobs: Iterable[Callable[[], Awaitable[Any]]],
    limit: int,
):
    if limit <= 0:
        raise ValueError("limit must be > 0")

    iterator = iter(jobs)
    next_index = 0
    results: list[tuple[int, Any]] = []

    def take_next():
        nonlocal next_index
        try:
            job = next(iterator)
        except StopIteration:
            return None
        idx = next_index
        next_index += 1
        return idx, job

    async def worker():
        while True:
            item = take_next()
            if item is None:
                return

            idx, job = item
            try:
                result = await job()
            except asyncio.CancelledError as exc:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                result = exc
            except Exception as exc:
                result = exc

            results.append((idx, result))

    workers = [asyncio.create_task(worker()) for _ in range(limit)]

    try:
        await asyncio.gather(*workers)
    except asyncio.CancelledError:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise
    except BaseException:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        raise

    results.sort(key=lambda item: item[0])
    return [value for _, value in results]
```

Only `limit` worker tasks are active, so a huge job generator does not create O(n) tasks. The result list itself is necessarily O(n) because the API returns all results.

---

## Manual review score

| Task | AWQ W4A16 | FP8 |
|---|---:|---:|
| 1 | 9.5 | 10.0 |
| 2 | 10.0 | 10.0 |
| 3 | 10.0 | 10.0 |
| 4 | 9.5 | 9.5 |
| 5 | 10.0 | 10.0 |
| 6 | 9.5 | 10.0 |
| 7 | 10.0 | 10.0 |
| 8 | 9.5 | 10.0 |
| **Total** | **78.0 / 80** | **79.5 / 80** |

This is a small practical test, not a replacement for SWE-bench or HumanEval. Its purpose was to see whether obvious W4 quantization degradation would appear on boundary conditions, async cancellation, multi-file semantics and complexity constraints. It did not in this run.
