# Coding benchmark: задачи и эталонные решения

[Назад к статье](README_RU.md) · [English version](TASKS_AND_SOLUTIONS.md)

Ниже приведены восемь задач, использованных для сравнения Qwen3.8 Flash-Next AWQ W4A16 и FP8. Решения в этом файле — компактные эталонные варианты для проверки требований. Это не попытка воспроизвести дословно ответы одной из моделей.

---

## Задача 1 — Python debugging / correctness

Нужно сгруппировать события пользователей в сессии. Новая сессия начинается, если между соседними событиями одного пользователя прошло **больше 30 минут**. Вход может быть неотсортирован. Входной список нельзя модифицировать. Результат должен быть отсортирован по `user_id`, затем по началу сессии. Разрыв ровно 30 минут остаётся внутри одной сессии. Сложность — не хуже `O(n log n)`.

### Эталонное решение

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

Ключевые ошибки исходного варианта: он предполагал уже отсортированный вход и использовал `>= 30 минут`, хотя по ТЗ ровно 30 минут должны оставаться в той же сессии.

---

## Задача 2 — Async Python / race condition

`AsyncCache.get(key, factory)` должен гарантировать одно общее дорогое вычисление для одного ключа, параллельность для разных ключей, повтор после исключения и отсутствие отмены общего вычисления при отмене одного клиента.

### Эталонное решение

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

Глобальный lock держится только на коротких операциях со словарями; `factory()` выполняется без него. `asyncio.shield()` защищает общее вычисление от отмены отдельного ожидающего клиента.

---

## Задача 3 — минимальный покрывающий диапазон

Дано до 1000 отсортированных списков и до 1 000 000 чисел суммарно. Нужно найти кратчайший диапазон `[L, R]`, содержащий хотя бы одно число из каждого списка. При равной длине выбрать меньший `L`. Нельзя сливать всё в один гигантский отсортированный массив.

### Эталонное решение

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

Пусть `N` — общее число элементов, `K` — число списков. В heap всегда не более `K` элементов, поэтому время `O(N log K)`, память `O(K)`.

---

## Задача 4 — PostgreSQL

Нужно одним запросом вернуть по каждому клиенту количество и сумму `COMPLETED` заказов за последние 30 дней, а также `id` и `created_at` последнего такого заказа. Клиенты без заказов тоже должны присутствовать. Последний заказ определяется по `created_at DESC, id DESC`. Нельзя использовать отдельный запрос на каждого клиента.

### Эталонное решение

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

Практичный индекс:

```sql
CREATE INDEX orders_completed_customer_created_id_idx
ON orders (customer_id, created_at DESC, id DESC)
INCLUDE (total)
WHERE status = 'COMPLETED';
```

Поскольку граница «последние 30 дней» движется, фиксировать её в partial-index нельзя; условие `status='COMPLETED'` — стабильное и полезное.

---

## Задача 5 — TypeScript / строгий parsePort

Нужно принимать `undefined`, `null`, пустую строку как fallback; принимать целые числа и строки только с десятичным целым числом и необязательными пробелами; отклонять `8080abc`, дроби, boolean, массивы, объекты, `NaN`, `Infinity`, а также значения вне `1..65535`. Некорректный fallback должен давать `RangeError`.

### Эталонное решение

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

## Задача 6 — multi-file storage semantics

`register()` и `login()` должны считать email без учёта регистра, при этом сохранять исходную строку email. Внутренний storage не должен отдавать изменяемую ссылку наружу, а результат `register()`/`login()` нельзя использовать для изменения сохранённого объекта.

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

## Задача 7 — LRU + TTL, амортизированное O(1)

TTL считается от последнего `put()`, `get()` только меняет LRU-позицию. Протухшие элементы можно удалять лениво. Нельзя обходить весь cache на каждой операции.

### Эталонное решение

Для LRU удобно использовать `OrderedDict`, а для expiry — `deque`. Поскольку TTL единый и `time.monotonic()` не убывает, каждый новый `put()` создаёт expiry timestamp не меньше предыдущего; значит expiry-записи можно хранить в очереди и лениво вычищать с головы. Обновления ключа получают новую `generation`, поэтому старые expiry-записи становятся безопасно устаревшими.

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

Каждая expiry-запись добавляется и удаляется из `deque` максимум один раз, поэтому операции остаются амортизированно `O(1)`.

---

## Задача 8 — bounded-concurrency run_jobs

Нельзя заранее создавать task на каждый job. Одновременно должно существовать `O(limit)` активных job-задач. Результаты — в исходном порядке. Исключение одной job не отменяет остальные. При внешней отмене `run_jobs()` все worker tasks должны быть отменены и дожидаться завершения.

### Эталонное решение

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

Активных worker tasks ровно `limit`, поэтому огромный генератор jobs не создаёт огромный список task. Возвращаемый список результатов, конечно, занимает `O(n)` — это неизбежно из самой сигнатуры функции.

---

## Итог ручной проверки двух моделей

| Задача | AWQ W4A16 | FP8 |
|---|---:|---:|
| 1 | 9.5 | 10.0 |
| 2 | 10.0 | 10.0 |
| 3 | 10.0 | 10.0 |
| 4 | 9.5 | 9.5 |
| 5 | 10.0 | 10.0 |
| 6 | 9.5 | 10.0 |
| 7 | 10.0 | 10.0 |
| 8 | 9.5 | 10.0 |
| **Всего** | **78.0 / 80** | **79.5 / 80** |

Это небольшой авторский тест, а не замена SWE-bench/HumanEval. Его задача — проверить, проявится ли очевидная деградация 4-битной AWQ на задачах с граничными условиями, async cancellation, multi-file semantics и требованиями к сложности. В этом прогоне сильной деградации не наблюдалось.
