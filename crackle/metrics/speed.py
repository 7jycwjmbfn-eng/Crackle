from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer_ms(target: dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        target[key] = target.get(key, 0.0) + 1000.0 * (time.perf_counter() - start)
