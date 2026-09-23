"""Bounded synchronous search execution with caller ContextVars per task."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import os


def runtime_limit(name: str, default: int, *, maximum: int = 8) -> int:
    """Read an infrastructure concurrency cap without changing run snapshots."""
    try:
        return min(max(int(os.getenv(name, str(default))), 1), maximum)
    except ValueError:
        return default


def map_in_order(fn, items, max_workers: int):
    """Run independent calls concurrently and return in input order.

    A Context object cannot be entered by two threads at once, so every item
    receives its own copy. The copied cache scope shares only synchronized stats.
    """
    items = list(items)
    if max_workers <= 1 or len(items) < 2:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = [pool.submit(copy_context().run, fn, item) for item in items]
        return [future.result() for future in futures]
