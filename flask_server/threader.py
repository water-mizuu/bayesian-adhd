import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
executor = ThreadPoolExecutor(max_workers=4)  # adjust for your CPU

def threaded(func):
    """
    Decorator to run a function in a background thread pool.
    Works with both sync and async Flask views.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop in current thread (regular Flask sync)
            pass

        if loop and loop.is_running():
            # If inside an async context (Quart / FastAPI)
            return loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))
        else:
            # Synchronous Flask: block until done (but in another thread)
            future = executor.submit(func, *args, **kwargs)
            return future.result()

    return wrapper