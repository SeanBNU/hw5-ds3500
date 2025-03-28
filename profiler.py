import time
from collections import defaultdict
from functools import wraps

def profile(f):
    return Profiler.profile(f)

class Profiler:
    calls = defaultdict(int)
    time = defaultdict(float)
    
    @staticmethod
    def profile(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time_ns()
            val = f(*args, **kwargs)
            end = time.time_ns()
            elapsed = (end - start) / 10**9
            Profiler.calls[f.__name__] += 1
            Profiler.time[f.__name__] += elapsed
            return val
        return wrapper
    
    @staticmethod
    def report():
        print("Function              Calls     TotSec   Sec/Call")
        for name, num in Profiler.calls.items():
            sec = Profiler.time[name]
            print(f'{name:20s} {num:6d} {sec:10.6f} {sec / num:10.6f}')
