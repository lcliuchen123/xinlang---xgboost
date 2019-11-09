
import time
from functools import wraps

def run_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        print(func.__name__,end-start)
        return result
    return wrapper

# @run_time
# def add(a,b):
#     return a+b
# print(add(3,4))
