import scipy as sp
import multiprocessing.dummy
from . import config

def mpMap(f, data):
    pool = multiprocessing.dummy.Pool(config["numThreads"])
    res = pool.map(f, data)
    return sp.hstack(res)

def npMap(f, data):
    return f(data.T).T

def pyMap(f, data):
    return sp.hstack(list(map(f, data)))
