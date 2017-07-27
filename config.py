import os

config = {"num_threads": 4
         ,"abs_err": 10**(-4)}

for cp in config.keys():
    ep = cp.upper()
    if ep in os.environ.keys():
        config[cp] = os.environ[ep]
