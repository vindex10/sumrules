from .Config import Config

constants = Config({"g": 0.6 # "+" for repulsive, "-" for attractive
                   ,"m": 1.27
                   ,"e": 0.303*(2/3)
                   ,"Nc": 3
                   ,"eps": 0.01
                   ,"dimfactor": (1.973**2)*10**5})

config = Config({"numThreads": 4})
