from .Config import Config

constants = Config({"g": 0.6 # "+" for repulsive, "-" for attractive
         ,"m": 1.27
         ,"e1": 0.303
         ,"eps": 0.01
         ,"dimfactor": (1.973**2)*10**5})
constants.update({"mu": constants["m"]/2})

config = Config({"numThreads": 4})
