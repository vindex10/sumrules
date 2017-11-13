"""@package sumrules
Implementation of everything needed for computation of sumrules.

And even more... :)
"""

from .Config import Config

## @var constants
#This variable is used for managing models' global parameters
#
#Description of fields:
#   * g: positive for repulsive and negative for attractive case,
#   * m: doubled reduced mass of bound state,
#   * e: charge of particles in bound state,
#   * Nc: number of colors in QCD,
#   * eps: needed to regularize wave functions in momentum representation

constants = Config({"g": 0.6
                   ,"m": 1.27
                   ,"e": 0.303*(2/3)
                   ,"Nc": 3
                   ,"eps": 0.01
                   ,"dimfactor": (1.973**2)*10**5
                   })


## @var config
#  Technical configuration variables
#
# Description of fields:
#   * numThreads: number of threads available for multiprocessing

config = Config({"numThreads": 4
                ,"maxTaskPerChild": 1000
                }) 
