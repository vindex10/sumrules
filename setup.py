from distutils.core import setup
from Cython.Build import cythonize
import scipy as sp

setup(
    name = "Sumrules",
    ext_modules = cythonize("lib/analytics.pyx", [sp.get_include()])
)
