from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("rsipoly/models/utils/polygon_processor_cython.pyx")
)
