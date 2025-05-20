#File: setup.py
#!/usr/bin/python
from distutils.core import setup, Extension

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()
    
pht_module = Extension('_bilateralfilter', 
                        sources=['bilateralfilter_wrap.cxx',
                                'bilateralfilter.cpp',
                                'permutohedral.cpp'
                                ],
                       extra_compile_args = ["-fopenmp"],
                       include_dirs = [numpy_include]
                      )

setup(name = 'bilateralfilter',	
        version = '0.1',
        author = 'SWIG Docs',
        description = 'Simple swig pht from docs',
        ext_modules = [pht_module], 
        py_modules = ['bilateralfilter'],
    )

