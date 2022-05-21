from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extension = Extension(
    'draw',
    ['draw.pyx'],
    include_dirs=[numpy.get_include()],
    libraries=["m"]
)

setup(
    ext_modules=cythonize(extension, annotate=True),
)
