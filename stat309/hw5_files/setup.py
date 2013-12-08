from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

gsdl_source = ['gsdl.pyx']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("gsdl", gsdl_source, extra_compile_args=['-O3', '-std=gnu99'])],
    include_dirs=[np.get_include()]
)
