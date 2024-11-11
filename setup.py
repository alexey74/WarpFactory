import numpy
import pythran
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='WarpFactory',
    packages=['analyzer', 'analyzer.utils', 'constants', 'solver', 'metrics', 'metrics.tests', 'metrics.utils', 'misc', 'solver.tests', 'solver.utils'],
    author='Lina',
    ext_modules=cythonize(['.\\metrics\\utils\\sph2cart_diag.pyx', '.\\metrics\\utils\\alphanumeric_solver.pyx',
                           '.\\metrics\\utils\\find_min_idx.pyx', '.\\metrics\\utils\\compact_sigmoid.pyx',
                           '.\\metrics\\utils\\shape_func_alcubierre.pyx', '.\\solver\\utils\\legendre_radial_interp.pyx'],
                          compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include(), pythran.get_include()],
)
