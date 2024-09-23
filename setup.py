import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    name='WarpFactory',
    packages=['Solver', 'Solver.utils', 'Metrics', 'Metrics.utils', 'Analyzer', 'Analyzer.utils'],
    author='Lina',
    ext_modules=cythonize(['.\\Metrics\\utils\\sph2cart_diag.pyx', '.\\Metrics\\utils\\alphanumeric_solver.pyx',
                           '.\\Solver\\utils\\legendre_radial_interp.pyx', '.\\Metrics\\utils\\find_min_idx.pyx',
                           '.\\Metrics\\utils\\compact_sigmoid.pyx', '.\\Metrics\\utils\\shape_func_alcubierre.pyx'],
                          compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
)
