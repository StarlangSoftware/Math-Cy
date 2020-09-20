from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["Math/*.pyx", "Math/*.pxd"],
                          compiler_directives={'language_level': "3"}),
    name='NlpToolkit-Math-Cy',
    version='1.0.0',
    packages=['Math'],
    url='https://github.com/olcaytaner/Math-Cy',
    license='',
    author='olcaytaner',
    author_email='olcaytaner@isikun.edu.tr',
    description='Math library'
)