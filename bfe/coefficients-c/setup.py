from distutils.core import setup
from distutils.extension import Extension
from Cython.Build  import cythonize

extensions = Extension(
        name="coefficients",
        sources=["coefficients.pyx"],
        libraries=["gsl", "gslcblas", "m"],
        )

setup(
    name=" coefficients",
    ext_modules=cythonize([extensions], language_level=3)
)

