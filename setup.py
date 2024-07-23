from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "spayk.CIntegrators",
        ["spayk/CIntegrators.pyx"],
        # include_dirs=['/some/path/to/include/'], # not needed for fftw unless it is installed in an unusual place
        # libraries=['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads', 'fftw3f_threads', 'fftw3l_threads'],
        # library_dirs=['/some/path/to/include/'], # not needed for fftw unless it is installed in an unusual place
    ),
]

setup(
    name='Spayk',
    version='0.1.1',    
    description='SPAYK: An environment for spiking neural network simulation',
    url='https://github.com/aggelen/Spayk',
    author='Aykut Görkem Gelen',
    author_email='aggelen@outlook.com',
    license='BSD 3-clause',
    packages=['spayk'],
    install_requires=['matplotlib==3.5.0',
                      'numpy==1.25.1',
                      'seaborn==0.11.2',
                      'tqdm==4.62.3',    
                      'torchvision==0.16.0',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    include_dirs=[numpy.get_include()],
    ext_modules = cythonize(extensions)
)


