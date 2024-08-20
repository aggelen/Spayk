#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:39:12 2024

@author: gelenag
"""

import os
import subprocess
import numpy as np
import shutil

def generate_cython_code():
    cython_code = """
from libc.stdint cimport int32_t
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cpdef int compute_sum(int[:] numbers):
    cdef int total = 0
    cdef int i
    for i in range(numbers.shape[0]):
        total += numbers[i]
    return total
"""
    with open("generated_code.pyx", "w") as f:
        f.write(cython_code)

def create_and_run_setup_py():
    setup_code = """
from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.core import Extension
import numpy

ext_modules = [
    Extension(
        name="generated_code",
        sources=["generated_code.pyx"],
        include_dirs=[numpy.get_include()]  # Ensure numpy headers are included
    )
]

setup(
    name="generated_code",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': "3"}
    ),
    cmdclass={'build_ext': build_ext}
)
"""
    with open("setup.py", "w") as f:
        f.write(setup_code)

    result = subprocess.run([os.sys.executable, "setup.py", "build_ext", "--inplace"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to build Cython extension")

def test_cython_code():
    import generated_code
    if not hasattr(generated_code, 'compute_sum'):
        raise AttributeError("Module 'generated_code' has no attribute 'compute_sum'")
    numbers = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result = generated_code.compute_sum(numbers)
    print(f"Sum using Cython: {result}")
    return result

def clean_up():
    os.remove("setup.py")
    os.remove("generated_code.pyx")

    build_dir = "build"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    all_files = os.listdir()
    for item in all_files:
        if item.endswith(".so") or item.endswith(".c") or item.endswith(".cpp") or item.endswith(".pyx"):
            os.remove(item)


generate_cython_code()
create_and_run_setup_py()
result = test_cython_code()
clean_up()



