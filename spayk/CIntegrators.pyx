#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:18:21 2024

@author: gelenag
"""

import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.float64_t real
ctypedef unsigned int number

cdef class Array:
    cdef real* data

    def __setitem__(self, number index, real value):
        self.data[index] = value

    def __getitem__(self, number index):
        return self.data[index]

cdef class ODES:

    cdef int nvars

    def __cinit__(self):
        self.nvars = 2

    def func(self, np.ndarray[real, ndim=1] x, real t,
                   np.ndarray[real, ndim=1] dxdt):
        self._func(<real*>x.data, t, <real*>dxdt.data)

    cdef void _func(self, real* x, real t, real* dxdt):
        raise NotImplementedError

    cpdef euler(self, np.ndarray[real, ndim=1] x0, np.ndarray[real, ndim=1] t):

        cdef int n, m, N, M
        cdef np.ndarray[real, ndim=2] X
        cdef np.ndarray[real, ndim=1] x
        cdef np.ndarray[real, ndim=1] dxdt
        cdef real dt, tcur, tlast, *px, *pt, *pdxdt, *pX

        N = len(x0)
        M = len(t)

        X = np.zeros((M, N), float)
        x = np.zeros(N, float)
        dxdt = np.zeros(N, float)

        px = <real*>x.data
        pt = <real*>t.data
        pdxdt = <real*>dxdt.data
        pX = <real*>X.data

        # Pre-loop setup
        for n in range(N):
            pX[0 + n] = px[n] = <real>x0[n]
        tlast = t[0]

        # Main loop
        for m in range(1, M):
            tcur = pt[m]
            dt = tcur - tlast
            self._func(px, tlast, pdxdt)
            for n in range(N):
                px[n] += pdxdt[n] * dt
                pX[m*N + n] = px[n]
            tlast = tcur
        return X

cdef class pyODES(ODES):

    cdef void _func(self, real* x, real t, real* dxdt):
        cdef Array ax, adxdt
        ax = Array()
        adxdt = Array()
        ax.data = x
        adxdt.data = dxdt
        self.func(ax, t, adxdt)

    cpdef func(self, Array x, real t, Array dxdt):
        raise NotImplementedError

cdef class ODES_sub(ODES):

    cdef void _func(self, real* x, real t, real* dxdt):
        dxdt[0] = x[1]
        dxdt[1] = - x[0]

def euler(x0, t):
    return ODES_sub().euler(x0, t)

# import numpy as np
# cimport numpy as np

# ctypedef np.float64_t DTYPE_t

# cpdef euler(f, np.ndarray[DTYPE_t, ndim=1] x0, np.ndarray[DTYPE_t, ndim=1] t):
#     cdef int n, m, N, M
#     cdef np.ndarray[DTYPE_t, ndim=2] X
#     cdef np.ndarray[DTYPE_t, ndim=1] x, dxdt
#     cdef double dt, tcur, tlast

#     N = len(x0)
#     M = len(t)

#     X = np.zeros((M, N), float)
#     x = np.zeros(N, float)
#     dxdt = np.zeros(N, float)

#     # Pre-loop setup
#     for n in range(N):
#         X[0, n] = x[n] = x0[n]
#     tlast = t[0]

#     # Main loop
#     for m in range(1, M):
#         tcur = t[m]
#         dt = tcur - tlast
#         f(x, tlast, dxdt)
#         x += dxdt * dt
#         X[m, :] = x
#         tlast = tcur
#     return X

# cpdef func(np.ndarray[DTYPE_t, ndim=1] x, double t,
#            np.ndarray[DTYPE_t, ndim=1] dxdt):
#     dxdt[0] = x[1]
#     dxdt[1] = - x[0]