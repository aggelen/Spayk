#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    sts = system(command);
    return PyLong_FromLong(sts);
}

// module methods
static PyMethodDef SpamModuleMethods[] = {
    { "system", spam_system, METH_VARARGS, "Execute system's command line from input string." }
};
// module definition
static PyModuleDef SpamModuleDef = {
    PyModuleDef_HEAD_INIT,
    "SpamModule",
    "SpamModule for practicing",
    -1,
    SpamModuleMethods
};

PyMODINIT_FUNC PyInit_spam(void)
{
    Py_Initialize();
    return PyModule_Create(&SpamModuleDef);
}