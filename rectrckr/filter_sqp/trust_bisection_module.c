#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"




double length(double alpha, npy_int Nlam, double* lam, double x)
{
  double output = 0.0;
  int i;
  for (i=0; i<Nlam; ++i)
    {
      double y = alpha/(lam[i]+x);  
      output += y * y;
    }
  return sqrt(output);
}


static PyObject * calculate_distance(PyObject *self, PyObject *args)
{  
  double alpha;
  double x;
  PyArrayObject *lam_obj;

  if (!PyArg_ParseTuple(args, "dO!d",
                        &alpha,
                        &PyArray_Type, &lam_obj,
                        &x))
    return NULL;

  double *lam = (double*)lam_obj->data;
  npy_int Nlam = lam_obj->dimensions[0];

  return Py_BuildValue("d", length(alpha, Nlam, lam, x));
}

static PyObject * find_step_size(PyObject *self, PyObject *args)
{  
  double alpha;
  PyArrayObject *lam_obj;

  if (!PyArg_ParseTuple(args, "dO!",
                        &alpha,
                        &PyArray_Type, &lam_obj))
    return NULL;

  double *lam = (double*)lam_obj->data;
  npy_int Nlam = lam_obj->dimensions[0];

  /* ln is the smallest lambda */
  double ln = lam[0];
  int i;
  for(i=1; i<Nlam; ++i)
    {
      if (lam[i] < ln)
        {
          ln = lam[i];
        }
    }





  return Py_BuildValue("d", alpha);
}

static PyMethodDef TrustBisectionMethods[] =
  {
    {"find_step_size", find_step_size, METH_VARARGS,
     "Help.\n"},
    {"calculate_distance", calculate_distance, METH_VARARGS,
     "Help.\n"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC

inittrust_bisection(void) {
  (void) Py_InitModule("trust_bisection", TrustBisectionMethods);
  import_array();
}
