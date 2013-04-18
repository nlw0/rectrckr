#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

double dev(double * p, npy_int step) {
  return (-0.125 * p[-2*step] - 0.25 * p[-1*step] + 0.25 * p[1*step] + 0.125 * p[2*step]);
}

/* Python wrapper to the low-level irf calculation. */
static PyObject * find_edges(PyObject *self, PyObject *args) {
  PyObject *input_img;
  PyArrayObject *img_obj;
  npy_int cx, cy;
  npy_int direction;

  if (!PyArg_ParseTuple(args, "Oiii", &input_img, &cx, &cy, &direction))
    return NULL;

  img_obj = (PyArrayObject *)
    PyArray_ContiguousFromObject(input_img, PyArray_DOUBLE, 2, 2);
  if (img_obj == NULL) return NULL;

  npy_int dimy = img_obj->dimensions[0];
  npy_int dimx = img_obj->dimensions[1];
  double *img = (double*)img_obj->data;

  npy_int j;
  double dd0, dd1, dd2;

  if(direction==0) {
    dd1 = dev(img + cy * dimx + (cx - 1), 1);
    dd2 = dev(img + cy * dimx + cx, 1);
    for(j = cx; j < dimx-2; j++) {
      dd0 = dd1;
      dd1 = dd2;
      dd2 = dev(img + cy * dimx + (j+1), 1);
      if ((dd1 < -10.0) && (dd1 < dd0) && (dd1 < dd2)) {
        break;
      }
    }
  }
  else if(direction == 1) {
    dd1 = dev(img + cy * dimx + (cx + 1), 1);
    dd2 = dev(img + cy * dimx + cx, 1);
    for(j = cx; j > 2; j--) {
      dd0 = dd1;
      dd1 = dd2;
      dd2 = dev(img + cy * dimx + (j-1), 1);
      if (dd1 > 10.0 && dd1 > dd0 && dd1 > dd2) {
        break;
      }
    }
  }
  else if(direction == 2) {
    dd1 = dev(img + (cy - 1) * dimx + cx, dimx);
    dd2 = dev(img + cy * dimx + cx, dimx);
    for(j = cy; j < dimy-2; j++) {
      dd0 = dd1;
      dd1 = dd2;
      dd2 = dev(img + (j+1) * dimx + cx, dimx);
      if (dd1 < -10.0 && dd1 < dd0 && dd1 < dd2) {
        break;
      }
    }
  }
  else if(direction == 3) {
    dd1 = dev(img + (cy + 1) * dimx + cx, dimx);
    dd2 = dev(img + cy * dimx + cx, dimx);
    for(j = cy; j > 2; j--) {
      dd0 = dd1;
      dd1 = dd2;
      dd2 = dev(img + (j-1) * dimx + cx, dimx);
      if (dd1 > 10.0 && dd1 > dd0 && dd1 > dd2) {
        break;
      }
    }
  }

  
  
  Py_DECREF(img_obj);
  return Py_BuildValue("i", j);
}

static PyMethodDef LowLevelMethods[] =
  {
    {"find_edges", find_edges, METH_VARARGS,
     "Find edges on line sweeps.\n"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC

initlowlevel(void) {
  (void) Py_InitModule("lowlevel", LowLevelMethods);
  import_array();
}
