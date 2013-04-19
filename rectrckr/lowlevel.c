#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

double dev(double * p, npy_int step) {
  return (-0.212 * p[-2*step] - 0.5 * p[-1*step] + 0.5 * p[1*step] + 0.212 * p[2*step]);
}

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
      if ((dd1 < -10.0) && (dd1 < dd0) && (dd1 < dd2))
        break;
    }
  }
  else if(direction == 1) {
    dd1 = dev(img + cy * dimx + (cx + 1), 1);
    dd2 = dev(img + cy * dimx + cx, 1);
    for(j = cx; j > 2; j--) {
      dd0 = dd1;
      dd1 = dd2;
      dd2 = dev(img + cy * dimx + (j-1), 1);
      if (dd1 > 10.0 && dd1 > dd0 && dd1 > dd2)
        break;     
    }
  }
  else if(direction == 2) {
    dd1 = dev(img + (cy - 1) * dimx + cx, dimx);
    dd2 = dev(img + cy * dimx + cx, dimx);
    for(j = cy; j < dimy-2; j++) {
      dd0 = dd1;
      dd1 = dd2;
      dd2 = dev(img + (j+1) * dimx + cx, dimx);
      if (dd1 < -10.0 && dd1 < dd0 && dd1 < dd2)
        break;
    }
  }
  else if(direction == 3) {
    dd1 = dev(img + (cy + 1) * dimx + cx, dimx);
    dd2 = dev(img + cy * dimx + cx, dimx);
    for(j = cy; j > 2; j--) {
      dd0 = dd1;
      dd1 = dd2;
      dd2 = dev(img + (j-1) * dimx + cx, dimx);
      if (dd1 > 10.0 && dd1 > dd0 && dd1 > dd2)
        break;
    }
  }

  
  
  Py_DECREF(img_obj);
  return Py_BuildValue("i", j);
}


static PyObject * linear_derivative(PyObject *self, PyObject *args) {
  PyObject *input_img;
  PyArrayObject *img_obj;
  npy_int cx, cy;
  npy_int direction;

  if (!PyArg_ParseTuple(args, "Oiii", &input_img, &cx, &cy, &direction))
    return NULL;

  img_obj = (PyArrayObject *)
    PyArray_ContiguousFromObject(input_img, PyArray_DOUBLE, 2, 2);
  if (img_obj == NULL) return NULL;

  npy_int dimx = img_obj->dimensions[1];
  npy_int dimy = img_obj->dimensions[0];
  double *img = (double*)img_obj->data;

  if ((cx < 2) || (cx > dimx-2) ||
      (cy < 2) || (cy > dimy-2))
    {
      Py_DECREF(img_obj);
      Py_RETURN_NONE;
    }

  double dd;

  if (direction==0) {
    dd = dev(img + cy * dimx + cx, 1);
  } else {
    dd = dev(img + cy * dimx + cx, dimx);
  }
  
  Py_DECREF(img_obj);
  return Py_BuildValue("d", dd);
}

static PyMethodDef LowLevelMethods[] =
  {
    {"find_edges", find_edges, METH_VARARGS,
     "Find edges on line sweeps.\n"},
    {"linear_derivative", linear_derivative, METH_VARARGS,
     "Calculate the derivative on a line or column.\n"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC

initlowlevel(void) {
  (void) Py_InitModule("lowlevel", LowLevelMethods);
  import_array();
}
