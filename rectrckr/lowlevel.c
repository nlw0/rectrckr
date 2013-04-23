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

  if(direction == 0) {
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
  else if(direction==1) {
    dd1 = dev(img + cy * dimx + (cx - 1), 1);
    dd2 = dev(img + cy * dimx + cx, 1);
    for(j = cx; j < dimx - 3; j++) {
      dd0 = dd1;
      dd1 = dd2;
      dd2 = dev(img + cy * dimx + (j+1), 1);
      if ((dd1 < -10.0) && (dd1 < dd0) && (dd1 < dd2))
        break;
    }
  }
  else if(direction == 2) {
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
  else if(direction == 3) {
    dd1 = dev(img + (cy - 1) * dimx + cx, dimx);
    dd2 = dev(img + cy * dimx + cx, dimx);
    for(j = cy; j < dimy - 3; j++) {
      dd0 = dd1;
      dd1 = dd2;
      dd2 = dev(img + (j+1) * dimx + cx, dimx);
      if (dd1 < -10.0 && dd1 < dd0 && dd1 < dd2)
        break;
    }
  }
  
  Py_DECREF(img_obj);
  return Py_BuildValue("i", j);
}


static PyObject * find_edges_gradient(PyObject *self, PyObject *args) {
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
  double dx0, dx1, dx2;
  double dy0, dy1, dy2;
  double dd0, dd1, dd2;

  if(direction == 0) {
    gradient(&dx1, &dy1, img, dimx, cx + 1, cy);
    dd1=dx1*dx1+dy1*dy1;
    gradient(&dx2, &dy2, img, dimx, cx, cy);
    dd2=dx2*dx2+dy2*dy2;
    for(j = cx; j > 2; j--) {
      dx0 = dx1; dy0 = dy1; dd0 = dd1;
      dx1 = dx2; dy1 = dy2; dd1 = dd2;
      gradient(&dx2, &dy2, img, dimx, j-1, cy);
      dd2=dx2*dx2+dy2*dy2;
      if (dd1 > (10*10.0) &&
          dd1 > dd0 &&
          dd1 > dd2)
        break;     
    }
  }
  if(direction == 1) {
    gradient(&dx1, &dy1, img, dimx, cx - 1, cy);
    dd1=dx1*dx1+dy1*dy1;
    gradient(&dx2, &dy2, img, dimx, cx, cy);
    dd2=dx2*dx2+dy2*dy2;
    for(j = cx; j < dimx - 3; j++) {
      dx0 = dx1; dy0 = dy1; dd0 = dd1;
      dx1 = dx2; dy1 = dy2; dd1 = dd2;
      gradient(&dx2, &dy2, img, dimx, j+1, cy);
      dd2=dx2*dx2+dy2*dy2;
      if (dd1 > (10*10.0) &&
          dd1 > dd0 &&
          dd1 > dd2)
        break;     
    }
  }

  if(direction == 2) {
    gradient(&dx1, &dy1, img, dimx, cx, cy + 1);
    dd1=dx1*dx1+dy1*dy1;
    gradient(&dx2, &dy2, img, dimx, cx, cy);
    dd2=dx2*dx2+dy2*dy2;
    for(j = cy; j > 2; j--) {
      dx0 = dx1; dy0 = dy1; dd0 = dd1;
      dx1 = dx2; dy1 = dy2; dd1 = dd2;
      gradient(&dx2, &dy2, img, dimx, cx, j-1);
      dd2=dx2*dx2+dy2*dy2;
      if (dd1 > (10*10.0) &&
          dd1 > dd0 &&
          dd1 > dd2)
        break;     
    }
  }
  if(direction == 3) {
    gradient(&dx1, &dy1, img, dimx, cx, cy - 1);
    dd1=dx1*dx1+dy1*dy1;
    gradient(&dx2, &dy2, img, dimx, cx, cy);
    dd2=dx2*dx2+dy2*dy2;
    for(j = cy; j < dimy - 3; j++) {
      dx0 = dx1; dy0 = dy1; dd0 = dd1;
      dx1 = dx2; dy1 = dy2; dd1 = dd2;
      gradient(&dx2, &dy2, img, dimx, cx, j+1);
      dd2=dx2*dx2+dy2*dy2;
      if (dd1 > (10*10.0) &&
          dd1 > dd0 &&
          dd1 > dd2)
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


double SHIGERU_FILTER[] = {
  -0.003776, -0.010199, 0., 0.010199, 0.003776,
  -0.026786, -0.070844, 0., 0.070844, 0.026786,
  -0.046548, -0.122572, 0., 0.122572, 0.046548,
  -0.026786, -0.070844, 0., 0.070844, 0.026786,
  -0.003776, -0.010199, 0., 0.010199, 0.003776
};

/* Scaled and rounded version of the Shigeru filter. Result should be divided
   by 256 after multiplication. */
npy_int INT_SHIGERU_FILTER[] = {
  -1, -2, 0, 2, 1,
  -7, -18, 0, 18, 7,
  -12, -32, 0, 32, 12,
  -7, -18, 0, 18, 7,
  -1, -2, 0, 2, 1
};

static PyObject * gradient_wrapper(PyObject *self, PyObject *args) {
  PyObject *input_img;
  PyArrayObject *img_obj;
  npy_int cx, cy;

  if (!PyArg_ParseTuple(args, "Oii", &input_img, &cx, &cy))
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

  double dx = 0, dy = 0;
  
  gradient(&dx, &dy, img, dimx, cx, cy);

  Py_DECREF(img_obj);
  return Py_BuildValue("dd", dx, dy);
}

void gradient(double* dx, double* dy, double* img, npy_int dimx, npy_int cx, npy_int cy) {
  npy_int j, k;
    
  for (j = -2; j < 3; j++) {
    for (k = -2; k < 3; k++) {
      (*dx) += img[(cy+j) * dimx + (cx+k)] * SHIGERU_FILTER[(2+j) * 5 + (2+k)];
      (*dy) += img[(cy+j) * dimx + (cx+k)] * SHIGERU_FILTER[(2+k) * 5 + (2+j)];
    }
  }  
}

static PyMethodDef LowLevelMethods[] =
  {
    {"find_edges", find_edges, METH_VARARGS,
     "Find edges on line sweeps.\n"},
    {"find_edges_gradient", find_edges_gradient, METH_VARARGS,
     "Find edges on line sweeps, calculating the image gradient at each point.\n"},
    {"linear_derivative", linear_derivative, METH_VARARGS,
     "Calculate the derivative on a line or column.\n"},
    {"gradient", gradient_wrapper, METH_VARARGS,
     "Image gradient around a given point, using the Shigeru filter.\n"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC

initlowlevel(void) {
  (void) Py_InitModule("lowlevel", LowLevelMethods);
  import_array();
}
