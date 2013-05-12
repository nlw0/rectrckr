#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <stdio.h>

#include "camera_models.h"
#include "target.h"

double dev(double * p, npy_int step) {
  return (-0.212 * p[-2*step] - 0.5 * p[-1*step] + 0.5 * p[1*step] + 0.212 * p[2*step]);
}
void gradient(double*, double*, double*, npy_int, npy_int, npy_int);

/* Sweeps the image on the given direction, until an edge is detected. */
static PyObject * find_edge(PyObject *self, PyObject *args) {
  PyObject *input_img;
  PyArrayObject *img_obj;
  npy_int px, py;
  npy_int direction;

  if (!PyArg_ParseTuple(args, "Oiii", &input_img, &px, &py, &direction))
    return NULL;

  img_obj = (PyArrayObject *)
    PyArray_ContiguousFromObject(input_img, PyArray_DOUBLE, 2, 2);
  if (img_obj == NULL) return NULL;

  npy_int dimy = img_obj->dimensions[0];
  npy_int dimx = img_obj->dimensions[1];
  double *img = (double*)img_obj->data;

  npy_int j=0;
  double dx0, dx1, dx2;
  double dy0, dy1, dy2;
  double dd0, dd1, dd2;

  double edge_threshold = 100.0;

  if(direction == 0) {
    gradient(&dx1, &dy1, img, dimx, px + 1, py);
    dd1 = dx1 * dx1 + dy1 * dy1;
    gradient(&dx2, &dy2, img, dimx, px, py);
    dd2 = dx2 * dx2 + dy2 * dy2;
    for(j = px; j > 2; j--) {
      dx0 = dx1; dy0 = dy1; dd0 = dd1;
      dx1 = dx2; dy1 = dy2; dd1 = dd2;

      gradient(&dx2, &dy2, img, dimx, j-1, py);
      dd2 = dx2 * dx2 + dy2 * dy2;

      if (dd1 > (edge_threshold) &&
          dd1 > dd0 &&
          dd1 > dd2)
        break;     
    }
  }

  if(direction == 1) {
    gradient(&dx1, &dy1, img, dimx, px - 1, py);
    dd1=dx1*dx1+dy1*dy1;
    gradient(&dx2, &dy2, img, dimx, px, py);
    dd2=dx2*dx2+dy2*dy2;
    for(j = px; j < dimx - 3; j++) {
      dx0 = dx1; dy0 = dy1; dd0 = dd1;
      dx1 = dx2; dy1 = dy2; dd1 = dd2;

      gradient(&dx2, &dy2, img, dimx, j+1, py);
      dd2=dx2*dx2+dy2*dy2;

      if (dd1 > (edge_threshold) &&
          dd1 > dd0 &&
          dd1 > dd2)
        break;     
    }
  }

  if(direction == 2) {
    gradient(&dx1, &dy1, img, dimx, px, py + 1);
    dd1=dx1*dx1+dy1*dy1;
    gradient(&dx2, &dy2, img, dimx, px, py);
    dd2=dx2*dx2+dy2*dy2;
    for(j = py; j > 2; j--) {
      dx0 = dx1; dy0 = dy1; dd0 = dd1;
      dx1 = dx2; dy1 = dy2; dd1 = dd2;
      gradient(&dx2, &dy2, img, dimx, px, j-1);
      dd2=dx2*dx2+dy2*dy2;
      if (dd1 > (edge_threshold) &&
          dd1 > dd0 &&
          dd1 > dd2)
        break;     
    }
  }

  if(direction == 3) {
    gradient(&dx1, &dy1, img, dimx, px, py - 1);
    dd1=dx1*dx1+dy1*dy1;
    gradient(&dx2, &dy2, img, dimx, px, py);
    dd2=dx2*dx2+dy2*dy2;
    for(j = py; j < dimy - 3; j++) {
      dx0 = dx1; dy0 = dy1; dd0 = dd1;
      dx1 = dx2; dy1 = dy2; dd1 = dd2;
      gradient(&dx2, &dy2, img, dimx, px, j+1);
      dd2=dx2*dx2+dy2*dy2;
      if (dd1 > (edge_threshold) &&
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
  npy_int px, py;
  npy_int direction;

  if (!PyArg_ParseTuple(args, "Oiii", &input_img, &px, &py, &direction))
    return NULL;

  img_obj = (PyArrayObject *)
    PyArray_ContiguousFromObject(input_img, PyArray_DOUBLE, 2, 2);
  if (img_obj == NULL) return NULL;

  npy_int dimx = img_obj->dimensions[1];
  npy_int dimy = img_obj->dimensions[0];
  double *img = (double*)img_obj->data;

  if ((px < 2) || (px > dimx-2) ||
      (py < 2) || (py > dimy-2))
    {
      Py_DECREF(img_obj);
      Py_RETURN_NONE;
    }

  double dd;

  if (direction==0) {
    dd = dev(img + py * dimx + px, 1);
  } else {
    dd = dev(img + py * dimx + px, dimx);
  }
  
  Py_DECREF(img_obj);
  return Py_BuildValue("d", dd);
}


static PyObject * gradient_wrapper(PyObject *self, PyObject *args) {
  PyObject *input_img;
  PyArrayObject *img_obj;
  npy_int px, py;

  if (!PyArg_ParseTuple(args, "Oii", &input_img, &px, &py))
    return NULL;

  img_obj = (PyArrayObject *)
    PyArray_ContiguousFromObject(input_img, PyArray_DOUBLE, 2, 2);
  if (img_obj == NULL) return NULL;

  npy_int dimx = img_obj->dimensions[1];
  npy_int dimy = img_obj->dimensions[0];
  double *img = (double*)img_obj->data;

  if ((px < 2) || (px > dimx-2) ||
      (py < 2) || (py > dimy-2))
    {
      Py_DECREF(img_obj);
      Py_RETURN_NONE;
    }

  double dx = 0, dy = 0;
  
  gradient(&dx, &dy, img, dimx, px, py);

  Py_DECREF(img_obj);
  return Py_BuildValue("dd", dx, dy);
}

double SHIGERU_FILTER[] = {
  -0.003776, -0.010199, 0., 0.010199, 0.003776,
  -0.026786, -0.070844, 0., 0.070844, 0.026786,
  -0.046548, -0.122572, 0., 0.122572, 0.046548,
  -0.026786, -0.070844, 0., 0.070844, 0.026786,
  -0.003776, -0.010199, 0., 0.010199, 0.003776
};

void gradient(double* dx, double* dy, double* img, npy_int dimx, npy_int px, npy_int py) {
  npy_int j, k;
    
  *dx = 0;
  *dy = 0;
  for (j = -2; j < 3; j++) {
    for (k = -2; k < 3; k++) {
      (*dx) += img[(py+j) * dimx + (px+k)] * SHIGERU_FILTER[(2+j) * 5 + (2+k)];
      (*dy) += img[(py+j) * dimx + (px+k)] * SHIGERU_FILTER[(2+k) * 5 + (2+j)];
    }
  }  
}

static PyObject * p_wrapper(PyObject *self, PyObject *args) {
  PyArrayObject *s_obj;
  PyArrayObject *t_obj;
  PyArrayObject *psi_obj;
  PyArrayObject *cm_obj;

  if (!PyArg_ParseTuple(args, "O!O!O!O!",
                        &PyArray_Type, &s_obj,
                        &PyArray_Type, &t_obj,
                        &PyArray_Type, &psi_obj,
                        &PyArray_Type, &cm_obj))
    return NULL;

  double *s = (double*)s_obj->data;
  double *t = (double*)t_obj->data;
  double *psi = (double*)psi_obj->data;
  CameraModel cm = *((CameraModel*)cm_obj->data);
  
  Vector2D pp = project(q(s, t, psi), cm);
  
  return Py_BuildValue("dd", pp.x, pp.y);
  
  // double *cm = (double*)cm_obj->data;
  // return Py_BuildValue("iddd", cm[0], cm[1], cm[2], cm[3]);
}


static PyMethodDef LowLevelMethods[] =
  {
    {"find_edge", find_edge, METH_VARARGS,
     "Sweep an image line on the given direction, until the first edge is detected.\n"},
    {"linear_derivative", linear_derivative, METH_VARARGS,
     "Calculate the derivative on a line or column.\n"},
    {"p_wrapper", p_wrapper, METH_VARARGS,
     "yah yah.\n"},
    {"gradient", gradient_wrapper, METH_VARARGS,
     "Image gradient around a given point, using the Shigeru filter.\n"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC

initlowlevel(void) {
  (void) Py_InitModule("lowlevel", LowLevelMethods);
  import_array();
}
