#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <iostream>



// Stores the model parameters, and calculates a step size for different nu values.
class LengthCalculator{
  double alpha;
  npy_int Nlam;
  double* lam;
  double rho;
public:
  LengthCalculator(double _alpha, npy_int _Nlam, double* _lam, double _rho):
    alpha(_alpha),
    Nlam(_Nlam),
    lam(_lam),
    rho(_rho)
  { }

  double operator()(double nu)
  {    
    double output = 0.0;
    int i;
    for (i=0; i<Nlam; ++i)
      {
        double y = alpha / (lam[i] + nu);
        output += y * y;
      }
    return sqrt(output) - rho;
  }  
};


static PyObject * length_wrapper(PyObject *self, PyObject *args)
{  
  double alpha;
  double nu;
  PyArrayObject *lam_obj;

  if (!PyArg_ParseTuple(args, "dO!d",
                        &alpha,
                        &PyArray_Type, &lam_obj,
                        &nu))
    return NULL;

  double *lam = (double*)lam_obj->data;
  npy_int Nlam = lam_obj->dimensions[0];

  LengthCalculator length(alpha, Nlam, lam, 0.0);

  return Py_BuildValue("d", length(nu));
}


static PyObject * find_step_size(PyObject *self, PyObject *args)
{  
  double alpha;
  PyArrayObject *lam_obj;
  double rho, rho_tol;

  if (!PyArg_ParseTuple(args, "dO!dd",
                        &alpha,
                        &PyArray_Type, &lam_obj,
                        &rho, &rho_tol))
    return NULL;

  double *lam = (double*)lam_obj->data;
  npy_int Nlam = lam_obj->dimensions[0];

  double ln = lam[0]; // the smallest lambda
  for(int i=1; i < Nlam; ++i)
    {
      if (lam[i] < ln)
        {
          ln = lam[i];
        }
    }

  // The lower bound for nu, and the initialization of the two points
  // from the secant method.
  double nu_min=0, nu_a=1, nu_b=2;
  if (ln > 0)
    {
      nu_min = 0.0;
      nu_a = 0.0;
      nu_b = ln;
    }
  else if (ln < 0)
    {
      nu_min = -ln;
      nu_a = nu_min;
      nu_b = nu_min * 2.0;
    }
  else if (ln == 0)
    {
      nu_min = 0;
      nu_a = 1.0;
      nu_b = 2.0;
    }

  LengthCalculator length(alpha, Nlam, lam, rho);

  // The body from our implementation of the "regula falsi", or false position method.
  for(int ki=0; ki < 100; ++ki)
    {
      double f_a = length(nu_min + nu_a);
      double f_b = length(nu_min + nu_b);

      std::cout << ki << std::endl
                << nu_min << std::endl
                << nu_a << " " << f_a << std::endl
                << nu_b << " " << f_b <<std::endl <<std::endl;

      if (isinf(f_a) || isinf(f_b) ) return NULL;

      // When the root is not inside the (nu_min+nu_a, nu_min+nu_b)
      // interval, we pick a new interval to the left or to the right,
      // depending on the wether the function seems to be approaching
      // zero or not.
      if (((f_a > 0) && (f_b > 0)) ||
          ((f_a < 0) && (f_b < 0)))
        {
          if (fabs(f_a) > fabs(f_b))
            {
              nu_a = nu_b;
              nu_b = nu_b * 2.0;
            }
          else
            {
              nu_b = nu_a;
              nu_a = nu_a / 2.0;
            }
          continue;
        }

      // Assert that the function (apparently) descending, and that
      // the root is (or seems to be) within the interval.
      if ((f_a < 0) || (f_b > 0)) return NULL;
      
      // New point position, same formula from the secant method.
      double nu_new = nu_b - f_b * (nu_b - nu_a) / (f_b - f_a);
      // If the point is too close from one of the current extremes,
      // pick the median instead, perfoming an iteration of the
      // regular bisection method.
      if ((nu_new < nu_a * 1.1) || (nu_b / 1.1 < nu_new))
        {
          nu_new = (nu_a + nu_b) / 2.0 ;
        }
      
      // Calculate new ||delta(nu)||
      double f_new = length(nu_min + nu_new);
      
      // Stop loop if we reached a good result.
      if (fabs(f_new) < rho_tol)
        {
          return Py_BuildValue("d", nu_min + nu_new);
        }
      else
        {
          // Update the interval.
          if (f_new > 0)
            {
              nu_a = nu_new;
              f_a = f_new;
            }
          else
            {
              nu_b = nu_new;
              f_b = f_new;
            }
        }
    }
  PyErr_SetString(PyExc_RuntimeError, "Too many secant method iterations.");
  return NULL;
}

static PyMethodDef TrustBisectionMethods[] =
  {
    {"find_step_size", find_step_size, METH_VARARGS,
     "Help.\n"},
    {"calculate_distance", length_wrapper, METH_VARARGS,
     "Help.\n"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC

inittrust_bisection(void) {
  (void) Py_InitModule("trust_bisection", TrustBisectionMethods);
  import_array();
}
