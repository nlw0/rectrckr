# cython: boundscheck=False
# cython. wraparound=False
# cython: cdivision=True
# cython: profile=True
# file: corisco_aux.pyx

# Copyright 2011 Nicolau Leal Werneck, Anna Helena Reali Costa and
# Universidade de São Paulo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
DTYPE2 = np.float64
ctypedef np.float64_t DTYPE2_t

cimport cython



######################################################################
## These two procedures are used to fit lines to sets of points, so we
## can later perform estimation via reprojection error.
@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_transformed_coordinates(
    np.ndarray[DTYPE_t, ndim=2, mode="c"] tp not None,
    np.ndarray[DTYPE_t, ndim=2, mode="c"] op not None,
    np.ndarray[DTYPE_t, ndim=1, mode="c"] pp not None,
    np.ndarray[DTYPE_t, ndim=1, mode="c"] vp not None,
    float coef_dist):

    ## Check out size of stuff
    assert tp.shape[1] == 2
    assert op.shape[1] == 2
    assert pp.shape[0] == 2
    assert vp.shape[0] == 2

    cdef int k
    
    cdef float t_dist
    cdef float a0,a1

    for k in range(op.shape[0]):
        a0,a1 = op[k,0]-pp[0],op[k,1]-pp[1]
        t_dist = 1 + coef_dist * (a0*a0+a1*a1)
        tp[k,0],tp[k,1] = a0*t_dist-vp[0], a1*t_dist-vp[1]

## Finds null space from a list of 2D points
@cython.boundscheck(False)
@cython.wraparound(False)
def find_line_err(
    np.ndarray[DTYPE_t, ndim=2, mode="c"] p not None, int Np):
    ## Check out size of stuff
    assert p.shape[1] == 2

    cdef float M00 = 0.0 
    cdef float M01 = 0.0 
    cdef float M11 = 0.0 
   
    cdef float p0, p1

    cdef int k=0
    for k in range(Np):
        p0=p[k,0]
        p1=p[k,1]
        M00 += p0*p0
        M01 += p0*p1
        M11 += p1*p1

    cdef float n0 = 0.0
    cdef float n1 = 1.0

    ## Inverse magnitude for the calculation of each expected vp direction
    cdef float nf_0_

    n0,n1 = n0*M00 + n1*M01, n0*M01 + n1*M11
    nf_0_ = (n0**2+n1**2)**-.5
    n0,n1 = n0*nf_0_,n1*nf_0_
    n0,n1 = n0*M00 + n1*M01, n0*M01 + n1*M11
    nf_0_ = (n0**2+n1**2)**-.5
    n0,n1 = n0*nf_0_,n1*nf_0_
    n0,n1 = n0*M00 + n1*M01, n0*M01 + n1*M11
    nf_0_ = (n0**2+n1**2)**-.5
    ## Retun orthogonal vertor
    n0,n1 = -n1*nf_0_,n0*nf_0_

    ## Now calculate error
    cdef float errsum = 0.0
    cdef float err = 0.0
    for k in range(Np):
        err = n0 * p[k,0] + n1 * p[k,1]
        errsum += err if err > 0 else -err

    return errsum
######################################################################







######################################################################
## Defnition of various functions used for M-estimation.
##

## Squared error
cdef inline double sq_mlogL(double x, double isig):
    return x*x*isig*isig

cdef inline double sq_L_x_over_L(double x, double isig):
    return 2*isig*isig*x

cdef inline double sq_L_x_over_L__x(double x, double isig):
    return 2*isig*isig

## Absolute linear error
cdef inline double abs_mlogL(double x, double isig):
    return -isig*x if x<0 else isig*x

cdef inline double abs_L_x_over_L(double x, double isig):
    return -isig if x<0 else isig if x>0 else 0

cdef inline double abs_L_x_over_L__x(double x, double isig):
    return 0

## Huber loss function (quadratic / linear)
cdef inline double huber_mlogL(double x, double isig, double k=1.4):
    ## Get absolute value
    x = x * isig
    x = -x if x<0 else x
    return x*x if x <= k else 2 * k * x - k * k  

cdef inline double huber_L_x_over_L(double x, double isig, double k=1.4):
    x = x * isig
    if x < k and x > -k:
        return isig * 2 * x
    else:
        return 2*isig*k if x>=0 else -2*isig*k

cdef inline double huber_L_x_over_L__x(double x, double isig, double k=1.4):
    x = x * isig
    if x < k and x > -k:
        return 2*isig*isig
    else:
        return 0

## Tukey's bisquare (or biweigth) loss function.
@cython.profile(False)
cdef inline double tukey_mlogL(double x, double isig, double k=3.44):
    #return 1-(1-(isig*x/k)**2)**3 if fabs(x*isig) <= k else 1
    x = x*isig
    if x > k or x < -k:
        return 1
    else:
        x = x/k
        x = 1-x*x
        x = 1-x*x*x
    return x


cdef inline double tukey_L_x_over_L(double x, double isig, double k=3.44):
    return 6*isig**2*x*(k**2-isig**2*x**2)**2/(k**6) if fabs(x*isig) <= k else 0

cdef inline double tukey_L_x_over_L__x(double x, double isig, double k=3.44):
    ax = -x if x<0 else x
    return (6*isig**2*(k**4-6*k**2*isig**2*x**2+5*isig**4*x**4))/k**6 if fabs(x*isig) <= k else 0

## Redescending absolute linear error (i.e. abs with a treshold)
cdef inline double rabs_mlogL(double x, double isig, double k=0.5):
    cdef double ix = isig*x/k if x>0 else -isig*x/k
    return 1 if ix>1 else ix

cdef inline double rabs_L_x_over_L(double x, double isig, double k=0.5):
    cdef double ix = isig*x/k if x>0 else -isig*x/k
    return 0 if ix>1 else (-isig/k if x<0 else isig/k if x>0 else 0)

cdef inline double rabs_L_x_over_L__x(double x, double isig, double k=0.5):
    return 0

## Overall function that calls any of the others.
@cython.profile(False)
cdef inline double mlogL(double x, double * rho_param):
    cdef int func = int(rho_param[0])

    if func==0:
        return sq_mlogL(x,rho_param[1])
    elif func==1:
        return abs_mlogL(x,rho_param[1])
    elif func==2:
        return huber_mlogL(x,rho_param[1]) if rho_param[2]==0 \
               else huber_mlogL(x,rho_param[1],rho_param[2])
    elif func==3:
        return tukey_mlogL(x,rho_param[1]) if rho_param[2]==0 \
               else tukey_mlogL(x,rho_param[1],rho_param[2])
    elif func==4:
        return rabs_mlogL(x,rho_param[1]) if rho_param[2]==0 \
               else rabs_mlogL(x,rho_param[1],rho_param[2])
    else:
        raise Exception('Loss function not defined.')

cdef inline double L_x_over_L(double x, double * rho_param):
    cdef int func = int(rho_param[0])

    if func==0:
        return sq_L_x_over_L(x,rho_param[1])
    elif func==1:
        return abs_L_x_over_L(x,rho_param[1])
    elif func==2:
        return huber_L_x_over_L(x,rho_param[1]) if rho_param[2]==0 \
               else huber_L_x_over_L(x,rho_param[1],rho_param[2])
    elif func==3:
        return tukey_L_x_over_L(x,rho_param[1]) if rho_param[2]==0 \
               else tukey_L_x_over_L(x,rho_param[1],rho_param[2])
    elif func==4:
        return rabs_L_x_over_L(x,rho_param[1]) if rho_param[2]==0 \
               else rabs_L_x_over_L(x,rho_param[1],rho_param[2])
    else:
        raise Exception('Loss function not defined.')

cdef inline double L_x_over_L__x(double x, double * rho_param):
    cdef int func = int(rho_param[0])

    if func==0:
        return sq_L_x_over_L__x(x,rho_param[1])
    elif func==1:
        return abs_L_x_over_L__x(x,rho_param[1])
    elif func==2:
        return huber_L_x_over_L__x(x,rho_param[1]) if rho_param[2]==0 \
               else huber_L_x_over_L__x(x,rho_param[1],rho_param[2])
    elif func==3:
        return tukey_L_x_over_L__x(x,rho_param[1]) if rho_param[2]==0 \
               else tukey_L_x_over_L__x(x,rho_param[1],rho_param[2])
    elif func==4:
        return rabs_L_x_over_L__x(x,rho_param[1]) if rho_param[2]==0 \
               else rabs_L_x_over_L__x(x,rho_param[1],rho_param[2])
    else:
        raise Exception('Loss function not defined.')

## End defining the M-estimation loss functions.
######################################################################

######################################################################
## Lots of auxiliary function declarations
##

## Calculate likelihood values and other stuff.
cdef inline double Likelihood(double x, double * rho_param):
    return exp(-mlogL(x, rho_param))

# def py_lik(double x, double * rho_param):
#     return (Likelihood(x,isig,func, k), mlogL(x, isig, func, k),
#             L_x_over_L(x, isig, func, k), L_x_over_L__x(x, isig, func, k))

## 3D and 4D vector products.
cdef inline double dot_product(double x0,double x1,double x2,
                               double y0,double y1,double y2):
    return x0*y0+x1*y1+x2*y2
cdef inline double dot_product4(double x0,double x1,double x2,double x3,
                               double y0,double y1,double y2,double y3):
    return x0*y0+x1*y1+x2*y2+x3*y3

## Normalize a vector "in place", using approximate reverse of square root.
@cython.profile(False)
cdef inline void normalize(double* x, double* y):
    # cdef double fn = 1./sqrt(x[0]*x[0]+y[0]*y[0])
    # cdef double fn = (x[0]**2+y[0]**2)**-.5
    cdef float fni = x[0]*x[0]+y[0]*y[0]
    cdef float fn
    SSE_rsqrt_NR(&fn,&fni)

    x[0] = fn*x[0]
    y[0] = fn*y[0]

## Normalize a vector "in place", using approximate reverse of square root.
cdef inline float normalizef2d(float* x, float* y):
    # cdef double fn = 1./sqrt(x[0]*x[0]+y[0]*y[0])
    # cdef double fn = (x[0]**2+y[0]**2)**-.5
    cdef float fni = x[0]*x[0]+y[0]*y[0]
    cdef float fn
    SSE_rsqrt_NR(&fn,&fni)

    x[0] = fn*x[0]
    y[0] = fn*y[0]
    return 1.0/fn

cdef inline void normalize3d(double* x, double* y, double* z):
    # cdef double fn = 1./sqrt(x[0]*x[0]+y[0]*y[0])
    # cdef double fn = (x[0]**2+y[0]**2)**-.5
    cdef float fni = x[0]*x[0]+y[0]*y[0]+z[0]*z[0]
    cdef float fn
    SSE_rsqrt_NR(&fn,&fni)

    x[0] = fn*x[0]
    y[0] = fn*y[0]
    z[0] = fn*z[0]

## Return just the reverse (inverse, whatever) like from above
cdef inline double inverse_norm(double x, double y):
    # cdef double fn = 1./sqrt(x[0]*x[0]+y[0]*y[0])
    # cdef double fn = (x[0]**2+y[0]**2)**-.5
    cdef float fni = x*x+y*y
    cdef float fn
    SSE_rsqrt_NR(&fn,&fni)
    return fn

## Calculate the coordinates from a point after mapping with the
## Harris lens distortion model. Note that the inverse map is given by
## this exact same function, using -kappa as the distortion parameter.
cdef inline void harris_map(double* px, double* py, double kappa_in):
    # cdef double fn = 1.0 / sqrt(fabs(1 + kappa * (px[0] * px[0] + py[0] * py[0])))
    cdef double kappa = -2*1e-9 * kappa_in
    cdef float fni = fabs(1 + kappa * (px[0]*px[0] + py[0]*py[0]))
    cdef float fn
    SSE_rsqrt_NR(&fn,&fni)
    # fn = 1./sqrt(fn)

    px[0] = px[0] * fn
    py[0] = py[0] * fn


def py_harris_map(np.ndarray[DTYPE2_t, ndim=2, mode="c"] ed not None, double kappa):
    cdef int j
    cdef double x,y

    cdef np.ndarray out = np.zeros([ed.shape[0],2], dtype=DTYPE2)
    
    for j in range(ed.shape[0]):
        x = ed[j,0]
        y = ed[j,1]
        harris_map(&x, &y, kappa)
        out[j,0] = x
        out[j,1] = y
    return out

def py_harris_vec(np.ndarray[DTYPE2_t, ndim=2, mode="c"] ed not None, double kappa):
    cdef int j
    cdef double px,py,ux,uy

    cdef np.ndarray out = np.zeros([ed.shape[0],4], dtype=DTYPE2)
    
    for j in range(ed.shape[0]):
        px = ed[j,0]
        py = ed[j,1]
        ux = ed[j,2]
        uy = ed[j,3]
        harris_vec(&ux, &uy, px,py,kappa)
        out[j,0] = px
        out[j,1] = py
        out[j,2] = ux
        out[j,3] = uy
    return out
        



## Give just the jacobian coefficients for the distortion of a point
## over px py
cdef inline void harris_jacobian(double* Jxx, double* Jxy, double* Jyy,
                     double px, double py, double kappa_in):
    cdef double kappa = -2*1e-9 * kappa_in
    ## Jacobian terms (missing a scale factor)
    Jxx[0] = 1 + kappa * py * py
    Jxy[0] = -kappa * px * py
    Jyy[0] = 1 + kappa * px * px

## From the given undistorted coordinates px py and an original
## direction wx wy, find the distorted direction vx vy using the
## Jacobian and normalizing the answer.
cdef inline void harris_vec(double* vx, double* vy,
                     double px, double py, double kappa_in):
    cdef double kappa = -2*1e-9 * kappa_in
    ## Jacobian terms (missing a scale factor)
    cdef double Jxx = 1 + kappa * py * py
    cdef double Jxy = -kappa * px * py
    cdef double Jyy = 1 + kappa * px * px

    cdef double tmp
    tmp = Jxx * vx[0] + Jxy * vy[0]
    vy[0] = Jxy * vx[0] + Jyy * vy[0]
    vx[0] = tmp
    normalize(vx,vy)

## |r| to the -1.5 and -2.5 powers
cdef inline double inv_norm_1_5(double x, double y):
    cdef float fni = x*x+y*y
    fni *= fni*fni
    cdef float fn
    SSE_rsqrt_NR(&fn,&fni)
    return fn

cdef inline double inv_norm_2_5(double x, double y):
    cdef float fni = x*x+y*y
    fni *= fni*fni*fni*fni
    cdef float fn
    SSE_rsqrt_NR(&fn,&fni)
    return fn








######################################################################
## The general camera model procedure that calcultes the Jabocian of
## the projection from the direction calculated using the reverse
## projection of the given qx, qy point.

cdef inline void calculate_jacobian(
    double * dxdX, double * dxdY, double * dxdZ,
    double * dydX, double * dydY, double * dydZ,
    double qx, double qy,
    double* i_params):

    cdef double X,Y,Z
    cdef double tht, phi
    cdef double Jxx, Jxy, Jyy, kappa #For the Harris model
    
    cdef double vM

    cdef int which_camera_model = int(i_params[0])

    if which_camera_model==0:
    ## Simple perspective projection / aka gnomic projection, point projection
        X = (qx - i_params[2])
        Y = (qy - i_params[3])
        Z = i_params[1]

        ## cartesian -> polar
        # x = X * f / Z
        # y = Y * f / Z

        # dxdX[0] = f / Z
        # dydX[0] = 0
        # dxdY[0] = 0
        # dydY[0] = f / Z
        # dxdZ[0] = -X * f / (Z*Z)
        # dydZ[0] = -Y * f / (Z*Z)

        dxdX[0] = Z
        dxdY[0] = 0
        dxdZ[0] = -X
        dydX[0] = 0
        dydY[0] = Z
        dydZ[0] = -Y

    if which_camera_model==1:
    ## Polynomial
        X = (qx - i_params[2])
        Y = (qy - i_params[3])
        Z = i_params[1]

        ## cartesian -> polar
        # x = X * f / Z
        # y = Y * f / Z

        # dxdX[0] = f / Z
        # dydX[0] = 0
        # dxdY[0] = 0
        # dydY[0] = f / Z
        # dxdZ[0] = -X * f / (Z*Z)
        # dydZ[0] = -Y * f / (Z*Z)

        dxdX[0] = Z
        dydX[0] = 0
        dxdY[0] = 0
        dydY[0] = Z
        dxdZ[0] = -X
        dydZ[0] = -Y

    if which_camera_model==2:
    ## Harris model

        ## First we calculate the rectified image coordinates
        kappa = i_params[4]
        qx -= i_params[2]
        qy -= i_params[3]
        harris_map(&qx, &qy, -kappa)

        ## Now use regular point projection, find spatial
        ## direction. The jacobian of the point projection is given by
        ## these simple coefficients in the comments
        X = qx
        Y = qy
        Z = i_params[1]

        # dxdX[0] = Z
        # dxdY[0] = 0
        # dxdZ[0] = -X
        # dydX[0] = 0
        # dydY[0] = Z
        # dydZ[0] = -Y

        ## Get the distortion jacobian
        harris_jacobian(&Jxx, &Jxy, &Jyy, qx, qy, kappa)

        ## Now get the matrix multiplication J_harris x J_proj
        dxdX[0] = Jxx * Z
        dxdY[0] = Jxy * Z
        dxdZ[0] = Jxx * -X + Jxy * -Y
        dydX[0] = Jxy * Z
        dydY[0] = Jyy * Z
        dydZ[0] = Jxy * -X + Jyy * -Y


    elif which_camera_model==3:
    ## Polar azimuthal equidistant / aka Fisheye camera projection 
        X = (qx-i_params[2])
        Y = (qy-i_params[3])
        ## Normalize the XY vector, and keep the vector norm
        phi = sqrt(X*X+Y*Y)
        if phi > 0:
            X = X / phi
            Y = Y / phi
        ## Now find the angle by multiplying th enorm by the focal distance
        phi = phi / i_params[1]




        ## Now rotate the vector towards the Z axis
        # X = sin(phi) * X
        # Y = sin(phi) * Y
        # Z = cos(phi)

        #####
        ## Forward projection
        ## x = arccos(Z/sqrt(X*X+Y*Y+Z*Z))  X / sqrt(X*X+Y*Y)
        ## y = arccos(Z/sqrt(X*X+Y*Y+Z*Z))  Y / sqrt(X*X+Y*Y)


        vM = sin(phi) #sqrt(X*X+Y*Y)

        ## fiz meio na mão
        # dxdX[0] = X*X*Z * vM  + Y*Y*phi
        # dxdY[0] = X*Y*Z * vM  - X*Y*phi
        # dxdZ[0] = -X * (vM*vM*vM)
        # dydX[0] = X*Y*Z * vM  - X*Y*phi
        # dydY[0] = Y*Y*Z * vM  + X*X*phi
        # dydZ[0] = -Y * (vM*vM*vM)

        ## do Mathematica
        # dxdX[0] = X*X*Z/(vM*vM) - X*X * phi/(vM*vM*vM) + phi/vM
        # dxdY[0] = X*Y*Z/(vM*vM) - X*Y * phi/(vM*vM*vM)
        # dxdZ[0] = -X/vM
        # dydX[0] = X*Y*Z/(vM*vM) - X*Y * phi/(vM*vM*vM)
        # dydY[0] = Y*Y*Z/(vM*vM) - Y*Y * phi/(vM*vM*vM) + phi/vM
        # dydZ[0] = -Y/vM

        # dxdX[0] = X*X*Z/(vM) - X*X * phi/(vM*vM) + phi
        # dxdY[0] = X*Y*Z/(vM) - X*Y * phi/(vM*vM)
        # dxdZ[0] = -X
        # dydX[0] = X*Y*Z/(vM) - X*Y * phi/(vM*vM)
        # dydY[0] = Y*Y*Z/(vM) - Y*Y * phi/(vM*vM) + phi
        # dydZ[0] = -Y

        vM = sin(phi)
        dxdZ[0] = -vM * X
        dydZ[0] = -vM * Y
        vM = vM * cos(phi) - phi
        dxdX[0] = vM * X * X + phi
        dxdY[0] = vM * X * Y
        dydX[0] = dxdY[0]
        dydY[0] = vM * Y * Y + phi

    elif which_camera_model==4:
    ## Equirectangular / aka Geographic, lat-lon
        ## Angles from image coordinates
        tht = (qx-i_params[2]) / i_params[1]
        phi = (qy-i_params[3]) / i_params[1]

        ## Polar -> cartesian, normalized vector
        # vM = cos(phi)
        # X = vM * sin(tht)
        # Y = sin(phi)
        # Z = vM * cos(tht)

        ## cartesian -> polar
        # x = tht = arctan2(X,Z)
        # y = phi = arcsin(Y/sqrt(X*X+Y*Y+Z*Z))        

        # dxdX = -Z / (X*X+Z*Z)
        # dxdY = 0
        # dxdZ = X / (X*X+Z*Z)
        # dydX = -X Y / sqrt(1+Y*Y)
        # dydY = (1 -Y*Y) / sqrt(1 + Y * Y)
        # dydZ = -Z Y / sqrt(1+Y*Y)

        ## Multiply everything by sqrt(X*X+Z*Z) == sqrt(1-Y*Y) == cos(phi)
        # dxdX[0] = Z/vM
        # dxdY[0] = 0
        # dxdZ[0] = -X/vM
        # dydX[0] = -X * Y
        # dydY[0] = (1 - Y*Y)
        # dydZ[0] = -Z * Y

        
        vM = sin(phi)*cos(phi)
        X = sin(tht)
        Y = cos(phi)
        Z = cos(tht)
        dxdX[0] = Z
        dxdY[0] = 0
        dxdZ[0] = -X
        dydX[0] = -vM*X
        dydY[0] = Y*Y
        dydZ[0] = -vM*Z



#######################################################################
## Calculates the components of a quaternoin aligned to the v direction
def aligned_quaternion(np.ndarray[DTYPE2_t, ndim=1, mode="c"] n_a not None,
                       np.ndarray[DTYPE2_t, ndim=1, mode="c"] n_b not None):
    cdef double * a = <double*>n_a.data
    cdef double * b = <double*>n_b.data

    cdef double v[3], ee[3]
    cdef double q[4]

    ## Calculate a normalized vector orthogonal to n_a and n_b
    v[0] = a[1]*b[2] - a[2]*b[1]
    v[1] = a[2]*b[0] - a[0]*b[2]
    v[2] = a[0]*b[1] - a[1]*b[0]
    normalize3d(v, v+1,v+2)

    ee[0] = 0
    ee[1] = 0
    ee[2] = 0

    ## The largest component
    cdef int ll
    if (fabs(v[0]) > fabs(v[1]) and fabs(v[0]) > fabs(v[2])):
        ll = 0
    elif fabs(v[1]) > fabs(v[2]):
        ll = 1
    else:
        ll = 2

    ee[ll] = 1 if v[ll]>0 else -1    

    ## Calculate a normalize vector orthogonal to n_a and n_b
    q[1] = v[1]*ee[2] - v[2]*ee[1]
    q[2] = v[2]*ee[0] - v[0]*ee[2]
    q[3] = v[0]*ee[1] - v[1]*ee[0]
    q[0] = sqrt(1-(q[1]*q[1]+q[2]*q[2]+q[3]*q[3]))

    # R = q.sqrt()

    # if np.sign(v[ll]) > 0:
    #     if ll == 1:
    #         R = R*Quat(1,-1,-1,-1).normalize()
    #     if ll == 2:
    #         R = R*Quat(1,1,1,1).normalize()

    # if np.sign(v[ll]) < 0:
    #     if ll == 0:
    #         R = R*Quat(0,0,1,0).normalize()
    #     if ll == 1:
    #         R = R*Quat(1,1,-1,1).normalize()
    #     if ll == 2:
    #         R = R*Quat(1,-1,-1,1).normalize()
    # return R.inverse()
    return q[0]

##
######################################################################

######################################################################
## Calculation of the Jacobian matrices at each edgel, for storage,
## and calculation of the error function from these stored Jacobians.

## Calculate jacobians at each point
@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_all_jacobians(
                np.ndarray[DTYPE_t, ndim=2, mode="c"] edgels not None,
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] i_param not None):

    ## number of edgels
    cdef int Np = edgels.shape[0]

    cdef double* i_data = <double *>i_param.data

    ## Where to store all the jacobian values
    cdef np.ndarray jacobians = np.zeros([Np,6], dtype=DTYPE2)

    ## The vectors
    cdef double qx,qy

    ## The Jacobian coefficients
    cdef double dxdX, dxdY, dxdZ, dydX, dydY, dydZ

    ## Loop variable, index from the edgel
    cdef unsigned int N

    for N in range(Np):
        ## Read values for new edgel.
        qx = edgels[N, 0]
        qy = edgels[N, 1]

        calculate_jacobian(&dxdX, &dxdY, &dxdZ, &dydX, &dydY, &dydZ,
                            qx, qy, i_data)

        jacobians[N,0] = dxdX
        jacobians[N,1] = dxdY
        jacobians[N,2] = dxdZ
        jacobians[N,3] = dydX
        jacobians[N,4] = dydY
        jacobians[N,5] = dydZ

    return jacobians

## The calculation of the actual function value.
@cython.boundscheck(False)
@cython.wraparound(False)
def angle_error_with_jacobians(
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] q not None,
                np.ndarray[DTYPE_t, ndim=2, mode="c"] edgels not None,
                np.ndarray[DTYPE2_t, ndim=2, mode="c"] jacobians not None,
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] rho_param not None):

    ## number of edgels
    cdef int Np = edgels.shape[0]

    assert q.shape[0] == 4

    cdef double* J_data = <double *>jacobians.data
    cdef float* ed_data = <float *>edgels.data
    cdef double* rho_data = <double *>rho_param.data


    cdef double a=q[0]
    cdef double b=q[1]
    cdef double c=q[2]
    cdef double d=q[3]
    cdef double r00 = (a*a+b*b-c*c-d*d)
    cdef double r10 = (2*b*c-2*a*d)
    cdef double r20 = (2*b*d+2*a*c)
    cdef double r01 = (2*b*c+2*a*d)
    cdef double r11 = (a*a-b*b+c*c-d*d)
    cdef double r21 = (2*c*d-2*a*b)
    cdef double r02 = (2*b*d-2*a*c)
    cdef double r12 = (2*c*d+2*a*b)
    cdef double r22 = (a*a-b*b-c*c+d*d)

    ## Where the sum will be accumulated
    cdef double thesum = 0.0

    ## The vectors
    cdef double qx,qy,ux,uy,vx,vy

    ## Inverse magnitude for the calculation of each expected vp direction
    cdef double rx, ry, rz
    
    ## Value to be added at each iteration, calculated for each edgel
    cdef double vec_prod, err, minerr, lab_vec_prod=0
    cdef double dxdX, dxdY, dxdZ, dydX, dydY, dydZ
    cdef double X,Y,Z

    ## Loop variable, index from the edgel
    cdef unsigned int N, lab, lab_sel=0

    for N in range(Np):
        minerr=1.0e100
        ## Read values for new edgel.
        qx = ed_data[N * 4 + 0]
        qy = ed_data[N * 4 + 1]
        ux = ed_data[N * 4 + 2]
        uy = ed_data[N * 4 + 3]

        ## Read the Jacbian coefficients.
        dxdX = J_data[N * 6 + 0]
        dxdY = J_data[N * 6 + 1]
        dxdZ = J_data[N * 6 + 2]
        dydX = J_data[N * 6 + 3]
        dydY = J_data[N * 6 + 4]
        dydZ = J_data[N * 6 + 5]

        ##############################################################
        ## Calculate the predicted value and the error for each
        ## direction.
        for lab in range(3):
            ## Pick coordinates of current VP direction being tested
            rx = r00 if lab==0 else (r10 if lab==1 else r20)
            ry = r01 if lab==0 else (r11 if lab==1 else r21)
            rz = r02 if lab==0 else (r12 if lab==1 else r22)

            ## Find the predicted edgel direction.
            vx = dxdX * rx + dxdY * ry + dxdZ * rz
            vy = dydX * rx + dydY * ry + dydZ * rz
            normalize(&vx,&vy)
            ## Calculate angle error using vector product, then use
            ## the selected M-estimator to calculate the residue.
            vec_prod = ux*vx+uy*vy
            err = mlogL(vec_prod, rho_data)
            minerr = err if err < minerr else minerr
        ## Add the smallest error from this edgel to the total error.
        thesum += minerr

    return thesum
##
######################################################################


###############################################################################
##
@cython.boundscheck(False)
@cython.wraparound(False)
def classify_edgels(
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] q not None,
                np.ndarray[DTYPE_t, ndim=2, mode="c"] edgels not None,
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] i_param not None,
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] rho_param not None):

    ## number of edgels
    cdef int Np = edgels.shape[0]

    assert q.shape[0] == 4

    cdef double* i_data = <double *>i_param.data


    cdef double a=q[0]
    cdef double b=q[1]
    cdef double c=q[2]
    cdef double d=q[3]
    cdef double r00 = (a*a+b*b-c*c-d*d)
    cdef double r10 = (2*b*c-2*a*d)
    cdef double r20 = (2*b*d+2*a*c)
    cdef double r01 = (2*b*c+2*a*d)
    cdef double r11 = (a*a-b*b+c*c-d*d)
    cdef double r21 = (2*c*d-2*a*b)
    cdef double r02 = (2*b*d-2*a*c)
    cdef double r12 = (2*c*d+2*a*b)
    cdef double r22 = (a*a-b*b-c*c+d*d)

    ## Where the sum will be accumulated
    cdef double thesum = 0.0

    ## Where the classes will be stored.
    cdef np.ndarray class_out = np.zeros([Np,2], dtype=DTYPE2)
    
    ## The vectors
    cdef double qx,qy,ux,uy,vx,vy

    ## Inverse magnitude for the calculation of each expected vp direction
    cdef double rx, ry, rz
    
    ## Value to be added at each iteration, calculated for each edgel
    cdef double vec_prod, err, minerr, lab_vec_prod=0
    cdef double dxdX, dxdY, dxdZ, dydX, dydY, dydZ
    cdef double X,Y,Z

    ## Loop variable, index from the edgel
    cdef unsigned int N, lab, lab_sel=0

    for N in range(Np):
        minerr=1.0e100
        lab_sel = 0
        ## Read values for new edgel.
        qx = edgels[N, 0]
        qy = edgels[N, 1]
        ux = edgels[N, 2]
        uy = edgels[N, 3]

        calculate_jacobian(&dxdX, &dxdY, &dxdZ, &dydX, &dydY, &dydZ,
                            qx, qy, i_data)

        ##############################################################
        ## Calculate the predicted value and the error for each
        ## direction.
        for lab in range(3):
            ## Pick coordinates of current VP direction being tested
            rx = r00 if lab==0 else (r10 if lab==1 else r20)
            ry = r01 if lab==0 else (r11 if lab==1 else r21)
            rz = r02 if lab==0 else (r12 if lab==1 else r22)        

            ## Find the predicted edgel direction.
            vx = dxdX * rx + dxdY * ry + dxdZ * rz
            vy = dydX * rx + dydY * ry + dydZ * rz
            normalize(&vx,&vy)
            ## Calculate angle error using vector product, then use
            ## the selected M-estimator to calculate the residue.
            vec_prod = ux*vx+uy*vy
            err = mlogL(vec_prod, <double*>rho_param.data)
            if err < minerr:
                minerr = err
                lab_sel = lab

        ## Add the smallest error from this edgel to the total error.
        class_out[N,0] = lab_sel
        class_out[N,1] = minerr

    return class_out
##
###############################################################################


###############################################################################
## Auxiliary functions for the gradient calculation. Used in the three
## important procedures below.
cdef inline double rx_(double* q, int lab, int k):
    if lab==0:
        if   k==0: return  q[0]
        elif k==1: return  q[1]
        elif k==2: return -q[2]
        elif k==3: return -q[3]
    elif lab==1:
        if   k==0 : return -q[3]
        elif k==1 : return  q[2]
        elif k==2 : return  q[1]
        elif k==3 : return -q[0]
    elif lab==2:
        if   k==0 : return  q[2]
        elif k==1 : return  q[3]
        elif k==2 : return  q[0]
        elif k==3 : return  q[1]
    return 0
cdef inline double ry_(double* q, int lab, int k):
    if lab==0:
        if   k==0: return  q[3]
        elif k==1: return  q[2]
        elif k==2: return  q[1]
        elif k==3: return  q[0]
    elif lab==1:
        if   k==0 : return  q[0]
        elif k==1 : return -q[1]
        elif k==2 : return  q[2]
        elif k==3 : return -q[3]
    elif lab==2:
        if   k==0 : return -q[1]
        elif k==1 : return -q[0]
        elif k==2 : return  q[3]
        elif k==3 : return  q[2]
    return 0
cdef inline double rz_(double* q, int lab, int k):
    if lab==0:
        if   k==0: return -q[2]
        elif k==1: return  q[3]
        elif k==2: return -q[0]
        elif k==3: return  q[1]
    elif lab==1:
        if   k==0 : return  q[1]
        elif k==1 : return  q[0]
        elif k==2 : return  q[3]
        elif k==3 : return  q[2]
    elif lab==2:
        if   k==0 : return  q[0]
        elif k==1 : return -q[1]
        elif k==2 : return -q[2]
        elif k==3 : return  q[3]
    return 0

## Auxiliary functions for the Hessian calculation, these are the
## derivatives from the previous function.
cdef inline double rx__(int lab, int j, int k):
    if lab==0:
        if   j==0 and k==0: return  1
        elif j==1 and k==1: return  1
        elif j==2 and k==2: return -1
        elif j==3 and k==3: return -1
    elif lab==1:
        if   j==0 and k==3: return -1
        elif j==1 and k==2: return  1
        elif j==2 and k==1: return  1
        elif j==3 and k==0: return -1
    elif lab==2:
        if   j==0 and k==2: return  1
        elif j==1 and k==3: return  1
        elif j==2 and k==0: return  1
        elif j==3 and k==1: return  1
    return 0
cdef inline double ry__(int lab, int j, int k):
    if lab==0:
        if   j==0 and k==3: return  1
        elif j==1 and k==2: return  1
        elif j==2 and k==1: return  1
        elif j==3 and k==0: return  1
    elif lab==1:
        if   j==0 and k==0: return  1
        elif j==1 and k==1: return -1
        elif j==2 and k==2: return  1
        elif j==3 and k==3: return -1
    elif lab==2:
        if   j==0 and k==1: return -1
        elif j==1 and k==0: return -1
        elif j==2 and k==3: return  1
        elif j==3 and k==2: return  1
    return 0
cdef inline double rz__(int lab, int j, int k):
    if lab==0:
        if   j==0 and k==2: return -1
        elif j==1 and k==3: return  1
        elif j==2 and k==0: return -1
        elif j==3 and k==1: return  1
    elif lab==1:
        if   j==0 and k==1: return  1
        elif j==1 and k==0: return  1
        elif j==2 and k==3: return  1
        elif j==3 and k==2: return  1
    elif lab==2:
        if   j==0 and k==0: return  1
        elif j==1 and k==1: return -1
        elif j==2 and k==2: return -1
        elif j==3 and k==3: return  1
    return 0
##
###############################################################################


###############################################################################
## These next three functions are the current "official" ones used in
## Corisco, to calculate the objective function value, gradient and
## hessian for minimization by FilterSQP.  They re-calculate the
## Jacobians every time.
@cython.boundscheck(False)
@cython.wraparound(False)
def angle_error(
    np.ndarray[DTYPE2_t, ndim=1, mode="c"] q not None,
    np.ndarray[DTYPE_t, ndim=2, mode="c"] edgels not None,
    np.ndarray[DTYPE2_t, ndim=1, mode="c"] i_param not None,
    np.ndarray[DTYPE2_t, ndim=1, mode="c"] rho_param not None):

    ## number of edgels
    cdef int Np = edgels.shape[0]

    assert q.shape[0] == 4

    cdef double* i_data = <double *>i_param.data


    cdef double a=q[0]
    cdef double b=q[1]
    cdef double c=q[2]
    cdef double d=q[3]
    cdef double r00 = (a*a+b*b-c*c-d*d)
    cdef double r10 = (2*b*c-2*a*d)
    cdef double r20 = (2*b*d+2*a*c)
    cdef double r01 = (2*b*c+2*a*d)
    cdef double r11 = (a*a-b*b+c*c-d*d)
    cdef double r21 = (2*c*d-2*a*b)
    cdef double r02 = (2*b*d-2*a*c)
    cdef double r12 = (2*c*d+2*a*b)
    cdef double r22 = (a*a-b*b-c*c+d*d)

    ## Where the sum will be accumulated
    cdef double thesum = 0.0

    ## The vectors
    cdef double qx,qy,ux,uy,vx,vy

    ## Inverse magnitude for the calculation of each expected vp direction
    cdef double rx, ry, rz
    
    ## Value to be added at each iteration, calculated for each edgel
    cdef double vec_prod, err, minerr, lab_vec_prod=0
    cdef double dxdX, dxdY, dxdZ, dydX, dydY, dydZ
    cdef double X,Y,Z

    ## Loop variable, index from the edgel
    cdef unsigned int N, lab, lab_sel=0

    for N in range(Np):
        minerr=1.0e100
        ## Read values for new edgel.
        qx = edgels[N, 0]
        qy = edgels[N, 1]
        ux = edgels[N, 2]
        uy = edgels[N, 3]

        calculate_jacobian(&dxdX, &dxdY, &dxdZ, &dydX, &dydY, &dydZ,
                            qx, qy, i_data)

        ##############################################################
        ## Calculate the predicted value and the error for each
        ## direction.
        for lab in range(3):
            ## Pick coordinates of current VP direction being tested
            rx = r00 if lab==0 else (r10 if lab==1 else r20)
            ry = r01 if lab==0 else (r11 if lab==1 else r21)
            rz = r02 if lab==0 else (r12 if lab==1 else r22)        

            ## Find the predicted edgel direction.
            vx = dxdX * rx + dxdY * ry + dxdZ * rz
            vy = dydX * rx + dydY * ry + dydZ * rz
            normalize(&vx, &vy)
            ## Calculate angle error using vector product, then use
            ## the selected M-estimator to calculate the residue.
            vec_prod = ux * vx + uy * vy
            err = mlogL(vec_prod, <double*>rho_param.data)
            minerr = err if err < minerr else minerr
        ## Add the smallest error from this edgel to the total error.
        thesum += minerr

    return thesum


@cython.boundscheck(False)
@cython.wraparound(False)
def angle_error_gradient(
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] q not None,
                np.ndarray[DTYPE_t, ndim=2, mode="c"] edgels not None,
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] i_param not None,
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] rho_param not None):

    ## number of edgels
    cdef int Np = edgels.shape[0]

    assert q.shape[0] == 4

    cdef double a=q[0]
    cdef double b=q[1]
    cdef double c=q[2]
    cdef double d=q[3]
    cdef double* q_data = <double *> q.data
    cdef double* rho_data = <double *> rho_param.data
    cdef double* i_data = <double *>i_param.data

    cdef double r00 = (a*a+b*b-c*c-d*d)
    cdef double r10 = (2*b*c-2*a*d)
    cdef double r20 = (2*b*d+2*a*c)
    cdef double r01 = (2*b*c+2*a*d)
    cdef double r11 = (a*a-b*b+c*c-d*d)
    cdef double r21 = (2*c*d-2*a*b)
    cdef double r02 = (2*b*d-2*a*c)
    cdef double r12 = (2*c*d+2*a*b)
    cdef double r22 = (a*a-b*b-c*c+d*d)

    ## Where the sum will be accumulated
    cdef double thesum = 0.0

    ## The vectors
    cdef double qx,qy,ux,uy
    cdef double vx,vy,vdx, vdy, vdnx, vdny, vdx_sel, vdy_sel

    ## Derivatives of the calculated vector
    cdef double vx_[4], vy_[4], vdnx_[4], vdny_[4], vdx_[4], vdy_[4]

    ## Where the sum will be accumulated
    cdef np.ndarray grad_out = np.zeros([4], dtype=DTYPE2)
    cdef double grad[4]

    ## Inverse magnitude for the calculation of each expected vp direction
    cdef double rx, ry, rz

    ## Value to be added at each iteration, calculated for each edgel
    cdef double vec_prod, err, minerr, lab_vec_prod=0
    cdef double dxdX, dxdY, dxdZ, dydX, dydY, dydZ

    cdef double deno15

    ## Loop variable, index from the edgel
    cdef unsigned int N, lab, lab_sel=0, k
    
    for k in range(4):
        grad[k] = 0

    for N in range(Np):
        minerr=1.0e100
        ## Read values for new edgel.
        qx = edgels[N, 0]
        qy = edgels[N, 1]
        ux = edgels[N, 2]
        uy = edgels[N, 3]

        calculate_jacobian(&dxdX, &dxdY, &dxdZ, &dydX, &dydY, &dydZ,
                            qx, qy, i_data)

        ##############################################################
        ## Calculate the predicted value and the error for each
        ## direction.
        for lab in range(3):
            ## Pick coordinates of current VP direction being tested
            rx = r00 if lab==0 else (r10 if lab==1 else r20)
            ry = r01 if lab==0 else (r11 if lab==1 else r21)
            rz = r02 if lab==0 else (r12 if lab==1 else r22)        

            ## Find the predicted edgel direction.
            vdx = dxdX * rx + dxdY * ry + dxdZ * rz
            vdy = dydX * rx + dydY * ry + dydZ * rz

            ## Normalize the distorted direction
            vdnx = vdx
            vdny = vdy
            normalize(&vdnx,&vdny)
            ## Calculate angle error using vector product, then use
            ## the selected M-estimator to calculate the residue.
            vec_prod = ux * vdnx + uy * vdny
            err = mlogL(vec_prod, rho_data)
            if err < minerr:
                minerr = err
                lab_sel = lab
                lab_vec_prod = vec_prod
                vdx_sel = vdx
                vdy_sel = vdy
                
        lab = lab_sel
        vdx = vdx_sel
        vdy = vdy_sel

        deno15 = inv_norm_1_5(vdx,vdy)
        err_ = L_x_over_L(lab_vec_prod, rho_data)

        for k in range(4):
            vdx_[k] = 2 * (dxdX * rx_(q_data, lab, k) +
                           dxdY * ry_(q_data, lab, k) +
                           dxdZ * rz_(q_data, lab, k))
            vdy_[k] = 2 * (dydX * rx_(q_data, lab, k) +
                           dydY * ry_(q_data, lab, k) +
                           dydZ * rz_(q_data, lab, k))

            vdnx_[k] = +vdy * (vdy * vdx_[k] - vdx * vdy_[k]) * deno15
            vdny_[k] = -vdx * (vdy * vdx_[k] - vdx * vdy_[k]) * deno15
            grad[k] += err_ * (ux * vdnx_[k] + uy * vdny_[k])

    for k in range(4):
        grad_out[k] = grad[k]
    return grad_out


@cython.boundscheck(False)
@cython.wraparound(False)
def angle_error_hessian(
    np.ndarray[DTYPE2_t, ndim=1, mode="c"] q not None,
    np.ndarray[DTYPE_t, ndim=2, mode="c"] edgels not None,
    np.ndarray[DTYPE2_t, ndim=1, mode="c"] i_param not None,
    np.ndarray[DTYPE2_t, ndim=1, mode="c"] rho_param not None):

    ## number of edgels
    cdef int Np = edgels.shape[0]

    assert q.shape[0] == 4

    cdef double a=q[0]
    cdef double b=q[1]
    cdef double c=q[2]
    cdef double d=q[3]
    cdef double* q_data = <double *> q.data
    cdef double* rho_data = <double *> rho_param.data
    cdef double* i_data = <double *>i_param.data

    cdef double r00 = (a*a+b*b-c*c-d*d)
    cdef double r10 = (2*b*c-2*a*d)
    cdef double r20 = (2*b*d+2*a*c)
    cdef double r01 = (2*b*c+2*a*d)
    cdef double r11 = (a*a-b*b+c*c-d*d)
    cdef double r21 = (2*c*d-2*a*b)
    cdef double r02 = (2*b*d-2*a*c)
    cdef double r12 = (2*c*d+2*a*b)
    cdef double r22 = (a*a-b*b-c*c+d*d)

    ## Where the sum will be accumulated
    cdef double thesum = 0.0

    ## The vectors
    cdef double qx,qy,ux,uy
    cdef double vx,vy,vdx, vdy, vdnx, vdny, vdx_sel, vdy_sel

    ## Where the sum will be accumulated
    cdef np.ndarray hess_out = np.zeros([4,4], dtype=DTYPE2)
    cdef double grad[4], hess[4][4]

    ## Inverse magnitude for the calculation of each expected vp direction
    cdef double rx, ry, rz

    ## Derivatives of the calculated vector
    cdef double vx_[4], vy_[4], vdnx_[4], vdny_[4], vdx_[4], vdy_[4]
    cdef double vdnx__[4][4], vdny__[4][4], vdx__[4][4], vdy__[4][4]

    ## Value to be added at each iteration, calculated for each edgel
    cdef double vec_prod, err, minerr, lab_vec_prod=0
    cdef double dxdX, dxdY, dxdZ, dydX, dydY, dydZ
    cdef double X,Y,Z,phi

    cdef double deno15, deno25

    ## Loop variable, index from the edgel
    cdef unsigned int N, lab, lab_sel=0, j, k


    for k in range(4):
        grad[k] = 0
        for j in range(4):
            hess[j][k] = 0
    
    for N in range(Np):
        minerr=1.0e100
        ## Read values for new edgel.
        qx = edgels[N, 0]
        qy = edgels[N, 1]
        ux = edgels[N, 2]
        uy = edgels[N, 3]

        calculate_jacobian(&dxdX, &dxdY, &dxdZ, &dydX, &dydY, &dydZ,
                            qx, qy, i_data)

        ##############################################################
        ## Calculate the predicted value and the error for each
        ## direction.
        for lab in range(3):
            ## Pick coordinates of current VP direction being tested
            rx = r00 if lab==0 else (r10 if lab==1 else r20)
            ry = r01 if lab==0 else (r11 if lab==1 else r21)
            rz = r02 if lab==0 else (r12 if lab==1 else r22)        

            ## Find the predicted edgel direction.
            vdx = dxdX * rx + dxdY * ry + dxdZ * rz
            vdy = dydX * rx + dydY * ry + dydZ * rz

            ## Normalize the distorted direction
            vdnx = vdx
            vdny = vdy
            normalize(&vdnx,&vdny)
            ## Calculate angle error using vector product, then use
            ## the selected M-estimator to calculate the residue.
            vec_prod = ux*vdnx+uy*vdny
            err = mlogL(vec_prod, rho_data)
            if err < minerr:
                minerr = err
                lab_sel = lab
                lab_vec_prod = vec_prod
                vdx_sel = vdx
                vdy_sel = vdy

        lab = lab_sel
        vdx = vdx_sel
        vdy = vdy_sel

        ####################################################################
        ## Redo previous block. This is terrible must create a proper
        ## encapsulating function.

        ## Pick coordinates of current VP
        ## direction being tested
        rx = r00 if lab==0 else (r10 if lab==1 else r20)
        ry = r01 if lab==0 else (r11 if lab==1 else r21)
        rz = r02 if lab==0 else (r12 if lab==1 else r22)        

        ## Find the predicted edgel direction.
        vdx = dxdX * rx + dxdY * ry + dxdZ * rz
        vdy = dydX * rx + dydY * ry + dydZ * rz

        ## Normalize the distorted direction
        vdnx = vdx
        vdny = vdy
        normalize(&vdnx,&vdny)
        ## Calculate angle error using vector product, then use
        ## the selected M-estimator to calculate the residue.
        vec_prod = ux*vdnx+uy*vdny
        err = mlogL(vec_prod, rho_data)
        ###################################################################

        deno15 = inv_norm_1_5(vdx,vdy)
        deno25 = inv_norm_2_5(vdx,vdy)
        err_ = L_x_over_L(lab_vec_prod, rho_data)
        err__ = L_x_over_L__x(lab_vec_prod, rho_data)

        ## Gradient calculation
        for k in range(4):
            vdx_[k] = 2*(dxdX * rx_(q_data,lab,k)
                         + dxdY * ry_(q_data,lab,k)
                         + dxdZ * rz_(q_data,lab,k))
            vdy_[k] = 2*(dydX * rx_(q_data,lab,k)
                         + dydY * ry_(q_data,lab,k)
                         + dydZ * rz_(q_data,lab,k))

            vdnx_[k] = +vdy * (vdy*vdx_[k] - vdx*vdy_[k]) * deno15
            vdny_[k] = -vdx * (vdy*vdx_[k] - vdx*vdy_[k]) * deno15
            grad[k] += err_ * (ux*vdnx_[k] + uy*vdny_[k])

        ## Hessian calculation
        for j in range(4):
            for k in range(4):
                vdx__[j][k] = 2*(dxdX * rx__(lab,j,k) + dxdY * ry__(lab,j,k)
                                 + dxdZ * rz__(lab,j,k))
                vdy__[j][k] = 2*(dydX * rx__(lab,j,k) + dydY * ry__(lab,j,k)
                                 + dydZ * rz__(lab,j,k))

                vdnx__[j][k] = ((vdx*vdx * vdy * (2 * vdx_[j] * vdy_[k]
                                                + 2 * vdx_[k] * vdy_[j]
                                                + vdx__[j][k] * vdy)
                                -vdx * vdy*vdy * (3 * vdx_[k] * vdx_[j]
                                                 -2 * vdy_[k] * vdy_[j]
                                                 + vdy * vdy__[j][k])
                                +vdy*vdy*vdy * (vdx_[j] * (-vdy_[k])
                                           -vdx_[k] * vdy_[j]
                                           +vdx__[j][k] * vdy)
                                +vdx*vdx*vdx * (-(vdy_[k] * vdy_[j]
                                             +vdy * vdy__[j][k]))
                                )) * deno25

                vdny__[j][k] = ((vdy*vdy*vdx*(2*vdy_[j]*vdx_[k]
                                            + 2 * vdy_[k] * vdx_[j]
                                            + vdy__[j][k] * vdx)
                                -vdy * vdx*vdx * (3 * vdy_[k] * vdy_[j]
                                                 -2 * vdx_[k] * vdx_[j]
                                                 + vdx * vdx__[j][k])
                                +vdx*vdx*vdx * (vdy_[j] * (-vdx_[k])
                                           -vdy_[k] * vdx_[j]
                                           +vdy__[j][k] * vdx)
                                +vdy*vdy*vdy * (-(vdx_[k] * vdx_[j]
                                             +vdx * vdx__[j][k]))
                                )) * deno25

                hess[j][k] += err__ \
                           * (ux*vdnx_[k] + uy*vdny_[k]) \
                           * (ux*vdnx_[j] + uy*vdny_[j]) \
                           + err_ * (ux*vdnx__[j][k] + uy*vdny__[j][k])

    for j in range(4):
        for k in range(4):
            hess_out[j,k] = hess[j][k]

    return hess_out
##
######################################################################


######################################################################
## Get the projected direction from the environment axes at the given
## camera orientation for multiple points. This is intended to plot
## the solution.
@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_vdirs(
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] q not None,
                np.ndarray[DTYPE_t, ndim=2, mode="c"] edgels not None,
                np.ndarray[DTYPE2_t, ndim=1, mode="c"] i_param not None):

    ## number of edgels
    cdef int Np = edgels.shape[0]

    assert q.shape[0] == 4

    cdef double a=q[0]
    cdef double b=q[1]
    cdef double c=q[2]
    cdef double d=q[3]
    cdef double r00 = (a*a+b*b-c*c-d*d)
    cdef double r10 = (2*b*c-2*a*d)
    cdef double r20 = (2*b*d+2*a*c)
    cdef double r01 = (2*b*c+2*a*d)
    cdef double r11 = (a*a-b*b+c*c-d*d)
    cdef double r21 = (2*c*d-2*a*b)
    cdef double r02 = (2*b*d-2*a*c)
    cdef double r12 = (2*c*d+2*a*b)
    cdef double r22 = (a*a-b*b-c*c+d*d)

    ## Where the output vectors will be stored
    cdef np.ndarray out = np.zeros([3,Np,2], dtype=DTYPE2)

    ## The vectors
    cdef double qx,qy,px,py,vx,vy

    ## Inverse magnitude for the calculation of each expected vp direction
    cdef double rx, ry, rz
    
    ## Value to be added at each iteration, calculated for each edgel
    cdef double dxdX, dxdY, dxdZ, dydX, dydY, dydZ

    ## Loop variable, index from the edgel
    cdef unsigned int N, lab, lab_sel=0

    for N in range(Np):
        minerr=1.0e100
        ## Read values for new edgel.
        qx = edgels[N, 0]
        qy = edgels[N, 1]


        calculate_jacobian(&dxdX, &dxdY, &dxdZ, &dydX, &dydY, &dydZ,
                            qx, qy, <double *>i_param.data)

        ##############################################################
        ## Calculate the predicted value and the error for each
        ## direction.
        for lab in range(3):
            ## Pick coordinates of current VP direction being tested
            rx = r00 if lab==0 else (r10 if lab==1 else r20)
            ry = r01 if lab==0 else (r11 if lab==1 else r21)
            rz = r02 if lab==0 else (r12 if lab==1 else r22)        

            vx = dxdX * rx + dxdY * ry + dxdZ * rz
            vy = dydX * rx + dydY * ry + dydZ * rz
            normalize(&vx,&vy)

            out[lab,N,0]=vx
            out[lab,N,1]=vy

    return out
##
###############################################################################


###############################################################################
## Calculate the normal vectors of the interpretation planes from a
## list of edgels, using the harris model for the lens distortion.
@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_normals(
    np.ndarray[DTYPE_t, ndim=2, mode="c"] edgels not None,
    np.ndarray[DTYPE2_t, ndim=1, mode="c"] i_param not None):

    ## number of edgels
    cdef int Np = edgels.shape[0]

    ## Where the residues will be stored
    cdef np.ndarray normals = np.zeros([Np,3], dtype=DTYPE2)

    ## The vectors
    cdef double qx,qy,ux,uy,px,py,vx,vy,nx,ny,nz

    cdef double dxdX, dxdY, dxdZ, dydX, dydY, dydZ
    cdef double X,Y,Z

    ## Normalization factor
    cdef double nf

    cdef double* i_data = <double *>i_param.data

    ## Loop variable, index from the edgel
    cdef unsigned int N
    
    for N in range(Np):
        minerr=1.0e100
        ## Read values for new edgel.
        qx = edgels[N, 0]
        qy = edgels[N, 1]
        ux = edgels[N, 2]
        uy = edgels[N, 3]

        ## Find the Jacobian
        calculate_jacobian(&dxdX, &dxdY, &dxdZ, &dydX, &dydY, &dydZ,
                            qx, qy, i_data)

        ### WARNING: Still would like to quadruple-check this.
        nx = ux * dxdX + uy * dydX
        ny = ux * dxdY + uy * dydY
        nz = ux * dxdZ + uy * dydZ
        normalize3d(&nx,&ny,&nz)
        normals[N,0] = nx
        normals[N,1] = ny
        normals[N,2] = nz
        
    return normals
##
###############################################################################


###############################################################################
## Edgel extractor
##
cdef inline int sign(double x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

cdef inline float mysqrt(float x):
    # return sqrt(xx)
    cdef float y
    SSE_rsqrt_NR(&y,&x)
    return x*y

cdef inline float myrsqrt(float x):
    # return sqrt(xx)
    cdef float y
    SSE_rsqrt_NR(&y,&x)
    return y

## Auxiliary functions that fetch the gradient values, and also
## Laplacean in the Zernike moments case. "Normalized" means the sign of
## the gradient is positive in the direction we are analyzing (x==1,
## y==0)
cdef inline void get_normalized_gradient(int direction, int Ginc, int j, int k,
                                         float* gg, float * gradx, float* grady):
    cdef float aux = 0
    if direction == 1:
        aux = gradx[3*k+Ginc*j]
        if aux > 0:
            gg[0] = aux
            gg[1] = grady[3*k+Ginc*j]
        else:
            gg[0] = -aux
            gg[1] = -grady[3*k+Ginc*j]
        aux = gradx[3*k+Ginc*j+1]
        if aux > 0:
            gg[0] += aux
            gg[1] += grady[3*k+Ginc*j+1]
        else:
            gg[0] += -aux
            gg[1] += -grady[3*k+Ginc*j+1]
        aux = gradx[3*k+Ginc*j+2]
        if aux > 0:
            gg[0] += aux
            gg[1] += grady[3*k+Ginc*j+2]
        else:
            gg[0] += -aux
            gg[1] += -grady[3*k+Ginc*j+2]
    else:
        aux = grady[3*k+Ginc*j]
        if aux > 0:
            gg[1] = aux
            gg[0] = gradx[3*k+Ginc*j]
        else:
            gg[1] = -aux
            gg[0] = -gradx[3*k+Ginc*j]
        aux = grady[3*k+Ginc*j+1]
        if aux > 0:
            gg[1] += aux
            gg[0] += gradx[3*k+Ginc*j+1]
        else:
            gg[1] += -aux
            gg[0] += -gradx[3*k+Ginc*j+1]
        aux = grady[3*k+Ginc*j+2]
        if aux > 0:
            gg[1] += aux
            gg[0] += gradx[3*k+Ginc*j+2]
        else:
            gg[1] += -aux
            gg[0] += -gradx[3*k+Ginc*j+2]

## Copy the values of a 2D vector.
cdef inline void copy_2D(float*a, float*b):
    a[0]=b[0]
    a[1]=b[1]

## Return the norm of a vector
cdef inline float norm_2D(float* x):
    cdef float fni = x[0]*x[0]+x[1]*x[1]
    cdef float fn
    SSE_rsqrt_NR(&fn,&fni)
    return fni * fn

@cython.boundscheck(False)
@cython.wraparound(False)
def edgel_extractor(
    int gstep, float glim,
    np.ndarray[DTYPE_t, ndim=3, mode="c"] gradx not None,
    np.ndarray[DTYPE_t, ndim=3, mode="c"] grady not None
    ):

    Nrows = gradx.shape[0]
    Ncols = gradx.shape[1]

    cdef int Ned = 0, Ned_max = 100000
    cdef np.ndarray edgels = np.zeros([Ned_max,4], dtype=DTYPE)

    cdef int direction

    cdef int j,k

    cdef int jini
    cdef int jend
    cdef int kini
    cdef int kend

    cdef float ga[2], gb[2], gc[2], gg[2]
    cdef float na, nb, nc

    cdef float sub

    cdef int Ginc = 3*gradx.shape[1]

    ## First the lines
    jini = 5//2+(gradx.shape[0]//2-5//2)%gstep
    jend = gradx.shape[0]-5//2
    j = jini
    while j < jend:
        k=4

        get_normalized_gradient(1,Ginc,j,k,gb,<float*>gradx.data,<float*>grady.data)
        nb = norm_2D(gb)
        get_normalized_gradient(1,Ginc,j,k+1,gc,<float*>gradx.data,<float*>grady.data)
        nc = norm_2D(gc)

        for k in range(5, gradx.shape[1]-5):
            copy_2D(ga,gb)
            na=nb
            copy_2D(gb,gc)
            nb=nc
            get_normalized_gradient(1,Ginc,j,k+1,gc,<float*>gradx.data,<float*>grady.data)
            nc = norm_2D(gc)

            ## Test direction of the gradient, if it's too steep move on.
            if gb[0] < 0.9*fabs(gb[1]):
                continue

            ## Test if this is a peak, and a strong one. If not, move on.
            if not (nb>=na and nb>nc and nb > glim):
                continue
            sub = 0.5 * (na - nc) / (nc + na - 2 * nb)
            (<float*>edgels.data)[4*Ned] = k+sub
            (<float*>edgels.data)[4*Ned+1] = j
            (<float*>edgels.data)[4*Ned+2] = gb[0]/nb
            (<float*>edgels.data)[4*Ned+3] = gb[1]/nb
            Ned += 1
            if Ned == Ned_max:
                    break
        if Ned == Ned_max:
            break
        j += gstep

    kini = 5//2+(gradx.shape[1]//2-5//2)%gstep
    kend = gradx.shape[1]-5//2
    k = kini
    while k < kend:
        j=4

        get_normalized_gradient(0,Ginc,j,k,gb,<float*>gradx.data,<float*>grady.data)
        nb = norm_2D(gb)
        get_normalized_gradient(0,Ginc,j+1,k,gc,<float*>gradx.data,<float*>grady.data)
        nc = norm_2D(gc)

        for j in range(5, gradx.shape[0]-5):
            copy_2D(ga,gb)
            na=nb
            copy_2D(gb,gc)
            nb=nc
            get_normalized_gradient(0,Ginc,j+1,k,gc,<float*>gradx.data,<float*>grady.data)
            nc = norm_2D(gc)

            ## Test direction of the gradient, if it's too steep move on.
            if gb[1] < 0.9*fabs(gb[0]):
                continue

            ## Test if this is a peak, and a strong one. If not, move on.
            if not (nb>=na and nb>nc and nb > glim):
                continue
            sub = 0.5 * (na - nc) / (nc + na - 2 * nb)
            (<float*>edgels.data)[4*Ned] = k
            (<float*>edgels.data)[4*Ned+1] = j+sub
            (<float*>edgels.data)[4*Ned+2] = gb[0]/nb
            (<float*>edgels.data)[4*Ned+3] = gb[1]/nb
            Ned += 1
            if Ned == Ned_max:
                    break
        if Ned == Ned_max:
            break
        k += gstep
        
    return edgels[:Ned]
##
###############################################################################


def directional_error(
    np.ndarray[DTYPE_t, ndim=2, mode="c"] edgels not None,
    int e1, int e2):

    cdef float *ed = <float*>edgels.data
    cdef float v[2]

    v[0] = ed[4*e2+0] - ed[4*e1+0]
    v[1] = ed[4*e2+1] - ed[4*e1+1]

    cdef float norm = normalizef2d(v, v+1)

    cdef float sig_a = 0.5

    cdef float pa1 = (v[0]*ed[4*e1+2]+v[1]*ed[4*e1+3])
    cdef float pa2 = (v[0]*ed[4*e2+2]+v[1]*ed[4*e2+3])
    pa1 *= pa1
    pa2 *= pa2
    return pa1+pa2, norm


def interpretation_plane_error(
    np.ndarray[DTYPE2_t, ndim=2, mode="c"] normals not None,
    int e1, int e2):

    cdef double *ed = <double*>normals.data
    cdef float vx,vy,vz, norm, inorm

    vx = ed[3*e1+1] * ed[3*e2+2] - ed[3*e1+2] * ed[3*e2+1]
    vy = ed[3*e2+2] * ed[3*e1+0] - ed[3*e2+0] * ed[3*e1+2]
    vz = ed[3*e1+0] * ed[3*e2+1] - ed[3*e1+1] * ed[3*e2+0]
    norm = vx*vx+vy*vy+vz*vz

    SSE_rsqrt(&inorm, &norm)
    norm *= inorm

    return norm

cdef extern void SSE_rsqrt(float*, float*)
cdef extern void SSE_rsqrt_NR(float*, float*)
cdef extern void myvec_sumacc(float*, float*)
cdef extern void myvec_mulacc(float*, float*)
cdef extern void myvec_pos_lim(float*)
cdef extern void myvec_abs(float*)
cdef extern void myvec_copy(float*, float*)
cdef extern void myvec_rsqrt(float*)
cdef extern void myvec_rcp(float*)
cdef extern float the_big_MAP_loop(float*, float, float, float, 
                                   float*, float*, float*, 
                                   unsigned int, unsigned int, float)


cdef extern double atan2(double,double)
cdef extern double floor(double)
cdef extern double hypot(double,double)
cdef extern double sqrt(double)
cdef extern double log10(double)
cdef extern double exp(double)
cdef extern double fabs(double)

cdef extern double cos(double)
cdef extern double sin(double)
cdef extern double acos(double)

cdef extern from "stdio.h":
    int printf(char*,...)

