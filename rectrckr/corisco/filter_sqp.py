#!/usr/bin/python
#coding: utf-8

# Copyright 2012 Nicolau Leal Werneck, Anna Helena Reali Costa and
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

## This module contains the procedure filterSQP, that implements the
## nonlinear programming technique of the same name, developed by
## Fletcher and Leyffer
## (http://www.springerlink.com/content/qqj37x00y79ygdl8/)
##
## The current implementation only supports a single constraint.


import numpy as np
from numpy import *
from numpy.linalg import *
import scipy.optimize

import cProfile

from filterSQP_filter import FletcherFilter

initial_guesses = array([[1,0,0,0],
                     [ 0.9239,  0.3827,  0.    ,  0.    ],
                     [ 0.9239,  0,0.3827,  0.],
                     [ 0.9239,  0,0,0.3827],
                     [0.8660, 0.288675, 0.288675, 0.288675],
                     [0.8660, 0.288675, 0.288675, -0.288675],
                     [0.8660, 0.288675, -0.288675, 0.288675],
                     [0.8660, 0.288675, -0.288675, -0.288675],
                     ])

def svd_eig(M):
    su,ss,sv = np.linalg.svd(np.dot(M.T,M))
    lam = np.diag(np.dot(np.dot(sv.T,M),sv.T))
    return lam, sv

def bisection_method(nu_min, nu_a, nu_b, myf, rho_tol):
    ki = 0
    while True:
        f_a = myf(nu_min+nu_a)
        f_b = myf(nu_min+nu_b)
        #print array([nu_min, nu_a,nu_b, f_a, f_b])
        assert not np.isinf(nu_a) and not np.isinf(nu_b)
        if f_a>0 and f_b>0 or f_a<0 and f_b<0:
            if np.abs(f_a)>np.abs(f_b):
                nu_a, nu_b = nu_b, nu_b*10
            else: #if np.abs(f_a)<np.abs(f_b):
                nu_a, nu_b = nu_a/10., nu_a
            continue
        assert f_a>0 and f_b<0

        ## New point position.
        nu_new = nu_b - f_b * (nu_b - nu_a) / (f_b - f_a)
        if nu_new < nu_a*1.1 or nu_b * .91 < nu_new:
            nu_new = (nu_a + nu_b)*.5
        ## Calculate new ||delta(nu)||.
        f_new = myf(nu_min+nu_new)

        ## Stop loop if we reached a good result.
        if np.abs(f_new) < rho_tol:
            return nu_min+nu_new
        else:
            # The two current points.
            if f_new > 0:
                nu_a, f_a = nu_new, f_new
            else:
                nu_b, f_b = nu_new, f_new
        ## Test number of iterations.
        ki += 1
        if ki > 100:
            raise Exception('Too many secant method iterations.')

def trust_region_step(G,g,rho,rho_tol):
    Nv = g.shape #number of dimensions

    ## Eigen decomposition of the matrix.
    lam, v = np.linalg.eig(G)
    ## lam, v = svd_eig(G) ## This is WRONG
    # print '==lam',lam
    # print '==v',v
    ln = lam.min()
    lni = np.nonzero(lam==ln)[0]
    lok = np.nonzero(1-(lam==ln))[0]

    ## Gradient projectin on the eigenvectors.
    alpha = np.dot(v.T, g)
    # print 'New QP problem:'
    # print 'lam:', lam
    # print 'rho:', rho
    # print lni, lok
    # print 'alpha:', alpha

    ## A plane. Just use the gradient...
    if (np.abs(lam)<1e-10).all():
        # print 'case 0'
        if np.linalg.norm(g) > 0:
            return -rho * g/np.linalg.norm(g)
        else:
            #print "joga pro santo"
            return rho * v[0]

    ## Test for case i -> Newton step (nu=0)
    if ln > 0:
        # print 'case 1'
        ## Matrix is positive definite.
        ## Test norm for the nu=0 case.
        beta = -alpha / lam
        nd = np.linalg.norm(beta)
        if nd <= rho + rho_tol:
        ## If norm is smaller than radius, this is "case i". The
        ## Newton step is feasible, so we just return it.
            delta = np.dot(v,beta)
            return delta
        ## If the Newton step is outside the feasible region we
        ## must look for a positive nu that satifies the radius
        ## limit.  Here we set the initial parameters for the
        ## root-finding procedure to find nu.
        nu_min = 0.
        nu_a = 0.
        nu_b = ln

    if ln <= 0:
        ## This may be case iii or ii. Test alpha.
        if (np.abs(alpha[lni]) < 1e-10).all():
            # print "case 3"
            ## This is case iii. First we test if the saddle point is
            ## outside the trust region, in which case we perofrm the
            ## normal calculation in the end. If not, we calculate the
            ## saddle point position, and return a vector stemming frmo it
            ## up to the trust region boundary.
            beta = np.zeros(Nv)
            if lok.shape[0]>0:
                beta[lok] = -alpha[lok] / (lam[lok] - ln)
            nd = np.linalg.norm(beta) ## this is h_bar from Fletcher's book, pg. 104
            if nd <= rho + rho_tol:
                # print "saddle in"
                ## If norm is smaller than radius, this is "case iii" with
                ## hk>=h_bar. We return the newton step plus a vector in the
                ## direction of the smallest eigenvalue...
                db = np.dot(v,beta) ## delta_bar from Fletcher's book
                mv = v[:,lni[0]]
                # print db, mv
                ## Solve to find the sum that touches the region border.
                # print '----', rho, nd
                delta = db + mv * np.sqrt((rho+rho_tol)**2 - nd**2)
                # print "Retornando um vetor maluco", delta
                return delta
        #     print "saddle out", nd, rho
        # print 'case 2'
        ## Here we are either in case ii, or in case iii with an
        ## infesible saddle-point. So we must look for a nu larger
        ## than -ln that satifies the radius limit.
        #nu_a = -ln * (1+1e-2) if ln<0 else 1e-3
        nu_min = -ln
        nu_a = nu_min * 1e-4 if nu_min > 0 else nu_min + 1e-3
        nu_b = 2*nu_min if nu_min > 0 else 1e-2



    ##################################################################
    ## The solution is in the surface of the sphere, so we must
    ## perform an iterative root-finding procedure (secant method) to
    ## find nu.
    nu_min = -ln if ln<0 else 0
    # nu_min = nu_a if ln != 0 else 0
    myf = lambda x: np.linalg.norm(-alpha / (lam + x)) - rho
    #myf = lambda x: (1+exp(-(norm(-alpha / (lam + x)) - rho)))**-1-.5

    ## Start the secant method loop.
    nu_opt = bisection_method(nu_min,nu_a,nu_b,myf,rho_tol)
    
    beta = -alpha / (lam + nu_opt)
    delta = np.dot(v,beta)
    return delta

def SQP_step(x, lam, rho, filt,
             cons_val, cons_grad, cons_hess,
             func_val, func_grad, func_hess, func_args):
    '''LM iteration constrained over the three-sphere'''
    ## Number of variables
    Nv = x.shape[0]

    ## Calculate the local gradient and Hessian matrix
    g = func_grad(x,*func_args)
    G = func_hess(x,*func_args)
    vcon = cons_val(x)
    gcon = cons_grad(x)
    Gcon = cons_hess(x)

    ## The coefficients from the constraint function linearized around
    ## x. Our constraint is the unit circle, and the simplest way to
    ## implement the constraint function is to use a paraboloid. Thus
    ## the linearization is simply x itself, and the null space is the
    ## hyperplane orthogonal to the current solution estimate.
    #A = r_[[ 2 * x ]]
    A = np.r_[[ gcon ]]

    ## Find Y amd Z, the pseudo-inverse and nullspace of the linearized
    ## constraint coefficients, A.
    Au,As,Avt = np.linalg.svd(A)
    Y = Au * As**-1 * Avt[0]
    Z = Avt[1:]

    ## The value of the constraint function at the point.
    # c_x = dot(x,x) - 1 #quadratic constraint
    c_x = vcon - 1 #conic constraint
    b = -c_x

    ## Find out the Hessian of the Lagrangian
    W = G - lam * Gcon # The constraint hessian (con from "cone")
    ## Calculate the reduced Hessian
    M = np.dot(np.dot(Z,W),Z.T)
    ## Calculate the reduced gradient
    m = np.dot(Z,g + np.dot(np.dot(G,Y.T)[:,0],b))
    
    ## Find the restricted step using a trust-region solver.
    y = trust_region_step(M,m,rho,rho*1e-1)
    if y.shape[0]==1:
        y = r_[[y]]
        delta_null = dot(y.T, Z)[0]
    else:
        delta_null = dot(y.T, Z)
    ## # delta = (dot(Y.T,b) + dot(Z.T,y))[:,0]
    # print '-----', y, delta_null
    nd = norm(delta_null)

    ##################################################################
    #### Perform second-order correction (SOC).
    ## Calculate new estimate for c, and consequently b
    c_hat = c_x + dot(gcon,delta_null)\
            + .5*dot(dot(delta_null,Gcon),delta_null)
    b = -c_hat
    ## Repeat previous steps with new b, and consequently m, to find
    ## the displacement in the cosntraint null space.
    m = dot(Z,g + dot(dot(G,Y.T)[:,0],b))
    y = trust_region_step(M,m,rho,rho*1e-1)
    if y.shape[0]==1:
        y = r_[[y]]
        delta_null = dot(y.T, Z)[0]
    else:
        delta_null = dot(y.T, Z)
    nd = norm(delta_null)
    delta_q = .5 * dot(dot(y,M),y) + dot(m, y)
    ####
    ##################################################################

    ## Now we calculate the delta in the direction of the "range" of
    ## the local approcximation to the constraint space, and then the
    ## total displacement.
    delta_range = dot(b, Y[0])
    delta = delta_null + delta_range
    ## Calculate the new value of lambda from the result.
    newlam = dot(Y,g)
    newx = x + delta

    filt_pass = True

    ## Use filter to test point, and update filter
    pt = array([func_val(newx,*func_args), np.abs(cons_val(newx)-1)])
    # print pt
    # print filt.values[filt.valid]
    # print filt.dominated(pt, delta_q)
    # print 70*'-'
    if not filt.dominated(pt, delta_q):
        filt_pass = True
        # rho = min(0.8, rho * 2.0)
        # rho = min(0.8, nd * 2.0)
        filt.add(pt, delta_q, newlam)
    else:
        filt_pass = False
        rho = nd *.25
        newlam = lam
        delta = 0

    return delta, newlam, rho, filt_pass, pt

def filterSQP(x0, lam0, rho0, funcs, args_f, delta_tol=1e-15):
    x = copy(x0)
    delta = zeros(2)
    rho = rho0
    lam = lam0
    val_c, grad_c, hess_c, val_f, grad_f, hess_f = funcs
    Niter = 100
    filt = FletcherFilter()

    Llam = [lam0]
    Lrho = [rho0]
    for it in range(0,Niter):
        delta, lam, rho, filt_pass, pt = SQP_step(x, lam, rho, filt,
                                                 val_c, grad_c, hess_c,
                                                 val_f, grad_f, hess_f, args_f)
        Llam.append(lam)
        Lrho.append(rho)
        if not filt_pass:
            continue
        x = x + delta        

        if np.abs(val_c(x)-1) < 1e-15 and np.abs(delta).sum() < delta_tol:
            break
    return x, val_f(x,*args_f), it, Llam, Lrho

def run_lots_of_cases(Ncase):
    for case in range(Ncase):
        print '====',case
        x0 = rand(3)*4-2
        run_SQP(x0, rho0, lam0, funcs, args_f)


def val_f(x,a,b):
    return x[0]**2 + 2*x[0]*x[1] + x[1]**2
def grad_f(x,a,b):
    return array([2*x[0] + 2*x[1], 2*x[0] + 2*x[1]])
def hess_f(x,a,b):
    return array([[2.0,2],[2,2]])

## Conic constraint
def val_c(x):
    return np.linalg.norm(x)
def grad_c(x):
    return x/np.linalg.norm(x)
def hess_c(x):
    nx = np.linalg.norm(x)
    return (nx**2 * np.identity(x.shape[0]) - np.outer(x,x)) / nx**3



## Test the FilterSQP algorithm with a simple function, a parabolic
## cylinder constrained to the unit circle.
if __name__ == '__main__':

    from pylab import *

    np.set_printoptions(precision=10)

    ion()

    Ndim = 2
    xini = array([1.1, 1.1])
    # xini = rand(Ndim)*4-2
    # xini = xini/norm(xini)
    # xini = array([ 1.4108939222, -0.5160438254, -1.8843145671, -0.614710015 ])
    # xini = sqrt(1./3) * array([1.01,.99,1.01])
    # xini = array([.5,-.75,2.])

    Niter = 200

    funcs = [val_f, grad_f, hess_f, val_c, grad_c, hess_c]

    # args_f = (np.array([[-.8,.8,0],[-.78,.82,0],[-.4,-.4,.8],[-.4,-.39,.81]]),)
    args_f = (np.array([[1.,0.1,0,0],[-1.,-0.1,0,0],
                        [0.1,1,0,0],[-0.1,-1,0,0],
                        [0,0.1,1,0],[0,-.1,-1,0]
                        ]),4)
    # args_f = (np.array([
    #             [1.,-2.1,0,0],
    #             [-.1,1,0,0],
    #             [0,-0.4,1,0],
    #             [1.,0.1,0,0],
    #             [0.1,1,0,0],
    #             [0,0.9,1,0],
    #             ]),4)
    # args_f = (np.array([
    #     [1,0,0,0],        [0,0.7,0,0],
    #     [0,0,.5,0],        [0,0,0,.2],                ]),4)


    rho0 = .3
    #rho0 = .1
    #rho0 = .01
    lam0 = 10.0

    if False:
        Ncase = 100
        logxx = zeros((Ncase,2))
        cProfile.runctx("run_lots_of_cases(Ncase)",globals(),locals(),'sqp.prof')

        myxtplykz()
    ###########################################################

    logx = zeros((Niter+1,Ndim))
    logfh = zeros((Niter+1,2))
    logr = zeros((Niter+1,1))
    logrho = zeros((Niter+1,1))
    logl = zeros((Niter+1,1))
    logfil = zeros((Niter+1,1))
    logpt = zeros((Niter+1,2))
    logdelta = zeros((Niter+1,Ndim))

    r = 1
    rho = rho0
    x = copy(xini)
    lam = lam0

    delta = zeros(2)

    filt = FletcherFilter()
    filt_pass = True


    for it in range(0,Niter):
        logx[it] = x
        logl[it] = lam
        logrho[it] = rho
        logr[it] = r
        logfil[it] = filt_pass

        logfh[it] = array([val_f(x,*args_f), np.abs(norm(x)-1)])

        delta, lam, rho, filt_pass,pt = SQP_step(x, lam, rho, filt,
                                              val_c, grad_c, hess_c,
                                              val_f, grad_f, hess_f, args_f)
        logdelta[it] = delta
        logpt[it] = pt
        if not filt_pass:
            continue
        x = x + delta        
        if np.abs(val_c(x)-1) < 1e-15 and np.abs(delta).sum() < 1e-15:
            break

    logx[it+1] = x
    logfh[it+1] = array([val_f(x,*args_f), np.abs(norm(x)-1)])
    logr[it+1] = r
    logrho[it+1] = rho
    logl[it+1] = lam
    logfil[it+1] = filt_pass
    logdelta[it+1] = delta

    logx = logx[:it+2]
    logfh = logfh[:it+2]
    logr = logr[:it+2]
    logrho = logrho[:it+2]
    logl = logl[:it+2]
    logfil = logfil[:it+2]
    logdelta = logdelta[:it+2]

    xx,yy = mgrid[-1.2:1.2:0.01,-1.2:1.2:0.01]
    xxyy = 1-xx.ravel()**2-yy.ravel()**2
    xxyy[xxyy<0]=0
    # ff = reshape(val_f(r_[[xx.ravel(),yy.ravel(), sqrt(xxyy) ]],*args_f), xx.shape)
    # ff2 = reshape(val_f(r_[[xx.ravel(),yy.ravel(), -sqrt(xxyy) ]],*args_f), xx.shape)
    ff = reshape(val_f(r_[[xx.ravel(),yy.ravel(), zeros(xxyy.shape[0]), zeros(xxyy.shape[0]) ]],*args_f), xx.shape)
    ff2 = reshape(val_f(r_[[xx.ravel(),yy.ravel(), zeros(xxyy.shape[0]), zeros(xxyy.shape[0]) ]],*args_f), xx.shape)



    


    figure(1)
    plot()

    figure(1, figsize=(12,6))
    title(u'Trajectory of a FilterSQP optimization')
    tt = mgrid[0:2*pi:0.01]
    ff = reshape(val_f(r_[[xx.ravel(),yy.ravel(), zeros(xxyy.shape[0]), zeros(xxyy.shape[0]) ]],*args_f), xx.shape)
    vv = mgrid[0.1:ff.max():0.1]
    #contour(xx,yy,ff,50, colors='r')
    contour(xx,yy,ff,vv)
    plot(logx[:,0], logx[:,1], '-o')
    plot(sin(tt), cos(tt), 'k-')
    plot(0,0,'sk')
    grid()
    axis('equal')
    axis([-1.12,1.12,-0.6,1.15])








    # figure(1, figsize=(12,6))
    # subplot(1,2,1)
    # title('Optimization track')
    # tt = mgrid[0:2*pi:0.01]
    # subplot(2,4,1)
    # ff = reshape(val_f(r_[[xx.ravel(),yy.ravel(), zeros(xxyy.shape[0]), zeros(xxyy.shape[0]) ]],*args_f), xx.shape)
    # contour(xx,yy,ff,50, colors='r')
    # plot(logx[:,0], logx[:,1], '-+')
    # plot(sin(tt), cos(tt), 'k-')
    # grid()
    # axis('equal')

    # subplot(2,4,2)
    # ff = reshape(val_f(r_[[zeros(xxyy.shape[0]), yy.ravel(), xx.ravel(), zeros(xxyy.shape[0]) ]],*args_f), xx.shape)
    # contour(xx,yy,ff,50, colors='r')
    # plot(logx[:,2], logx[:,1], '-+')
    # plot(sin(tt), cos(tt), 'k-')
    # grid()
    # axis('equal')

    # subplot(2,4,5)
    # ff = reshape(val_f(r_[[xx.ravel(), zeros(xxyy.shape[0]), zeros(xxyy.shape[0]), yy.ravel() ]],*args_f), xx.shape)
    # contour(xx,yy,ff,50, colors='r')
    # plot(logx[:,0], logx[:,3], '-+')
    # plot(sin(tt), cos(tt), 'k-')
    # grid()
    # axis('equal')

    # subplot(2,4,6)
    # ff = reshape(val_f(r_[[zeros(xxyy.shape[0]), xx.ravel(), zeros(xxyy.shape[0]), yy.ravel() ]],*args_f), xx.shape)
    # contour(xx,yy,ff,50, colors='r')
    # plot(logx[:,1], logx[:,3], '-+')
    # plot(sin(tt), cos(tt), 'k-')
    # grid()
    # axis('equal')



    # #sol = array([ 0.586919044 ,  0.5724029341,  0.5726088689])
    # sol = array([ 0.0,0,0,1])
    # if logx[-1,-1] <0:
    #     sol = -sol
    
    # axis(1.5*array([-1,1,-1,1]))
    
    # subplot(2,4,3)
    # title('Convergence')
    # semilogy(np.abs(logx[:,0]-sol[0]), 'b-+')
    # semilogy(np.abs(logx[:,1]-sol[1]), 'b-+')
    # semilogy(np.abs(sqrt((logx**2).sum(1))-1), 'r-+')
    # semilogy(np.abs(logfh[:,1]), 'g-+')
    # ylim(1e-16,1e2)
    # grid()
    # ylabel('Absolute error')
    # xlabel('Iteration')

    # subplot(2,4,4)
    # plot(logfh[:,0], logfh[:,1], '-+')
    # grid()


    # subplot(2,2,4)
    # plot(logr, 'b-+', label='r')
    # plot(logfil, 'ko')
    # #plot(logl, 'g-+', label='lambda')
    # #plot(sqrt(((logx[1:]-logx[:-1])**2).sum(1)), 'g-+', label='lambda')
    # grid()
    # ylim(-2,3)

    # twinx()
    # # plot(log10(logrho), 'r-+', label='rho')
    # semilogy(logrho, 'r-+', label='rho')
    # semilogy(sqrt(((logx[1:]-logx[:-1])**2).sum(1)), 'g-+', label='lambda')

    # # savefig('sqp_demo.png', dpi=100)
