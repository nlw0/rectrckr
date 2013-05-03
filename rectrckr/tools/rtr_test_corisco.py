#!/usr/bin/python2.7
#coding: utf-8

import argparse
import code

import rectrckr
import rectrckr.lowlevel as lowlevel

from pylab import *
import scipy.misc
import scipy.signal

from rectrckr.kalman_filter import KalmanFilter

import rectrckr.corisco as corisco_module
from rectrckr.corisco.quaternion import Quat, random_quaternion

class Scene:
    def __init__(self):
        # self.points = array([
        #         [-sqrt(2), 0, 0.0],
        #         [sqrt(2), 0, 0.0],
        #         [0, -1.0, 0.0],
        #         [0, 1.0, 0.0],
        #         ])

        self.points = array([
                [-1, 0,-1.0],
                [ 1, 0,-1],
                [-1, 0, 1],
                [ 1, 0, 1],
                [ 0,-1,-1],
                [ 0, 1,-1],
                [ 0,-1, 1],
                [ 0, 1, 1],
                [-1,-1, 0],
                [-1, 1, 0],
                [ 1,-1, 0],
                [ 1, 1, 0],
                ])

        self.lab = array([
                1,1,1,1,
                0,0,0,0,
                2,2,2,2
                ])

    def project(self, ori):
        fd = 500.0
        ppoint = array([320,240.0])

        M = ori.rot().T
        ppr = dot(self.points, M)
        ppr[:,2] += 6.0

        edgels = zeros((12,4))
        edgels[:,:2] = ppr[:,:2] * fd / c_[2*[ppr[:,2]]].T + ppoint
        
        r90 = array([[0,1],[-1,0.0]])
        for n in range(12):
            ll = self.lab[n]
            X,Y,Z = ppr[n]
            H = array([
                    [Z,0,-X],
                    [0,Z,-Y],
                    ])
            v = dot(r90,dot(H, M[ll]))
            edgels[n, 2:] = v/norm(v)

        return array(edgels, dtype=float32)


def corisco_function_test(scene, Niters):
    qori = random_quaternion().canonical()
    edgels = scene.project(qori)

    ## Estimate orientation using RANSAC.
    corisco = corisco_module.Corisco(edgels)
    random_search_result = corisco.silly_random_search(Niters)
    qest = random_search_result.canonical()

    return qori, qest


def corisco_ransac_test(scene, Niters):
    qori = random_quaternion().canonical()
    edgels = scene.project(qori)

    ## Estimate orientation using RANSAC.
    corisco = corisco_module.Corisco(edgels)
    random_search_result = corisco.ransac_search(Niters)[0]
    qest = random_search_result.canonical()

    return qori, qest


def corisco_gradient_test(scene, qori, Niters, qdelta):
    edgels = scene.project(qori)

    qtest = qori * qdelta
    print qtest
    ## Calculate the gradient.
    corisco = corisco_module.Corisco(edgels)
    grad1 = corisco.target_function_gradient(qtest.q)
    grad2 = corisco.target_function_gradient_numeric(qtest.q, dx=1e-4)

    return grad1, grad2


def main():
    parser = argparse.ArgumentParser(description='Test Corisco.')
    parser.add_argument('test', type=str)
    parser.add_argument('--multiple', type=int)
    parser.add_argument('--test_iters', type=int, default=2**12)

    args = parser.parse_args()

    if args.test == 'fvalue':
        if args.multiple is None:
            single_fvalue_test(args.test_iters)
        else:
            multiple_fvalue_test(args.test_iters, args.multiple)

    elif args.test == 'ransac':
        if args.multiple is None:
            single_ransac_test(args.test_iters)
        else:
            multiple_ransac_test(args.test_iters, args.multiple)
        
    elif args.test == 'gradient':
        if args.multiple is None:
            single_gradient_test()
        else:
            multiple_gradient_test(args.multiple)
        

def single_fvalue_test(Ntest_iters):
    scene = Scene()
    qori, qest = corisco_function_test(scene, Ntest_iters)
    print_oris_get_err(qori,qest)    
    plot_results(scene, qori, qest)


def single_ransac_test(Ntest_iters):
    scene = Scene()
    qori, qest = corisco_ransac_test(scene, Ntest_iters)
    print_oris_get_err(qori,qest)
    plot_results(scene, qori, qest)


def single_gradient_test(Ntest_iters):
    scene = Scene()

    dq = 2e-2
    qq = [
        Quat(1.0, 0, 0,0),
        Quat(1.0, dq,0,0),
        Quat(1.0,-dq,0,0),
        Quat(1.0,0, dq,0),
        Quat(1.0,0,-dq,0),
        Quat(1.0,0,0, dq),
        Quat(1.0,0,0,-dq),
        ]

    qori = random_quaternion().canonical()
    #qori = Quat(1.0,0,0,0).normalize()
    #qori = Quat(0.0,1,0,0).normalize()
    #qori = Quat(1.0,0.1,0.1,0).normalize()

    print qori
    print

    for qdelta in qq:
        g1, g2 = corisco_gradient_test(scene, qori, Ntest_iters, qdelta)
        print g1, dot((qori * qdelta).q, g1)
        print g2, dot((qori * qdelta).q, g2)
        print

    plot_results(scene, qori, qori * qdelta)

def multiple_fvalue_test(Ntest_iters, Ntests):
    scene = Scene()
    err = zeros(Ntests)
    for k in xrange(Ntests):
        qori, qest = corisco_function_test(scene, Ntest_iters)
        print 'Iteration: {: 3d}'.format(k)
        err[k] = print_oris_get_err(qori,qest)
    print 'Error mean and std', err.mean(), err.std()
    plot_err(err)


def multiple_ransac_test(Ntest_iters, Ntests):
    scene = Scene()
    err = zeros(Ntests)
    for k in xrange(Ntests):
        qori, qest = corisco_ransac_test(scene, Ntest_iters)
        print 'Iteration: {: 3d}'.format(k)
        err[k] = print_oris_get_err(qori,qest)
    print 'Error mean and std', err.mean(), err.std()
    plot_err(err)


def multiple_gradient_test(Ntests):
    scene = Scene()

    l1 = zeros((Ntests, 4))
    l2 = zeros((Ntests, 4))
    l3 = zeros((Ntests, 4))
    l4 = zeros((Ntests, 4))
    lang = zeros(Ntests)
    for k in xrange(Ntests):
        qori = random_quaternion()
        corisco = corisco_module.Corisco(scene.project(qori))
        qrand = random_quaternion()

        # Test gradient at solution. Should be pretty close to 0.
        l1[k] = corisco.target_function_gradient(qori.q)
        l2[k] = corisco.target_function_gradient_numeric(qori.q, dx=1e-6)

        # Test gradient at random point
        l3[k] = corisco.target_function_gradient(qrand.q)
        l4[k] = corisco.target_function_gradient_numeric(qrand.q, dx=1e-6)

        lang[k] = dot(qrand.q, l3[k])


    print 'Gradient projection on quaternion should be pretty low', mean(lang), std(lang)
    print 'Mean and covariances from gradients at the solution, arithmetic and '\
        'numeric. Should all be low.'
    print mean(l1,0)
    print cov(l1.T)
    print mean(l2,0)
    print cov(l2.T)

    print 'Covariances from gradients at generic points, and the difference '\
        'between the arithmetic and numeric.'
    print cov(l3.T)
    print cov(l4.T)
    print cov((l3-l4).T)


def print_oris_get_err(qori, qest):
    err = (qori / qest).canonical().angle()
    print 'Generated orientation:', qori
    print 'Estimated orientation:', qest
    print 'Angular error: {:.2f}'.format(err)
    return err


def plot_err(err):
    ion()
    figure()
    ax = subplot(1,1,1)
    title('Angle error distribution')
    plot(sort(err), mgrid[0.0:err.shape[0]]/err.shape[0], '-+')
    grid()

    code.interact(local=locals())


def plot_results(scene, qori, qest=None):
    ## Create objects again
    edgels = scene.project(qori)
    corisco = corisco_module.Corisco(edgels)

    ion()
    figure()
    ax = subplot(1,1,1)
    title('simulated data')
    axis('equal')
    corisco.plot_vps(ax, qori)
    corisco.plot_edgels(ax)
    axis([0,640,480,0])

    if qest is not None:
        figure()
        ax = subplot(1,1,1)
        title('orientation estimate')
        axis('equal')
        corisco.plot_vps(ax, qest)
        corisco.plot_edgels(ax)
        axis([0,640,480,0])

    code.interact(local=locals())
