#!/usr/bin/python2.7
#coding: utf-8

import code

import rectrckr
import rectrckr.lowlevel as lowlevel

from pylab import *
import scipy.misc

from rectrckr.kalman_filter import KalmanFilter

import rectrckr.corisco as corisco
from rectrckr.corisco.quaternion import Quat, random_quaternion
import rectrckr.corisco.corisco_aux as corisco_aux

class Camera:
    def __init__(self):
        self.points = array([
                [-sqrt(2), 0, 0.0],
                [sqrt(2), 0, 0.0],
                [0, -1.0, 0.0],
                [0, 1.0, 0.0],
                ])

        self.lab = array([1,1,0,0])

    def project(self, ori):

        fd = 500.0
        ppoint = array([320,240.0])

        M = ori.rot().T
        ppr = dot(self.points, M)
        ppr[:,2] += 6.0

        print ppr

        edgels = zeros((4,4))
        edgels[:,:2] = ppr[:,:2] * fd / c_[2*[ppr[:,2]]].T + ppoint
        
        r90 = array([[0,1],[-1,0.0]])
        for n in range(4):
            ll = self.lab[n]
            X,Y,Z = ppr[n]
            H = array([
                    [Z,0,-X],
                    [0,Z,-Y],
                    ])
            v = dot(r90,dot(H, M[ll]))
            edgels[n, 2:] = v/norm(v)

        return array(edgels, dtype=float32)


def plot_edgels(edgels, ax):
    vv = array([-1,1])*10.0
    for ee in edgels:
        ax.plot(ee[0]+vv*ee[2], ee[1]+vv*ee[3], 'r-', lw=2)
    axis('equal')
    axis([0,640,480,0.0])


def main():

    cc = Camera()

    answer = -pi/12
    #ori_correct = Quat(sqrt(1.0 - answer ** 2), answer, 0.0, 0.0)
    ori_correct = Quat(cos(answer), sin(answer), 0.0, 0.0)
    edgels = cc.project(ori_correct)

    co = corisco.Corisco(edgels)
    

    fps = array([
            [1.0,3.0,0],
            [2.0,3.0,0],
            [3.0,1.0,0.15],
            ])

    rc("mathtext",fontset='stixsans')

    Nang = 2000
    angles = .25*pi * mgrid[0.0:Nang + 1] / Nang - pi/4

    ccc=['#ea6949', '#51c373', '#a370ff', '#444444']
    ck=0

    qini = ori_correct * random_quaternion(0.2)

    co = corisco.Corisco(edgels)

    qopt = co.estimate_orientation(qini=qini)

    print qini, co.target_function_value(qini.q)
    print qopt, co.target_function_value(qopt.q)
    print ori_correct, co.target_function_value(ori_correct.q)

    ion()
    figure()
    ax = subplot(1,1,1)
    title('rectrckr simulation')
    axis('equal')
    co.plot_vps(ax, ori_correct)
    co.plot_edgels(ax)
    axis([0,640,480,0])

    figure()
    ax = subplot(1,1,1)
    title('Estimation initial state')
    axis('equal')
    co.plot_vps(ax, qini)
    co.plot_edgels(ax)
    axis([0,640,480,0])

    figure()
    ax = subplot(1,1,1)
    title('Estimation result')
    axis('equal')
    co.plot_vps(ax, qopt)
    co.plot_edgels(ax)
    axis([0,640,480,0])

    
    code.interact(local=locals())

