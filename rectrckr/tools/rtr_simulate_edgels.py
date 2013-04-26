#!/usr/bin/python2.7
#coding: utf-8

import code

import rectrckr
import rectrckr.lowlevel as lowlevel

from pylab import *

from rectrckr.kalman_filter import KalmanFilter

import rectrckr.corisco as corisco
from rectrckr.corisco.quaternion import Quat
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

    ion()
    title(r'$\mathsf{Corisco}$ objective function for different loss functions')
    styles=[]
    for fp in fps:
        ff = array([
                co.target_function(Quat(sqrt(1.0 - tt ** 2), tt, 0.0, 0.0).q, fp)
                for tt in sin(angles)
                ])

        styles.append(plot(-angles*2*180/pi, ff, lw=2, color=ccc[ck])[0])
        ck+=1

    plot([-answer*2*180/pi, -answer*2*180/pi], [0.0,2.0], 'k--')
    grid()

    legend(styles, ('Absolute','Quadratic','Tukey bisquare'), loc='lower right')


    figure()
    title('rectrckr simulation')
    tt = -pi/12
    qini = Quat(sqrt(1.0 - tt ** 2), tt, 0.0, 0.0)

    co = corisco.Corisco(edgels)
    qopt = co.estimate_orientation(qini=qini)

    print qini
    print qopt
    print ori_correct, co.target_function(ori_correct.q),\
        co.gradient_function(ori_correct.q)

    ion()
    ax = subplot(1,1,1)
    axis('equal')
    co.plot_vps(ax)
    co.plot_edgels(ax)
    axis([0,640,480,0])

    code.interact(local=locals())

