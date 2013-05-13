#!/usr/bin/python2.7
#coding: utf-8
import argparse
import code

import rectrckr.lowlevel as lowlevel

# import matplotlib as mpl
# mpl.use( "agg" )
from pylab import *
import code

from rectrckr.kalman_filter import KalmanFilter

import rectrckr.corisco as corisco_module
from rectrckr.corisco.quaternion import Quat, random_quaternion

def mod(*pp):
    return array([tuple(pp)],
                 dtype=[('cm','=L8'),
                        ('cx','=f8'),
                        ('cy','=f8'),
                        ('fl','=f8'),
                        ('kappa','=f8')])

def main():
    parser = argparse.ArgumentParser(description='Test camera model, projection and stuff.')

    args = parser.parse_args()

    ## Just some points on a plane
    s = ascontiguousarray(array(c_[mgrid[-10:11:2,-10:11:2,0:1].reshape(3,-1).T], dtype=float64))

    ## Tilt the camera a bit, and make it point to the origin.
    ion()
    figure(1)
    theta = pi / 16
    t = array([0.0, 25.0 * tan(theta), -25.0])
    psi = array([cos(theta/2), sin(theta/2), 0, 0])

    cx, cy = 320.0, 240.0
    mods = [
        mod(1, cx, cy, 500.0, 0.0),
        mod(2, cx, cy, 500.0, 3e-4),
        mod(2, cx, cy, 500.0, -3e-4),
        ]
    plist = [array([lowlevel.p_wrapper(sk, t, psi, cmod) for sk in s]) for cmod in mods]

    plot_pp(plist)
    title('Camera model test - Harris distortion')
    axis([0,640,480,0])

    #####
    theta = pi / 16
    figure(2)
    t = array([0.0, tan(theta), -1.0]) * 5.0
    psi = array([cos(theta/2), sin(theta/2), 0, 0])
    cx, cy = 320.0, 240.0
    mods = [
        mod(3, cx, cy, 180.0, 0.0),
        ]
    plist = [array([lowlevel.p_wrapper(sk, t, psi, cmod) for sk in s]) for cmod in mods]
    plot_pp(plist)
    title('Camera model test - equidistant distortion')
    axis([0,640,480,0])

    #####
    figure(3, figsize=(8,4))
    title('Camera model test - equirectangular distortion')
    theta = pi / 16
    t = array([0.0, tan(theta), -1.0]) * 3.0
    psi = array([cos(theta/2), sin(theta/2), 0, 0])
    cx, cy = 0.0, 0.0
    mods = [
        mod(4, cx, cy, 1.0, 0.0),
        ]
    plist = [array([lowlevel.p_wrapper(sk, t, psi, cmod) for sk in s]) for cmod in mods]
    plot_pp(plist, cx, cy)
    axis([-pi,pi,pi/2,-pi/2])




    code.interact(local=locals())


def plot_pp(plist, cx=320.0, cy=240.0):

    for pp in plist:
        plot(pp[:,0], pp[:,1], '.')

    plot([cx-500,cx+500], [cy,cy], 'k:')
    plot([cx,cx], [cy-500,cy+500], 'k:')
    axis('equal')
