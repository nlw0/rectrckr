#!/usr/bin/python2.7
#coding: utf-8
import argparse
import code

import rectrckr
import rectrckr.lowlevel as lowlevel
from rectrckr.pose_string import PoseString

# import matplotlib as mpl
# mpl.use( "agg" )
from pylab import *
import code

from rectrckr.kalman_filter import KalmanFilter

import rectrckr.corisco as corisco_module
from rectrckr.corisco.quaternion import Quat, random_quaternion

def main():
    parser = argparse.ArgumentParser(description='Initial rectangle localization.')
    parser.add_argument('pose', type=PoseString)

    args = parser.parse_args()

    # line_segments = array([
    #         [0, -1, +1, -1, -1],
    #         [0, -1, +1, -1,  1],
    #         [0, -1, +1,  1, -1],
    #         [0, -1, +1,  1,  1],
    #         [1, -1, +1, -1, -1],
    #         [1, -1, +1, -1,  1],
    #         [1, -1, +1,  1, -1],
    #         [1, -1, +1,  1,  1],
    #         [2, -1, +1, -1, -1],
    #         [2, -1, +1, -1,  1],
    #         [2, -1, +1,  1, -1],
    #         [2, -1, +1,  1,  1],
    #         ])

    # landmarks = array([
    #         [-1,-1, 1],
    #         [-1,-1,-1],
    #         [-1, 1, 1],
    #         [-1, 1,-1],
    #         [ 1,-1, 1],
    #         [ 1,-1,-1],
    #         [ 1, 1, 1],
    #         [ 1, 1,-1],
    #         ])


    line_segments = array([
            [0, -sqrt(2), +sqrt(2), -1,  0],
            [0, -sqrt(2), +sqrt(2), -1,  0],
            [0, -sqrt(2), +sqrt(2),  1,  0],
            [0, -sqrt(2), +sqrt(2),  1,  0],
            [1, -1, +1,  0, -sqrt(2)],
            [1, -1, +1,  0,  sqrt(2)],
            [1, -1, +1,  0, -sqrt(2)],
            [1, -1, +1,  0,  sqrt(2)],
            ])

    landmarks = array([
            [-sqrt(2),-1, 0],
            [-sqrt(2), 1, 0],
            [ sqrt(2),-1, 0],
            [ sqrt(2), 1, 0],
            ])



    scene = rectrckr.SceneModel(line_segments, landmarks)
    camera = rectrckr.Camera()

    camera.set_position(args.pose[:3])
    camera.set_orientation(Quat(array(args.pose[3:7])))

    figure()
    ion()
    camera.plot_points(gca(), scene)
    camera.plot_edges(gca(), scene)
    plot(camera.center[0], camera.center[1], 'kx')
    grid()
    show()

    code.interact(local=locals())
