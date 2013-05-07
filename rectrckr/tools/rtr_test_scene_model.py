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
    parser.add_argument('test', type=str, choices=['plot', 'error'])
    parser.add_argument('--pose', type=PoseString, required=True)
    parser.add_argument('--model', type=str, choices=['cube', 'a4paper'], required=True)
    parser.add_argument('--edgels', type=str)


    args = parser.parse_args()

    scene = rectrckr.SceneModel(args.model)
    camera = rectrckr.Camera()

    camera.set_position(args.pose[:3])
    camera.set_orientation(Quat(array(args.pose[3:7])))

    if args.edgels is not None:
        edgels = loadtxt(args.edgels)





    figure()
    ion()
    camera.plot_edges(gca(), scene)
    camera.plot_points(gca(), scene)
    plot(camera.center[0], camera.center[1], 'kx')

    if args.edgels is not None:
        camera.plot_edgels(gca(), edgels)
        print "Predicted edgels"
        pe = camera.edgels_from_edges(scene)

        for k in range(4):
            print edgel_error(edgels[k], pe[k])

    grid()
    show()



    code.interact(local=locals())


def edgel_error(obs, pred):

    if abs(obs[0]) < abs(obs[1]):
        return (obs[0] - pred[0]) * -pred[2] / pred[3] + pred[1] - obs[1]
    else:
        return (obs[1] - pred[1]) * -pred[3] / pred[2] + pred[0] - obs[0]
