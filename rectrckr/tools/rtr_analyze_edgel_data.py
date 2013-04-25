#!/usr/bin/python2.7
#coding: utf-8
import matplotlib as mpl
mpl.use( "agg" )

import argparse
import code

import rectrckr
import rectrckr.lowlevel as lowlevel

from pylab import *
import code

from rectrckr.kalman_filter import KalmanFilter

import rectrckr.corisco as corisco
from rectrckr.corisco.quaternion import Quat


def main():

    parser = argparse.ArgumentParser(description='File containing edgel information, Nframes * Nedgels lines.')
    parser.add_argument('input', type=argparse.FileType('r'))
    parser.add_argument('log_edgel', type=argparse.FileType('r'))
    #parser.add_argument('frame', type=int)

    args = parser.parse_args()


    ds = rectrckr.DataSource(args.input)

    log_edgels = loadtxt(args.log_edgel)

    qopt = Quat(1.0,0,0,0)

    fig = figure(1)
    subplot(1,1,1)
    fig.tight_layout()
    ax = subplot(1,1,1)

    for k in range(0,772):
        edgels = array(log_edgels[k*12:(k+1)*12], dtype=float32)

        imgana = ds.get_image(k)
        img = imgana.img

        co = corisco.Corisco(edgels)
        qopt = co.estimate_orientation(qini=qopt)
        print k, qopt

        cla()
        imshow(img, cmap=cm.gray)
        axis('equal')
        co.plot_vps(ax)
        co.plot_edgels(ax)

        axis([0, 640, 480, 0])

        savefig(ds.get_output_path(k))

    code.interact(local=locals())
