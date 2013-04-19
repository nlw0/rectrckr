#!/usr/bin/python2.7
#coding: utf-8
import argparse

import rectrckr
import rectrckr.lowlevel as lowlevel

from pylab import *
import code

def main():

    parser = argparse.ArgumentParser(description='Initial rectangle localization.')
    parser.add_argument('input', type=argparse.FileType('r'))
    parser.add_argument('img_index', type=int)
    parser.add_argument('px', type=int)
    parser.add_argument('py', type=int)
    args = parser.parse_args()

    ## Object that provides an interface to the images, and contains
    ## information such as the camera parameters
    ds = rectrckr.DataSource(args.input)

    img = ds.get_image(args.img_index)

    ## Initial position guess
    px, py = args.px, args.py

    ## Extract edgels
    edges = ds.extract_edgels(px,py)
    lx = edges[0,0] - edges[1,0]
    ly = edges[2,1] - edges[3,1]

    print px, py, lx, ly

    trckr = rectrckr.SimpleTracker(array([px,py,lx,ly,0,0,0,0]))

    Ns = 5
    ss = zeros((Ns,2))
    ss[0] = trckr.state[:2]
    for tt in range(1,Ns):
        edges = ds.extract_edgels(*trckr.state[:2])
        trckr.update(edges)
        ss[tt] = trckr.state[:2]



    ##
    ## Plot image and the extracted edges
    ion()
    figure(1)
    imshow(img, cmap=cm.gray)
    plot(px,py, 'bo')
    plot(edges[:,0], edges[:,1], 'ro')
    plot(ss[:,0], ss[:,1], 'go')

    code.interact(local=locals())


if __name__ == '__main__':
    from rectrckr.tools import rtr_init
    rtr_init.main()
