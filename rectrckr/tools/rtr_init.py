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
    args = parser.parse_args()

    ds = rectrckr.DataSource(args.input)

    print ds
    img = ds.get_image(0)

    find 

    fil_dev = -array([-1,-2,0,2,1])/4.0

    cx,cy = ds.camera_parameters["center"]

    sx = int(cx)
    sy = int(cy)

    lh = img[240,:]
    lhd = convolve(lh, fil_dev)[2:-2]

    edges = zeros((4,2))
    for x in xrange(1,img.shape[1]-1):
        if lhd[x] < -10 and lhd[x] < lhd[x-1] and lhd[x] <= lhd[x+1]:
            edges[0] = (x, 240)
        if lhd[x] > 10 and lhd[x] > lhd[x-1] and lhd[x] >= lhd[x+1]:
            edges[1] = (x, 240)

    lv = img[:,320]
    lvd = convolve(lv, fil_dev)[2:-2]

    for y in xrange(1,img.shape[0]-1):
        if lvd[y] < -10 and lvd[y] < lvd[y-1] and lvd[y] <= lvd[y+1]:
            edges[2] = (320, y)
        if lvd[y] > 10 and lvd[y] > lvd[y-1] and lvd[y] >= lvd[y+1]:
            edges[3] = (320, y)
    

    print lowlevel.find_edges(img, sx, sy, 0)
    print lowlevel.find_edges(img, sx, sy, 1)
    print lowlevel.find_edges(img, sx, sy, 2)
    print lowlevel.find_edges(img, sx, sy, 3)


    ion()

    figure(1)
    imshow(img, cmap=cm.gray)
    plot(cx,cy, 'bo')
    plot(edges[:,0], edges[:,1], 'ro')


    figure(2)
    subplot(2,1,1)
    plot(lh, 'b-')
    twinx()
    plot(lhd, 'r-+')
    subplot(2,1,2)
    plot(img[:,320])
    twinx()
    plot(lvd, 'r-+')



    code.interact(local=locals())



if __name__ == '__main__':
    from rectrckr.tools import rtr_init
    rtr_init.main()
