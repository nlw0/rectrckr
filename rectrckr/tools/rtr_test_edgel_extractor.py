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

    ## Object that provides the images and contains information such
    ## as the camera parameters
    ds = rectrckr.DataSource(args.input)

    imgana = ds.get_image(args.img_index)
    img = imgana.img

    px, py = args.px, args.py

    ## Get data to plot the image levels and derivatives over the
    ## vertical and horizontal lines crossing (px,py)
    lh = img[py,:]
    lv = img[:,px]
    lhd = array([
            norm(array(lowlevel.gradient(img, x, py)))**2
            for x in mgrid[2:img.shape[1]-2]])
    lvd = array([
            norm(array(lowlevel.gradient(img, px, y)))**2
            for y in mgrid[2:img.shape[0]-2]])

    ## Extract edgels
    z = imgana.extract_edgels(px,py)
    edges = array([
            [z[0],py], 
            [z[1],py], 
            [px,z[2]], 
            [px,z[3]], 
            ])

    print edges
    dirs = array([imgana.estimate_direction_at_point(int(edges[k,0]),
                                                     int(edges[k,1]))
                  for k in range(4)])
    print dirs
    dirs = array([dd/norm(dd) for dd in dirs])
    
    ##
    ## Plot image and the extracted edges
    ion()
    figure(1)
    imshow(img, cmap=cm.gray)
    plot(px,py, 'bo')
    plot(edges[:,0], edges[:,1], 'ro')

    vv = array([-1.0,1.0])*20
    cc = array([.0,1.0])*20
    for k in range(4):
        
        plot(edges[k,0]+dirs[k,0]*cc,
             edges[k,1]+dirs[k,1]*cc,
             'b-')

        plot(edges[k,0]+dirs[k,1]*vv,
             edges[k,1]-dirs[k,0]*vv,
             'r-')
    axis([0,img.shape[1],img.shape[0],0])

    ## Plot image levels and derivatives
    figure(2)
    subplot(2,1,1)
    plot(lh, 'b-')
    twinx()
    plot(mgrid[2:img.shape[1]-2], lhd, 'r-+')
    subplot(2,1,2)
    plot(lv)
    twinx()
    plot(mgrid[2:img.shape[0]-2], lvd, 'r-+')
    code.interact(local=locals())
