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
            lowlevel.gradient(img, x, py) for x in mgrid[2:img.shape[1]-2]])
    lhd = lhd[:,0]**2 + lhd[:,1]**2
    lvd = array([
            lowlevel.gradient(img, px, y) for y in mgrid[2:img.shape[0]-2]])
    lvd = lvd[:,0]**2 + lvd[:,1]**2

    ## Extract edgels
    edges = imgana.extract_edgels(px,py)

    medges = imgana.extract_moar_edgels(px,py)

    dirs = array([imgana.estimate_direction_at_point(int(medges[k,0]),
                                                     int(medges[k,1]))
                  for k in range(12)])

    ##
    ## Plot image and the extracted edges
    ion()
    figure(1)
    imshow(img, cmap=cm.gray)
    plot(px,py, 'bo')
    plot(medges[:,0], medges[:,1], 'ro')

    vv = array([-1.0,1.0])
    cc = array([.0,1.0])
    for k in range(12):
        
        plot(medges[k,0]+dirs[k,0]*cc,
             medges[k,1]+dirs[k,1]*cc,
             'b-')

        plot(medges[k,0]+dirs[k,1]*vv,
             medges[k,1]-dirs[k,0]*vv,
             'r-')
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
