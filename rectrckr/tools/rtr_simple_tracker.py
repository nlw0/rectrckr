#!/usr/bin/python2.7
#coding: utf-8
import argparse
import code

import rectrckr
import rectrckr.lowlevel as lowlevel

from pylab import *
import code

from rectrckr.kalman_filter import KalmanFilter

def main():

    parser = argparse.ArgumentParser(description='Initial rectangle localization.')
    parser.add_argument('input', type=argparse.FileType('r'))
    parser.add_argument('img_index', type=int)
    parser.add_argument('px', type=int)
    parser.add_argument('py', type=int)
    args = parser.parse_args()

    ## Initial position guess
    px, py = args.px, args.py

    ## Object that provides an interface to the images, and contains
    ## information such as the camera parameters
    ds = rectrckr.DataSource(args.input)

    img = ds.get_image(args.img_index)
    ## Extract edgels
    edges = img.extract_edgels(px,py)
    lx = edges[1] - edges[0]
    ly = edges[3] - edges[2]

    print px, py, lx, ly


    kalman = KalmanFilter()
    kalman.state = array([px,py,lx,ly,0,0,0,0])

    kalman.Mobser = array([
            [1,0,-1,0,0,0,0,0],
            [1,0,1,0,0,0,0,0],
            [0,1,0,-1,0,0,0,0],
            [0,1,0,1,0,0,0,0],
            ])

    new_data = array([
            50,150,100,200.0
            ])

    dt = 10.0

    xout = zeros((773,8))
    for k in range(1,772,1):

        img = ds.get_image(k)
        px,py = kalman.state[:2]
        new_data = img.extract_edgels(px,py)
        

        print 70*'-'
        kalman.predict_state(dt)
        print 'pred st: ', kalman.state
        kalman.predict_observations()
        print 'pred obs:', kalman.z_hat
        print 'measured:', new_data
        merr = norm(kalman.z_hat - new_data)
        print 'measurement error:', merr

        kalman.update_from_observations(new_data)
        print 'updt st:', kalman.state

        # ## Log predicted state and outputs
        xout[k/5] = kalman.state



    ##
    ## Plot image and the extracted edges
        cla()
        imshow(copy(img.img), cmap=cm.gray)
        plot(new_data[:2], [py,py], 'k--')
        plot([px,px], new_data[2:4], 'k--')
        plot(new_data[:2], [py,py], 'ro')
        plot([px,px], new_data[2:4], 'ro')
        plot(px,py, 'bo')

        axis([0,img.img.shape[1],img.img.shape[0],0])
        savefig('aaa-{:04d}.png'.format(k))

    code.interact(local=locals())


if __name__ == '__main__':
    from rectrckr.tools import rtr_init
    rtr_init.main()
