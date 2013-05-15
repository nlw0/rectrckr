#!/usr/bin/python2.7
#coding: utf-8
import argparse
import code

import rectrckr.lowlevel as lowlevel

# import matplotlib as mpl
# mpl.use( "agg" )
from pylab import *
import numpy.testing
import code

from rectrckr.kalman_filter import KalmanFilter

import rectrckr.corisco as corisco_module
from rectrckr.corisco.quaternion import Quat, random_quaternion

CAMERA_MODEL_TEST_REFERENCE_FILE = 'camera_model_test_reference.npy'

def mod(*pp):
    return array([tuple(pp)],
                 dtype=[('cm','=L8'),
                        ('cx','=f8'),
                        ('cy','=f8'),
                        ('fl','=f8'),
                        ('kappa','=f8')])

def main():
    parser = argparse.ArgumentParser(description='Test camera model, projection and stuff.')

    parser.add_argument('test', type=str, choices=['simple_plot', 'camera_models', 'objective'])
    parser.add_argument('--save-test-reference', default=False, action='store_true')

    args = parser.parse_args()

    if args.test == "simple_plot":
        simple_plot()
    elif args.test == "camera_models":
        test_camera_models(args.save_test_reference)
    elif args.test == "objective":
        test_objective()


def test_objective():

    ## TODO: Move more of that to external files, make the test just
    ## read and then check against stored results.
    s = array(mgrid[-1:2:2, -1:2:2, -1:2:2].T.reshape(-1,3), dtype=numpy.float64)

    t, psi = random_pose()
    print 'trans:', t
    print 'psi:', psi
    cmod = mod(1, 320.0, 240.0, 500.0, 0.0)

    pp = project(s, t, psi, cmod)

    ion()
    plot(pp[:,0], pp[:,1], '.')
    title('Camera model test - equidistant distortion')
    axis('equal')
    axis([0,640,480,0])
    grid()
    code.interact(local=locals())



def random_pose():
    psi = random_quaternion()
    t = dot(psi.rot(), array([6*rand()-3,6*rand()-3,8.0 + 4 * rand()]))
    return t, psi.q

def project(s, t, psi, cmod):
    return array([lowlevel.p_wrapper(sk, t, psi, cmod) for sk in s], dtype=numpy.float64)


def test_camera_models(save):
    '''
    Test if the camera models are working. Compare projections to
    previously calculated values, recorded one a bright sunny day when
    we though everything was working fine. The reason we have this
    test is to check out if the program is still working after any
    changes, it is not an actual validation of the models.

    '''

    ## TODO: Move more of that to external files, make the test just
    ## read and then check against stored results.
    s = ascontiguousarray(array(c_[mgrid[-10:11:2,-10:11:2,0:1].reshape(3,-1).T], dtype=float64))
    theta = pi / 16
    t = array([0.0, tan(theta), -1.0]) * 25.0
    psi = array([cos(theta/2), sin(theta/2), 0, 0])
    cx, cy = 320.0, 240.0
    mods = [
        mod(1, cx, cy, 500.0, 0.0),
        mod(2, cx, cy, 500.0, 3e-4),
        mod(2, cx, cy, 500.0, -3e-4),
        mod(3, cx, cy, 180.0, 0.0),
        mod(4, cx, cy, 1.0, 0.0),
        ]
    plist = array([array([lowlevel.p_wrapper(sk, t, psi, cmod) for sk in s]) for cmod in mods])
    if save:
        print plist
        np.save(CAMERA_MODEL_TEST_REFERENCE_FILE, plist)
        print "Reference saved."
    else:
        ref = np.load(CAMERA_MODEL_TEST_REFERENCE_FILE)
        numpy.testing.assert_almost_equal(plist, ref)
        print "Re-calculated projections are correct."


def simple_plot():
    ## Just some points on a plane
    s = ascontiguousarray(array(c_[mgrid[-10:11:2,-10:11:2,0:1].reshape(3,-1).T], dtype=float64))

    ## Tilt the camera a bit, and make it point to the origin.
    ion()
    figure(1)
    theta = pi / 16
    t = array([0.0, tan(theta), -1.0]) * 25.0
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
