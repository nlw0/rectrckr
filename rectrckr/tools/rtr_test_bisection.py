#!/usr/bin/python2.7
#coding: utf-8

import argparse
import code
from pylab import *

from rectrckr.filter_sqp.trust_bisection import find_step_size, calculate_distance

def main():
    parser = argparse.ArgumentParser(description='Test bisection technique to find step size in trust region optimization methods.')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--lam', type=float, nargs='+')
    parser.add_argument('--rho', type=float)

    args = parser.parse_args()

    print 'α:', args.alpha
    print 'λ:', args.lam

    alpha = args.alpha
    lam = array(args.lam)
    rho = args.rho

    min_lam = max(0,min(lam))
    rez=0.01
    xx = mgrid[min_lam+rez:10.0:rez]
    yy = array([calculate_distance(alpha, lam, x) for x in xx])





    ion()
    loglog(xx, yy)
    loglog(xx[[0,-1]], [rho, rho], 'r-')

    grid()

    code.interact()
