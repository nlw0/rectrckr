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

    alpha = args.alpha
    lam = array(args.lam)
    rho = args.rho

    print 'α:', alpha
    print 'λ:', lam
    print 'ρ:', rho

    nu = find_step_size(alpha, lam, rho, 1e-5)
    print 'nu=', nu
    print "f(nu)-rho=", calculate_distance(alpha, lam, nu) - rho

    rez=0.01

    xx = mgrid[nu/100.0:nu*100:rez]
    yy = array([calculate_distance(alpha, lam, x) for x in xx])

    ion()
    loglog(xx, yy)
    #loglog(xx[[1,-2]], [rho, rho], 'r-')

    loglog([nu/10.0, nu*10.0], [rho, rho], 'r-')
    loglog(nu, calculate_distance(alpha, lam, nu), 'rd')

    grid()

    code.interact()


