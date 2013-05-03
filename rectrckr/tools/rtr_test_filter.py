#!/usr/bin/python2.7
#coding: utf-8

import argparse
from pylab import *
import scipy.stats

from rectrckr.particle_filter import sample_multinomial

from rectrckr.corisco.quaternion import Quat, random_quaternion

from rectrckr.particle_filter import ParticleFilter

def main():
    parser = argparse.ArgumentParser(description='Test Corisco.')
    parser.add_argument('test', type=str)

    args = parser.parse_args()

    if args.test == 'multi':
        test_multinomial()
    if args.test == 'filt':
        test_filter()


def test_multinomial():
    Ns = 100000

    print '** Test uniform distribution'
    ww = ones(13)
    test_from_weights(Ns, ww)

    print '** Test random distribution'
    ww = rand(13)
    test_from_weights(Ns, ww)

def test_from_weights(Ns, ww):
    Nd = ww.shape[0]
    ss = sample_multinomial(Ns, ww)
    pp = ww / ww.sum()

    count = zeros(Nd)
    for k in xrange(Nd):
        count[k] = (ss == k).sum()

    print 'Count from {} uniform multinomial samples, {} cetegories:'.format(Ns, Nd)
    print count
    print 'Expected count'
    print pp * Ns
    print 'Chi-squared H0 test:'
    chi2, p_value = scipy.stats.chisquare(count, pp * Ns)
    print 'chi2:{} p_value:{}'.format(chi2, p_value)
    assert p_value < 0.9995 and p_value > 0.0005


def dynamical_model(st):
    covariance = diag([0.1,0.1,0.1])
    new_pos = multivariate_normal(st[:3], covariance)
    new_ori = Quat(st[3:7]) * random_quaternion(0.01)
    return r_[new_pos, new_ori.q]

def test_filter():
    ini = array([0.0,0.0,0.0,1.0,0.0,0.0,0.0])
    
    pafi = ParticleFilter(ini, dynamical_model)
    print pafi.state
    print pafi.weights
    print
    pafi.predict_state()
    print pafi.state
    print pafi.weights
    print
    pafi.predict_state()
    print pafi.state
    print pafi.weights
    print
    pafi.predict_state()
    print pafi.state
    print pafi.weights
    print
    pafi.predict_state()
    print pafi.state
    print pafi.weights
    print
    pafi.predict_state()
    print pafi.state
    print pafi.weights
    print

    

