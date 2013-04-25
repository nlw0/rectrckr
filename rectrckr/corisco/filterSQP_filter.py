#!/usr/bin/python
#coding:utf-8

# Copyright 2012 Nicolau Leal Werneck, Anna Helena Reali Costa and
# Universidade de São Paulo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## This module contains a class that implements the filter that is
## used in the filter SQP nonlinear programming technique.

import numpy as np

class FletcherFilter:
    def __init__(self):
        max_pt = 2000
        self.values = np.zeros((max_pt,2))
        self.q = np.zeros((max_pt,1))
        self.mu = np.zeros((max_pt,1))
        self.valid = self.values[:,0]>1
        self.beta = 0.99
        self.alpha1 = 0.25
        self.alpha2 = 0.0001
        

    def dominated(self, pt, delta_q):
        ## Test if point is acceptable and return the opposite
        # test = self.values[self.valid] < pt
        # return (test[:,0] * test[:,1]).any()
        # testh = pt[1] <= self.values[self.valid,1]
        # testf = pt[0] <= self.values[self.valid,0]
        # return not (testh + testf).all()

        testh = pt[1] <= (self.beta * self.values[self.valid,1])
        testf1 = pt[0] <= (self.values[self.valid,0] - self.alpha1 * self.q[self.valid])
        testf2 = pt[0] <= (self.values[self.valid,0]
                             - self.alpha2 * self.values[self.valid,1] * self.mu[self.valid])
        
        ## The north-west rule.
        nw_rule = True
        if self.valid.any():
            leftmost = np.nonzero(self.valid)[0][np.argmin(self.values[self.valid,0])]
            lmu = 1e3*self.mu[leftmost]
            nw_rule = (pt[0]+lmu*pt[1]) <= (self.values[leftmost,0]+lmu*self.values[leftmost,1])

        return not nw_rule * (testh + testf1 * testf2).all()



    def add(self, pt, delta_q, lam):
        #print sum(self.valid)
        if self.dominated(pt, delta_q):
            return
        emp = np.nonzero(1-self.valid)[0]
        assert emp != []
        ei = emp[0]
        self.values[ei] = pt
        self.valid[ei] = True
        self.q[ei] = delta_q
        self.mu[ei] = np.clip(1e-6,np.max(np.abs(lam)), 1e6)
        dom = (pt < self.values)
        dom = (dom[:,0] * dom[:,1]) * self.valid
        self.valid[dom] = False

if __name__ == '__main__':
    from pylab import *

    ion()

    fil = FletcherFilter()
    Niter = 50
    logp = zeros((Niter,2))
    for k in range(Niter):
        while True:
            print k
            p = rand(2)
            if not fil.dominated(p):
                break
        logp[k] = p
        fil.add(p)
        ff = fil.values[fil.valid]
        #ff = k *.01* array([-1,-1])+ r_[[[0,1]],ff[argsort(ff[:,0])],[[1,0]]]
        ff = r_[[[0,1]],ff[argsort(ff[:,0])],[[1,0]]]
        ww = zeros((ff.shape[0]*2-1,2))
        ww[::2] = ff
        ww[1::2,0] = ff[1:,0]
        ww[1::2,1] = ff[:-1,1]
        #plot(ff[:,0], ff[:,1], '-')
        #plot(ww[:,0], ww[:,1], '-')
        loglog(ww[2:-3,0], ww[2:-3,1], '-')
    #plot(logp[:,0], logp[:,1], 'o')
    axis([0,1,0,1])
    axis('equal')
    grid()
        
