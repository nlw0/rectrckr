import sys
from pylab import *
from numpy.random import multivariate_normal

## The next two functions update the part of the respective matrices
## that depend on the time elapsed.
def transition_matrix(M, dt):
    M[0:4,4:8] = dt * identity(4)


def acceleration_matrix(M, dt):
    M[0:4,0:4] = .5*dt**2 * identity(4)
    M[4:8,0:4] = dt * identity(4)


def sample_multinomial(number, weights):
    cp = cumsum(weights)
    cp = cp / cp[-1]
    output = zeros(number)
    for k in xrange(number):
        output[k] = find(rand() < cp)[0]
    return output


class ParticleFilter:
    def __init__(self, initial_state, dyna):
        ## The system state: camera 3D position and orientation.
        self.Nsmp = 13
        self.Ndim = 7
        self.state = r_[self.Nsmp * [initial_state]]
        self.weights = ones(self.Nsmp)
        self.dynamical_model = dyna

    def predict_state(self):
        new_state = zeros((self.Nsmp, self.Ndim))
        cp = cumsum(self.weights)
        who = sample_multinomial(self.Nsmp, self.weights)
        for k in xrange(self.Nsmp):
            new_state[k] = self.dynamical_model(self.state[who[k]])
        self.state = new_state


    def predict_observations(self):
        pass

    def update_from_observations(self, z):
        pass
