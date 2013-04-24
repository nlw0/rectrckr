import sys
from pylab import *

## The next two functions update the part of the respective matrices
## that depend on the time elapsed.
def transition_matrix(M, dt):
    M[0:4,4:8] = dt * identity(4)

def acceleration_matrix(M, dt):
    M[0:4,0:4] = .5*dt**2 * identity(4)
    M[4:8,0:4] = dt * identity(4)

class KalmanFilter:
    def __init__(self):

        ## The system state: 2x 2d position and 2d velocity.
        self.state = zeros(8)
        self.previous_state = zeros(8)

        self.Cstate = zeros((13,13)) ## State covariance matrix
        self.Cstate = 0.1 * identity(8)

        ## Vector to store the predicted observation values, the
        ## positions of the vertical and horizontal edges.
        self.z_hat = zeros(4)

        ## Transition and observation matrices.
        self.Mtrans = identity(8)
        self.Mobser = zeros([4,8])

        ## The transition jacobian regarding the acceleration. Used to
        ## compute the transition covariance.
        self.Maccel = zeros([8, 4])

        ## Variance of the acceleration. (Should be properly measured
        ## and given as a parameter in the initialization.)
        # self.Caccel = 0.1
        self.Caccel = 1.0

        self.Cobs = identity(4) * 100.0 #nice
        #self.Cobs = identity(4) * 0.5 #notnice

        self.time = 0.0

        self.outlier_threshold = 30.0


    def predict_state(self, dt):
        ## Set the transition matrix from the current orientation
        transition_matrix(self.Mtrans, dt)
        acceleration_matrix(self.Maccel, dt)

        ## Store current state
        self.previous_state[:] = self.state
        ## Calculate the new state using the transition matrix
        self.state = dot(self.Mtrans, self.state)

        ########################################################################
        ## Update covariance of state estimate
        ##
        ## Calculate covariance to be added from an assumed normal
        ## random acceleration.
        Ctrans = self.Caccel * dot(self.Maccel,self.Maccel.T)
        self.Cstate = dot(dot(self.Mtrans,self.Cstate), self.Mtrans.T) + Ctrans

    def predict_observations(self):
        self.z_hat = dot(self.Mobser, self.state)
        # Cobs is constant

    def update_from_observations(self, z):
        residue = z - self.z_hat
        Cresidue = dot(dot(self.Mobser, self.Cstate), self.Mobser.T) + self.Cobs

        ## Kalman gain
        self.K = dot(dot(self.Cstate, self.Mobser.T), inv(Cresidue))

        ## Detect outliers, and replace then with predicted values
        residue[(array([-1,1,-1,1]) * residue) > self.outlier_threshold] = 0

        ## Incorporate new observations into state estimation.
        self.state = self.state + dot(self.K, residue)
        self.Cstate = dot(identity(8) - dot(self.K, self.Mobser),self.Cstate)


