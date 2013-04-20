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
        self.Caccel = 100.0

        self.Cobs = identity(4) * 0.2

        self.time = 0.0

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

        ## Incorporate new observations into state estimation.
        self.state = self.state + dot(self.K, residue)
        self.Cstate = dot(identity(8) - dot(self.K, self.Mobser),self.Cstate)


if __name__ == '__main__':

    xx = loadtxt(sys.argv[1])[:,:9]

    xx = xx[1100:1600]

    xx[:, [0,3,6]] -= 320
    xx[:, [1,4,7]] -= 240
    xx[:, [0,1,3,4,6,7]] *= 1157.0/580.0

    for n in range(1,xx.shape[0]):
       xx[n] = assoc(xx[n-1], xx[n])

    kalman = Kalman6DOF()
    kalman.state[6] = 1.0 ## "0" Quaternion

    kalman.pts = \
        array([[ 79.23584838,  58.55128818, -43.88465773],
               [-28.53663772, -99.4676606 ,  90.4078352 ],
               [-49.32580004,  38.82926386, -41.75416534]])

        # "Fitting" to the frist period
        # array([[ 79.23584838,  58.55128818, -43.88465773],
        #        [-28.53663772, -99.4676606 ,  90.4078352 ],
        #        [-49.32580004,  38.82926386, -41.75416534]])
        # array([[ 79.2384991 ,  58.57048068, -43.86812725],
        #        [-28.53136383, -99.44582314,  90.42602827],
        #        [-49.32450121,  38.8516144 , -41.73621901]])

        # first two periods
        # array([[  71.40769031,   45.05383917,  -32.16122414],
        #        [ -16.79256927, -101.14733083,   93.54881313],
        #        [ -53.22244106,   54.13339912,  -56.51138144]])

        ## First estimate, I forgot how I got it!
        # array([[ 78.49181092,  58.59872529, -45.32      ],
        #        [-28.5107408 , -97.26114368,  87.48      ],
        #        [-49.98107011,  38.66241839, -42.16      ]])

        
    kalman.pts = kalman.pts[arrj[5]]

    kalman.state[0:3] = mean(xx[0].reshape(-1,3))

    kalman.Cstate = 100.0 * identity(13) ## Initial state covariance

    xout = zeros((xx.shape[0], 13))
    zout = zeros((xx.shape[0], 12))

    eout = zeros(xx.shape[0])
    
    # mout = zeros((350,9))

    dt = .042
    for n in range(xx.shape[0]):
        new_data = xx[n]
        # new_data = assoc(kalman.z_hat, xx[n])

        # print 70*'-'
        kalman.predict_state(dt)
        # print 'pred st: ', kalman.state
        kalman.predict_observations()
        # print 'pred obs:', kalman.z_hat
        # print 'measured:', new_data
        merr = norm(kalman.z_hat - new_data)
        eout[n] = merr
        # print 'measurement error:', merr

        kalman.update_from_observations(new_data)
        # print 'updt st:', kalman.state

        ## Log predicted state and outputs
        xout[n] = kalman.state
        zout[n,:9] = kalman.z_hat
        zout[n,9:12] = kalman.state[:3]

    print '--> mean observation prediction error level:', mean(log10(((xx-zout[:,:9])**2).sum(1)))

    ion()

    figure(1)
    plot(xx, 'b+')
    plot(zout[:,:9], 'r-')

    figure(2)
    plot(eout)


