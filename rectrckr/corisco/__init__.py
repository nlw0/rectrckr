import filter_sqp as filter_sqp
import corisco_aux as corisco_aux
from quaternion import Quat

from pylab import *


######################################################################
## Functions used in the optimization

## Conic constraint
def val_c(x):
    return np.linalg.norm(x)
def grad_c(x):
    return x/np.linalg.norm(x)
def hess_c(x):
    nx = np.linalg.norm(x)
    return (nx**2 * np.identity(x.shape[0]) - np.outer(x,x)) / nx**3

## Target function
def val_f(x,*fargs):
    return corisco_aux.angle_error(x, fargs[0],fargs[1],fargs[2])
def grad_f(x,*fargs):
    return corisco_aux.angle_error_gradient(x, fargs[0],fargs[1],fargs[2])
def hess_f(x,*fargs):
    return corisco_aux.angle_error_hessian(x, fargs[0],fargs[1],fargs[2])
##
######################################################################

class Corisco():

    def __init__(self, edgels):
        self.edgels = edgels

        #focal_distance = 847.667796003
        #p_point = array([500, 375.0])
        #focal_distance = 500.0
        focal_distance = 500.0
        p_point = array([320.0, 240.0])

        self.sqp_funcs = (val_c, grad_c, hess_c, val_f, grad_f, hess_f)
        
        ## Error function parameters (Tukey bisquare (3) with scale=0.15)
        self.fp = array([3,1,0.15])
        #self.fp = array([3,1,1.0])
        #self.fp = array([2.0,1,0.1])
        #self.fp = array([0.0,1])
        #self.fp = array([1.0,1])
        #self.fp = array([4.0,1,.15])
        ## Intrinsic parameters. pinhole mode (0)
        self.i_param = array([0.0, focal_distance, p_point[0], p_point[1]])

    def estimate_orientation(self, qini):
        args_f = (self.edgels[~isnan(self.edgels[:,2])], self.i_param, self.fp)

        filterSQPout = filter_sqp.filterSQP(qini.q, .0, 1e-3,
                                            self.sqp_funcs,
                                            args_f)

        xo, err, sqp_iters,Llam,Lrho = filterSQPout
        self.qopt = Quat(xo)
        return self.qopt
    

    def target_function(self, x, fp=None):
        if fp is None:
            fp = self.fp
        args_f = (self.edgels[~isnan(self.edgels[:,2])], self.i_param, fp)
        return corisco_aux.angle_error(x, *args_f) 

    def gradient_function(self, x):
        args_f = (self.edgels[~isnan(self.edgels[:,2])], self.i_param, self.fp)
        return corisco_aux.angle_error_gradient(x, *args_f) 

    def plot_edgels(self, ax):
        scale = 20.0
        ax.plot((self.edgels[:,[0,0]] - scale*np.c_[-self.edgels[:,3], self.edgels[:,3]]).T,
                (self.edgels[:,[1,1]] + scale*np.c_[-self.edgels[:,2], self.edgels[:,2]]).T,
                '-',lw=3.0,alpha=1.0,color='#ff0000')

    def plot_vps(self, ax):
        #############################################################
        ## Plot the vanishing point directions at various pixels. ax
        ## is a matplotlib axes, taken with "gca()". Spacing the
        ## separation bweteen the points, and myR the rotation matrix.
        dir_colors=['#ea6949', '#51c373', '#a370ff', '#444444']
        Iheight, Iwidth = (480, 640)
        spacing = 50
        qq = spacing*0.45*np.array([-1,+1])
        bx = 0.+(Iwidth/2)%spacing
        by = 0.+(Iheight/2)%spacing
        qL = np.mgrid[bx:Iwidth:spacing,by:Iheight:spacing].T.reshape((-1,2))
        Nq = qL.shape[0]
        vL = corisco_aux.calculate_vdirs(self.qopt.q, np.array(qL, dtype=np.float32), self.i_param)
        LL = np.zeros((3,Nq,4))
        for lab in range(3):
            for num in range(Nq):
                vx,vy = vL[lab,num]
                k,j = qL[num]
                LL[lab,num,:] = np.r_[k+vx*qq, j+vy*qq]
        for lab in range(3):
            ax.plot( LL[lab,:,:2].T, LL[lab,:,2:].T, dir_colors[lab], lw=3, alpha=1.0 if lab<2 else 0.3)
        ##
        #############################################################
