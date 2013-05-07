import filter_sqp as filter_sqp
import corisco_aux as corisco_aux
from quaternion import Quat, random_quaternion

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

dir_colors=['#ea6949', '#51c373', '#a370ff', '#444444']


def aligned_quaternion(v):
    ## The largest component
    ll = np.argmax(np.abs(v))
    ee = np.zeros(3)
    ee[ll] = np.sign(v[ll])
    q=Quat(np.cross(v, ee))
    R = q.sqrt()

    if np.sign(v[ll]) > 0:
        if ll == 1:
            R = R*Quat(1,-1,-1,-1).normalize()
        if ll == 2:
            R = R*Quat(1,1,1,1).normalize()

    if np.sign(v[ll]) < 0:
        if ll == 0:
            R = R*Quat(0,0,1,0).normalize()
        if ll == 1:
            R = R*Quat(1,1,-1,1).normalize()
        if ll == 2:
            R = R*Quat(1,-1,-1,1).normalize()
    return R.inverse()


class Corisco():

    def __init__(self, edgels):
        self.edgels = edgels
        self.Ned = self.edgels.shape[0]

        #focal_distance = 847.667796003
        #p_point = array([500, 375.0])
        #focal_distance = 500.0
        focal_distance = 500.0
        p_point = array([320.0, 240.0])

        self.sqp_funcs = (val_c, grad_c, hess_c, val_f, grad_f, hess_f)
        
        ## Error function parameters (Tukey bisquare (3) with scale=0.15)
        #self.loss = array([0.0,1])
        self.loss = array([3.0,1,0.15])
        #self.loss = array([3,1,1.0])
        #self.loss = array([2.0,1,0.1])
        #self.loss = array([1.0,1])
        #self.loss = array([4.0,1,.15])
        ## Intrinsic parameters. pinhole mode (0)
        self.i_param = array([0.0, focal_distance, p_point[0], p_point[1]])

        self.normals = corisco_aux.calculate_normals(self.edgels, self.i_param)

    def estimate_orientation(self, qini):
        args_f = (self.edgels[~isnan(self.edgels[:,2])], self.i_param, self.loss)
        # args_f = (self.edgels, self.i_param, self.loss)

        filterSQPout = filter_sqp.filterSQP(qini.q, .0, 1e-1,
                                            self.sqp_funcs,
                                            args_f)

        xo, err, sqp_iters,Llam,Lrho = filterSQPout
        return Quat(xo)

    def ransac_search(self, initial_trials):
        ## Estimate solution using RANSAC
        bestv = np.Inf ## Smallest value found
        args_f = (self.edgels[~isnan(self.edgels[:,2])], self.i_param, self.loss)
        for k in range(initial_trials):
            ## Pick indices of the reference normals. Re-sample until we
            ## get a list of three different values.
            pk_a = np.random.random_integers(0,self.Ned-1)
            pk_b = np.random.random_integers(0,self.Ned-1)
            while pk_b == pk_a:
                pk_b = np.random.random_integers(0,self.Ned-1)
            pk_c = np.random.random_integers(0,self.Ned-1)
            while pk_c == pk_a or pk_c == pk_b:
                pk_c = np.random.random_integers(0,self.Ned-1)

            ## Get the normals with the first two chosen indices, and
            ## calculate a rotation matrix that has the x axis aligned to
            ## them.
            n_a = self.normals[pk_a]
            n_b = self.normals[pk_b]
            vp1 = np.cross(n_a, n_b)
            vp1 = vp1 * (vp1**2).sum()**-0.5
            q1 = aligned_quaternion(vp1)

            ## Pick a new random third norm, and find the rotation to align
            ## the y direction to this edgel.
            n_c = self.normals[pk_c]
            vaux = np.dot(n_c, q1.rot())
            ang = np.arctan2(vaux[1], -vaux[2])
            q2 = Quat(np.sin(ang/2),0,0) * q1 ## The resulting orientation

            ## Find the value of the target function for this sampled
            ## orientation.
            newv = corisco_aux.angle_error(q2.q, *args_f) 

            ## If the value is the best yet, store solution.
            if newv <= bestv :
                bestv = newv
                bpk_a = pk_a
                bpk_b = pk_b
                bpk_c = pk_c
                qopt = q2
        return qopt, bpk_a, bpk_b, bpk_c

    def silly_random_search(self, initial_trials):
        ## Estimate solution using RANSAC
        bestv = np.Inf ## Smallest value found
        args_f = (self.edgels, self.i_param, self.loss)
        for k in range(initial_trials):
            qtest = random_quaternion().canonical()
            newv = corisco_aux.angle_error(qtest.q, *args_f)

            ## If the value is the best yet, store solution.
            if newv <= bestv:
                bestv = newv
                qopt = qtest
        return qopt

    def target_function_value(self, x, loss=None):
        if loss is None:
            loss = self.loss
        args_f = (self.edgels[~isnan(self.edgels[:,2])], self.i_param, loss)
        # args_f = (self.edgels, self.i_param, loss)
        return corisco_aux.angle_error(x, *args_f) 

    def target_function_gradient(self, x, loss=None):
        if loss is None:
            loss = self.loss
        args_f = (self.edgels[~isnan(self.edgels[:,2])], self.i_param, loss)
        # args_f = (self.edgels, self.i_param, loss)
        return corisco_aux.angle_error_gradient(x, *args_f)

    def target_function_gradient_numeric(self, x, loss=None, dx=1e-6):
        if loss is None:
            loss = self.loss
        args_f = (self.edgels, self.i_param, loss)
                
        return array([
                (corisco_aux.angle_error(x + array([ dx,0,0,0]), *args_f) - 
                 corisco_aux.angle_error(x + array([-dx,0,0,0]), *args_f)) / (2 * dx),
                (corisco_aux.angle_error(x + array([0, dx,0,0]), *args_f) - 
                 corisco_aux.angle_error(x + array([0,-dx,0,0]), *args_f)) / (2 * dx),
                (corisco_aux.angle_error(x + array([0,0, dx,0]), *args_f) - 
                 corisco_aux.angle_error(x + array([0,0,-dx,0]), *args_f)) / (2 * dx),
                (corisco_aux.angle_error(x + array([0,0,0, dx]), *args_f) - 
                 corisco_aux.angle_error(x + array([0,0,0,-dx]), *args_f)) / (2 * dx),
                ])

    def plot_edgels(self, ax):
        scale = 20.0
        ax.plot((self.edgels[:,[0,0]] - scale*np.c_[-self.edgels[:,3], self.edgels[:,3]]).T,
                (self.edgels[:,[1,1]] + scale*np.c_[-self.edgels[:,2], self.edgels[:,2]]).T,
                '-',lw=3.0,alpha=1.0,color='#ff0000')

    def plot_vps(self, ax, qopt):
        #############################################################
        ## Plot the vanishing point directions at various pixels. ax
        ## is a matplotlib axes, taken with "gca()". Spacing the
        ## separation bweteen the points, and myR the rotation matrix.
        Iheight, Iwidth = (480, 640)
        spacing = 50
        qq = spacing*0.45*np.array([-1,+1])
        bx = 0.+(Iwidth/2)%spacing
        by = 0.+(Iheight/2)%spacing
        qL = np.mgrid[bx:Iwidth:spacing,by:Iheight:spacing].T.reshape((-1,2))
        Nq = qL.shape[0]
        vL = corisco_aux.calculate_vdirs(qopt.q, np.array(qL, dtype=np.float32), self.i_param)
        LL = np.zeros((3,Nq,4))
        for lab in range(3):
            for num in range(Nq):
                vx,vy = vL[lab,num]
                k,j = qL[num]
                LL[lab,num,:] = np.r_[k+vx*qq, j+vy*qq]
        for lab in [2,1,0]:
            #ax.plot( LL[lab,:,:2].T, LL[lab,:,2:].T, dir_colors[lab], lw=3, alpha=1.0 if lab<2 else 0.3)
            ax.plot( LL[lab,:,:2].T, LL[lab,:,2:].T, dir_colors[lab], lw=3)
        ##
        #############################################################
