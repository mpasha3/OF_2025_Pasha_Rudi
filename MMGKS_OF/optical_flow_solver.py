from utilities.imports import *
import scipy
from scipy import signal
import cv2
from joblib import Parallel, delayed


def gen(n):
    D = scipy.sparse.diags(diagonals=np.ones(n-1), offsets=1, shape=None, format=None, dtype=None).toarray()
    D_ = scipy.sparse.diags(diagonals=np.ones(n-1)*(-1), offsets=-1, shape=None, format=None, dtype=None).toarray()
    L = D + D_
    L[0] = 0
    L[-1] = 0
    return( L)
def gen_2D(nx, ny):
    D_x = gen(nx)
    D_y = gen(ny)
    IDx = sparse.kron( sparse.identity(nx), D_x)
    DyI = sparse.kron(D_y, sparse.identity(ny))
    L = sparse.vstack((IDx, DyI))
    return L

def solve_of(u_traj,shape,t_end,v_trues,v_max = 2, n_iter = 60,reduction = False,pnorm=2,qnorm=2,proj_dim=1, regparam="gcv", centered_ut = True,power=0.5,epsilon=1e-2,v_scale = "cv2_pyrup",**kwargs):
    from MMGKS_OF import MMGKS2
    '''Solves the regularized optical flow equation'''
    if (reduction == True):
        scale = kwargs['scale'] if ('scale' in kwargs) else 2
    else:
        scale = 1
    parallel = kwargs['parallel'] if ('parallel' in kwargs) else True
    n_jobs = kwargs['n_jobs'] if ('n_jobs' in kwargs) else 20
    delta_x = v_max
    delta_y =delta_x
    size =shape[0]
    nx=shape[0];ny=shape[1]
    R = np.array(list(np.ndindex(*shape)))
    kernel_x = np.zeros((2*v_max+1,2*v_max+1))
    kernel_x[:,0] = -1 
    kernel_x[:,-1] = 1 
    kernel_y = kernel_x.T 
    kernel_t = np.ones((2*v_max+1,2*v_max+1))

    u_traj_  = deepcopy(u_traj)
    
    Ls = []
    for i in range(len(u_traj)-1):
        u_traj = deepcopy([u.reshape(shape) for u in u_traj_])
        n_pyr = int(np.log2(scale))
        shape_ = shape
        nx_ =nx
        ny_= ny
        if (n_pyr >=1 ):
            #print(n_pyr)
            for j in range(0,n_pyr):
                #print(j,shape_)
                if (i>0):
                    u_traj[i-1]  = cv2.pyrDown(u_traj[i-1].reshape(shape_)) 
                u_traj[i]  = cv2.pyrDown(u_traj[i].reshape(shape_)) # cv2.resize(u_traj_[i].reshape(shape), (nx//scale,ny//scale)) #
                u_traj[i+1]  = cv2.pyrDown(u_traj[i+1].reshape(shape_)) # cv2.resize(u_traj_[i+1].reshape(shape), (nx//scale,ny//scale)) #

                nx_ = nx_//2
                ny_ = ny_//2 
                shape_ = (nx_,ny_)
                #print(j,shape_)
        uy = -vec(signal.convolve2d(u_traj[i], kernel_x, boundary='symm', mode='same'))/(delta_x*2*kernel_x.shape[0])
        ux = -vec(signal.convolve2d(u_traj[i], kernel_y, boundary='symm', mode='same'))/(delta_y*2*kernel_y.shape[0])


        if centered_ut == True:
            if i==0:
                ut = vec(signal.convolve2d(u_traj[i+1], kernel_t, boundary='symm', mode='same') - signal.convolve2d(u_traj[i], kernel_t, boundary='symm', mode='same'))/(kernel_t.sum()) #vec(u_traj[i+1] - u_traj[i]) #
            else: 
                ut = vec(signal.convolve2d(u_traj[i+1], kernel_t, boundary='symm', mode='same') - signal.convolve2d(u_traj[i-1], kernel_t, boundary='symm', mode='same'))/(2*kernel_t.sum()) #vec(u_traj[i+1] - u_traj[i-1]) /2#
        else:
            ut = vec(signal.convolve2d(u_traj[i+1], kernel_t, boundary='symm', mode='same') - signal.convolve2d(u_traj[i], kernel_t, boundary='symm', mode='same'))/(kernel_t.sum())     
        Li = []
        for i in range(nx_*ny_): 
            Li.append((ux[i],uy[i],ut[i]))
        Li = np.array(Li)
        Ls.append(Li)

        del(ux,uy,ut)

    L = gen_first_derivative_operator_2D(nx_,ny_)

    from scipy.sparse import csr_matrix

    Lv =  scipy.sparse.block_diag([L,L])

    v_ests = []
    v_larges =[]
    infos = []

    def solver(i):
        if v_trues is not None:
            v_true=v_trues[i].reshape(v_trues[i].size,1).reshape(nx_*ny_,2).T.flatten().reshape(v_trues[i].size,1)
        else:
            v_true = None
        #Lx = Ls[i][:,0];Ly = Ls[i][:,1];Lt = Ls[i][:,2]

        ux_diag = scipy.sparse.block_diag([Ls[i][:,0][k] for k in range(nx_*ny_)])
        uy_diag = scipy.sparse.block_diag([Ls[i][:,1][k] for k in range(nx_*ny_)])

        ux_uy = scipy.sparse.hstack((ux_diag, uy_diag))
        #print(ux_uy.shape)
        ut = Ls[i][:,2]
        #print(ux_uy.shape, Lv.shape)
        #print(v_true.shape)
        (v_est, info) = MMGKS2(ux_uy, -ut.reshape((len(ut),1)), Lv, pnorm=pnorm, qnorm=qnorm, projection_dim=proj_dim, n_iter=n_iter, regparam=regparam,  
                        x_true=v_true, tqdm_ = False,power=power,epsilon=epsilon,x0 = None)
        #print(v_est.shape)
        del (ux_diag,uy_diag,ux_uy, ut)
        v_est = v_est.reshape(2, nx_*ny_).T.flatten()
        v_est = v_est.reshape(nx_*ny_,2)
        v_large = np.zeros((nx,ny,2))

        v_est = v_est.reshape((nx_,ny_,2))

        if (v_scale == "cv2_resize"):
            v_large[:,:,0] = cv2.resize(v_est[:,:,0], (ny,nx), interpolation=cv2.INTER_LINEAR) # cv2.pyrUp(v_est[:,:,0]) #cv2.resize(v_est[:,:,0], (nx,ny), interpolation=cv2.INTER_LINEAR)
            v_large[:,:,1] =cv2.resize(v_est[:,:,1], (ny,nx), interpolation=cv2.INTER_LINEAR) # cv2.pyrUp(v_est[:,:,1]) # cv2.resize(v_est[:,:,1], (nx,ny), interpolation=cv2.INTER_LINEAR)

        elif (v_scale == "cv2_pyrup"):
            v_temp_0 = v_est[:,:,0]
            v_temp_1 = v_est[:,:,1]

            for j in range(0,n_pyr):

                v_temp_0 = cv2.pyrUp(v_temp_0)#    cv2.resize(v_est[:,:,0], (ny,nx), interpolation=cv2.INTER_AREA) # cv2.pyrUp(v_est[:,:,0]) #cv2.resize(v_est[:,:,0], (nx,ny), interpolation=cv2.INTER_LINEAR)
                v_temp_1 = cv2.pyrUp(v_temp_1)#   cv2.resize(v_est[:,:,1], (ny,nx), interpolation=cv2.INTER_AREA) # cv2.pyrUp(v_est[:,:,1]) # cv2.resize(v_est[:,:,1], (nx,ny), interpolation=cv2.INTER_LINEAR)
            
            v_large[:,:,0] = v_temp_0#    cv2.resize(v_est[:,:,0], (ny,nx), interpolation=cv2.INTER_AREA) # cv2.pyrUp(v_est[:,:,0]) #cv2.resize(v_est[:,:,0], (nx,ny), interpolation=cv2.INTER_LINEAR)
            v_large[:,:,1] = v_temp_1#   cv2.resize(v_est[:,:,1], (ny,nx), interpolation=cv2.INTER_AREA) # cv2.pyrUp(v_est[:,:,1]) # cv2.resize(v_est[:,:,1], (nx,ny), interpolation=cv2.INTER_LINEAR)
            
        # v_large[:,:,0] = cv2.resize(v_est[:,:,0], (ny,nx), interpolation=cv2.INTER_LINEAR) # cv2.pyrUp(v_est[:,:,0]) #cv2.resize(v_est[:,:,0], (nx,ny), interpolation=cv2.INTER_LINEAR)
        # v_large[:,:,1] =cv2.resize(v_est[:,:,1], (ny,nx), interpolation=cv2.INTER_LINEAR) # cv2.pyrUp(v_est[:,:,1]) # cv2.resize(v_est[:,:,1], (nx,ny), interpolation=cv2.INTER_LINEAR)
        v_large = v_large.reshape((nx,ny,2))*scale
        return v_est,v_large,info

    if (parallel == True):
        result = Parallel(n_jobs=n_jobs)(delayed(solver)(i) for i in range(len(Ls)))
        for i in range(len(Ls)):
            v_ests.append(result[i][0])
            v_larges.append(result[i][1])
            infos.append(result[i][2])
    else:     
        for i in range(len(Ls)):
            result = solver(i)
            v_ests.append(result[0])
            v_larges.append(result[1])
            infos.append(result[2])

    return (v_ests, v_larges, infos)
