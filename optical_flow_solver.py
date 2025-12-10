from imports import *
import scipy
from scipy import signal

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


def solve_opt_flow_joint(u_traj,shape,t_end,v_true,v_max = 2, n_iter = 60):
    '''Solves the regularized optical flow equation'''
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


    Ls = []
    for i in range(len(u_traj)-1):

        u_traj[i] = u_traj[i].reshape(shape)
        u_traj[i+1] = u_traj[i+1].reshape(shape)

        uy = -vec(signal.convolve2d(u_traj[i], kernel_x, boundary='symm', mode='same'))/(delta_x*2*kernel_x.shape[0])
        ux = -vec(signal.convolve2d(u_traj[i], kernel_y, boundary='symm', mode='same'))/(delta_y*2*kernel_y.shape[0])

        ut = vec(signal.convolve2d(u_traj[i+1], kernel_t, boundary='symm', mode='same') - signal.convolve2d(u_traj[i], kernel_t, boundary='symm', mode='same'))/(kernel_t.size)
        Li = []
        for i in range(nx*ny):
            Li.append((ux[i],uy[i],ut[i]))
        Li = np.array(Li)
        Ls.append(Li)
    #print('Done')
    ux_uys = []
    for i in range(len(Ls)):
        Lx = Ls[i][:,0];Ly = Ls[i][:,1];Lt = Ls[i][:,2]

        ux_uy = scipy.linalg.block_diag(*[np.array([Lx[i], Ly[i]]) for i in range(size**2)])

        ux_uys.append(ux_uy)
    uts = [Ls[i][:,2] for i in range(len(Ls))] #[u_traj[i+1] - u_traj[i] for i in range(len(Ls))]#

    # L = gen_first_derivative_operator_2D(nx,ny)
    L = gen_2D(nx,ny)
    a=np.zeros((L.shape[0],2*L.shape[1]))
    from scipy.sparse import csr_matrix
    a = csr_matrix((L.shape[0],2*L.shape[1]))#.toarray()
    a[:,::2] = L
    b=np.zeros((L.shape[0],2*L.shape[1]))
    b = csr_matrix((L.shape[0],2*L.shape[1]))#.toarray()
    b[:,1::2] = L
    Lv =  sparse.vstack((a,b))


    ux_uy_bar = scipy.sparse.block_diag([ux_uys[i] for i in range(t_end-1)])#.toarray()

    Lv_bar = scipy.sparse.block_diag([Lv for i in range(t_end-1)])#.toarray()
    ut_bar = vectorize_func(np.array(uts))

    if v_true is not None:
        v_true = vec(np.array([v.reshape(v.size) for v in v_true])).reshape((vec(np.array([v.reshape(v.size) for v in v_true])).size,1))

    # print(np.isnan(ux_uy_bar.toarray()).any())
    (v_ests_, info) = MMGKS(ux_uy_bar, -ut_bar.reshape((len(ut_bar),1)), Lv_bar, pnorm=2, qnorm=1, projection_dim=1, n_iter=n_iter, regparam='gcv',
                        x_true=v_true, tqdm_ = False)

    return ([np.rint(v_ests_)[(len(v_ests_)//(t_end-1))*t:(len(v_ests_)//(t_end-1)*(t+1))].reshape(nx,ny,2) for t in range(t_end-1)],info)



def solve_opt_flow(u_traj,shape,t_end,v_trues,v_max = 2, n_iter = 60,reduction = False,pnorm=2,qnorm=2,proj_dim=1,**kwargs):
    '''Solves the regularized optical flow equation'''
    if (reduction == True):
        scale = kwargs['scale'] if ('scale' in kwargs) else 2
    else:
        scale = 1

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
    u_traj = deepcopy(u_traj)

    u_traj_  = deepcopy(u_traj)
    Ls = []

    ux_history = []
    uy_history = []

    for i in range(len(u_traj)-1):

            # Reshape the image array into a 4D array where each element is a 2x2 block
        blocks = u_traj_[i].reshape(nx//scale, scale, ny//scale, scale)

        # Take the mean along the last two axes to get the average of each 2x2 block
        u_traj[i] = np.mean(blocks, axis=(1, 3))

        blocks = u_traj_[i+1].reshape(nx//scale, scale, ny//scale, scale)

        # Take the mean along the last two axes to get the average of each 2x2 block
        u_traj[i+1] = np.mean(blocks, axis=(1, 3))

        shape_ = u_traj[i].shape
        nx_=shape_[0]
        ny_=shape_[1]

        u_traj[i] = u_traj[i].reshape(shape_)
        u_traj[i+1] = u_traj[i+1].reshape(shape_)

        uy = -vec(signal.convolve2d(u_traj[i], kernel_x, boundary='symm', mode='same'))/(delta_x*2*kernel_x.shape[0])
        ux = -vec(signal.convolve2d(u_traj[i], kernel_y, boundary='symm', mode='same'))/(delta_y*2*kernel_y.shape[0])
        ux_history.append(ux)
        uy_history.append(uy)

        if i == 0:
            ut = vec(signal.convolve2d(u_traj[i+1], kernel_t, boundary='symm', mode='same') - signal.convolve2d(u_traj[i], kernel_t, boundary='symm', mode='same'))/(kernel_t.size)
        else:
            ut = vec(signal.convolve2d(u_traj[i+1], kernel_t, boundary='symm', mode='same') - signal.convolve2d(u_traj[i-1], kernel_t, boundary='symm', mode='same'))/(2*kernel_t.size)

        Li = []
        for i in range(nx_*ny_):
            Li.append((ux[i],uy[i],ut[i]))
        Li = np.array(Li)
        Ls.append(Li)

    L = gen_first_derivative_operator_2D(nx_,ny_)
    # a=np.zeros((L.[0],2*L.[1]))
    from scipy.sparse import csr_matrix
    a = csr_matrix((L.shape[0],2*L.shape[1]))#.toarray()
    a[:,::2] = L
    # b=np.zeros((L.[0],2*L.shape[1]))
    b = csr_matrix((L.shape[0],2*L.shape[1]))#.toarray()
    b[:,1::2] = L
    Lv =  sparse.vstack((a,b))

    v_ests = []
    v_larges =[]
    ux_uy_history = []
    ut_history = []
    for i in range(len(Ls)):
        if v_trues is not None:
            v_true=v_trues[i].reshape(v_trues[i].size,1)
        else:
            v_true = None
        Lx = Ls[i][:,0];Ly = Ls[i][:,1];Lt = Ls[i][:,2]

        ux_uy = scipy.linalg.block_diag(*[np.array([Lx[i], Ly[i]]) for i in range(nx_*ny_)])
        ut = Lt
        (v_est, info) = MMGKS_2(ux_uy, -ut.reshape((len(ut),1)), Lv, pnorm=pnorm, qnorm=qnorm, projection_dim=proj_dim, n_iter=n_iter, regparam='gcv',
                        x_true=v_true, tqdm_ = False)
        v_est = v_est.reshape(nx_*ny_,2)
        v_large = np.zeros((nx,nx,2))
        block_size = scale
        # Iterate over each 2x2 block
        k=0
        for m in range(0, nx - block_size + 1, block_size):
            for n in range(0, ny - block_size + 1, block_size):
                # Extract the current 2x2 block
                v_large[m:m+block_size, n:n+block_size] = v_est[k]
                k+=1

        #v_est = v_large

        v_est = v_est.reshape((nx_,ny_,2))

        v_large = v_large.reshape((nx,ny,2))*scale

        v_ests.append(v_est)

        v_larges.append(v_large)
        ux_uy_history.append(ux_uy)
        ut_history.append(ut)
    return (v_ests, v_larges, info, ux_history, uy_history, ux_uy_history, ut_history)


def solve_opt_flow_joint(u_traj,shape,t_end,v_true,v_max = 2, n_iter = 60):
    '''Solves the regularized optical flow equation'''
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


    Ls = []
    for i in range(len(u_traj)-1):

        u_traj[i] = u_traj[i].reshape(shape)
        u_traj[i+1] = u_traj[i+1].reshape(shape)

        uy = -vec(signal.convolve2d(u_traj[i], kernel_x, boundary='symm', mode='same'))/(delta_x*2*kernel_x.shape[0])
        ux = -vec(signal.convolve2d(u_traj[i], kernel_y, boundary='symm', mode='same'))/(delta_y*2*kernel_y.shape[0])

        ut = vec(signal.convolve2d(u_traj[i+1], kernel_t, boundary='symm', mode='same') - signal.convolve2d(u_traj[i], kernel_t, boundary='symm', mode='same'))/(kernel_t.size)
        Li = []
        for i in range(nx*ny):
            Li.append((ux[i],uy[i],ut[i]))
        Li = np.array(Li)
        Ls.append(Li)
    #print('Done')
    ux_uys = []
    for i in range(len(Ls)):
        Lx = Ls[i][:,0];Ly = Ls[i][:,1];Lt = Ls[i][:,2]

        ux_uy = scipy.linalg.block_diag(*[np.array([Lx[i], Ly[i]]) for i in range(size**2)])

        ux_uys.append(ux_uy)
    uts = [Ls[i][:,2] for i in range(len(Ls))] #[u_traj[i+1] - u_traj[i] for i in range(len(Ls))]#

    # L = gen_first_derivative_operator_2D(nx,ny)
    L = gen_2D(nx,ny)
    a=np.zeros((L.shape[0],2*L.shape[1]))
    from scipy.sparse import csr_matrix
    a = csr_matrix((L.shape[0],2*L.shape[1]))#.toarray()
    a[:,::2] = L
    b=np.zeros((L.shape[0],2*L.shape[1]))
    b = csr_matrix((L.shape[0],2*L.shape[1]))#.toarray()
    b[:,1::2] = L
    Lv =  sparse.vstack((a,b))


    ux_uy_bar = scipy.sparse.block_diag([ux_uys[i] for i in range(t_end-1)])#.toarray()

    Lv_bar = scipy.sparse.block_diag([Lv for i in range(t_end-1)])#.toarray()
    ut_bar = vectorize_func(np.array(uts))

    if v_true is not None:
        v_true = vec(np.array([v.reshape(v.size) for v in v_true])).reshape((vec(np.array([v.reshape(v.size) for v in v_true])).size,1))

    # print(np.isnan(ux_uy_bar.toarray()).any())
    (v_ests_, info) = MMGKS(ux_uy_bar, -ut_bar.reshape((len(ut_bar),1)), Lv_bar, pnorm=2, qnorm=1, projection_dim=1, n_iter=n_iter, regparam='gcv',
                        x_true=v_true, tqdm_ = False)

    return ([np.rint(v_ests_)[(len(v_ests_)//(t_end-1))*t:(len(v_ests_)//(t_end-1)*(t+1))].reshape(nx,ny,2) for t in range(t_end-1)],info)



def solve_opt_flow_new(u_traj,shape,t_end,v_trues,v_max = 2, n_iter = 60,reduction = False,**kwargs):
    '''Solves the regularized optical flow equation'''
    if (reduction == True):
        scale = kwargs['scale'] if ('scale' in kwargs) else 2
    else:
        scale = 1

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
    u_traj = deepcopy(u_traj)

    u_traj_  = deepcopy(u_traj)
    Ls = []

    for i in range(len(u_traj)-1):

            # Reshape the image array into a 4D array where each element is a 2x2 block
        blocks = u_traj_[i].reshape(nx//scale, scale, ny//scale, scale)

        # Take the mean along the last two axes to get the average of each 2x2 block
        u_traj[i] = scipy.ndimage.zoom(u_traj_[i].reshape(shape),scale) # np.mean(blocks, axis=(1, 3))

        blocks = u_traj_[i+1].reshape(nx//scale, scale, ny//scale, scale)

        # Take the mean along the last two axes to get the average of each 2x2 block
        u_traj[i+1] = scipy.ndimage.zoom(u_traj_[i].reshape(shape),scale)

        shape_ = u_traj[i].shape
        nx_=shape_[0]
        ny_=shape_[1]

        u_traj[i] = u_traj[i].reshape(shape_)
        u_traj[i+1] = u_traj[i+1].reshape(shape_)

        uy = -vec(signal.convolve2d(u_traj[i], kernel_x, boundary='symm', mode='same'))/(delta_x*2*kernel_x.shape[0])
        ux = -vec(signal.convolve2d(u_traj[i], kernel_y, boundary='symm', mode='same'))/(delta_y*2*kernel_y.shape[0])

        ut = vec(signal.convolve2d(u_traj[i+1], kernel_t, boundary='symm', mode='same') - signal.convolve2d(u_traj[i], kernel_t, boundary='symm', mode='same'))/(kernel_t.size)
        Li = []
        for i in range(nx_*ny_):
            Li.append((ux[i],uy[i],ut[i]))
        Li = np.array(Li)
        Ls.append(Li)

    L = gen_first_derivative_operator_2D(nx_,ny_)
    # a=np.zeros((L.[0],2*L.[1]))
    from scipy.sparse import csr_matrix
    a = csr_matrix((L.shape[0],2*L.shape[1]))#.toarray()
    a[:,::2] = L
    # b=np.zeros((L.[0],2*L.shape[1]))
    b = csr_matrix((L.shape[0],2*L.shape[1]))#.toarray()
    b[:,1::2] = L
    Lv =  sparse.vstack((a,b))

    v_ests = []
    v_larges =[]

    for i in range(len(Ls)):
        if v_trues is not None:
            v_true=v_trues[i].reshape(v_trues[i].size,1)
        else:
            v_true = None
        Lx = Ls[i][:,0];Ly = Ls[i][:,1];Lt = Ls[i][:,2]

        ux_uy = scipy.linalg.block_diag(*[np.array([Lx[i], Ly[i]]) for i in range(nx_*ny_)])
        ut = Lt
        (v_est, info) = MMGKS(ux_uy, -ut.reshape((len(ut),1)), Lv, pnorm=2, qnorm=1, projection_dim=1, n_iter=n_iter, regparam='gcv',
                        x_true=v_true, tqdm_ = False)
        v_est = v_est.reshape(nx_*ny_,2)
        v_large = np.zeros((nx,nx,2))
        block_size = scale
        # Iterate over each 2x2 block
        k=0
        for m in range(0, nx - block_size + 1, block_size):
            for n in range(0, ny - block_size + 1, block_size):
                # Extract the current 2x2 block
                v_large[m:m+block_size, n:n+block_size] = v_est[k]
                k+=1

        #v_est = v_large

        v_est = v_est.reshape((nx_,ny_,2))

        v_large = v_large.reshape((nx,ny,2))*scale

        v_ests.append(v_est)

        v_larges.append(v_large)

    return (v_ests, v_larges, info)


def solve_opt_flow_(u_traj,shape,t_end,v_trues,v_max = 2, n_iter = 60):
    '''Solves the regularized optical flow equation'''
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


    Ls = []
    for i in range(len(u_traj)-1):

        u_traj[i] = u_traj[i].reshape(shape)
        u_traj[i+1] = u_traj[i+1].reshape(shape)

        uy = -vec(signal.convolve2d(u_traj[i], kernel_x, boundary='symm', mode='same'))/(delta_x*2*kernel_x.shape[0])
        ux = -vec(signal.convolve2d(u_traj[i], kernel_y, boundary='symm', mode='same'))/(delta_y*2*kernel_y.shape[0])

        ut = vec(signal.convolve2d(u_traj[i+1], kernel_t, boundary='symm', mode='same') - signal.convolve2d(u_traj[i], kernel_t, boundary='symm', mode='same'))/(kernel_t.size)
        Li = []
        for i in range(nx*ny):
            Li.append((ux[i],uy[i],ut[i]))
        Li = np.array(Li)
        Ls.append(Li)

    L = gen_first_derivative_operator_2D(nx,ny)
    # a=np.zeros((L.shape[0],2*L.shape[1]))
    from scipy.sparse import csr_matrix
    a = csr_matrix((L.shape[0],2*L.shape[1]))#.toarray()
    a[:,::2] = L
    # b=np.zeros((L.shape[0],2*L.shape[1]))
    b = csr_matrix((L.shape[0],2*L.shape[1]))#.toarray()
    b[:,1::2] = L
    Lv =  sparse.vstack((a,b))

    v_ests = []

    for i in range(len(Ls)):
        if v_trues is not None:
            v_true=v_trues[i].reshape(v_trues[i].size,1)
        else:
            v_true = None
        Lx = Ls[i][:,0];Ly = Ls[i][:,1];Lt = Ls[i][:,2]

        ux_uy = scipy.linalg.block_diag(*[np.array([Lx[i], Ly[i]]) for i in range(size**2)])
        ut = Lt
        (v_est, info) = MMGKS(ux_uy, -ut.reshape((len(ut),1)), Lv, pnorm=2, qnorm=1, projection_dim=1, n_iter=n_iter, regparam='gcv',
                        x_true=v_true, tqdm_ = False)

        v_est = v_est.reshape((nx,ny,2))

        v_ests.append(v_est)

    return (v_ests, info)
