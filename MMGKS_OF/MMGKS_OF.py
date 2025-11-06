from utilities.imports import *
from optical_flow_solver import *
from utilities.weights import *
#from optical_flow_solver import solve_opt_flow

#from optical_flow_solver import solve_opt_flow
from scipy.sparse import csr_matrix

def MMGKS2(A, b, L, pnorm=2, qnorm=1, projection_dim=3, n_iter=5, regparam='gcv', x0 = None, x_true=None, power = 1, tqdm_ = True,**kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    isoTV_option = kwargs['isoTV'] if ('isoTV' in kwargs) else False
    GS_option = kwargs['GS'] if ('GS' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    prob_dims = kwargs['prob_dims'] if ('prob_dims' in kwargs) else False
    non_neg = kwargs['non_neg'] if ('non_neg' in kwargs) else False
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]
    (U, B, V) = golub_kahan_2(A, b, projection_dim, dp_stop,tqdm_, **kwargs)
    
    x_history = []
    lambda_history = []
    residual_history = []
    e = 1
    if x0 is not None:
        x = x0
    else:
        x = A.T @ b # initialize x for reweighting
    AV = A@V
    if GS_option in  ['GS', 'gs', 'Gs']:
        nx = prob_dims[0]
        ny = prob_dims[1]
        nt = prob_dims[2]
        Ls = generate_first_derivative_operator_2d_matrix(nx, ny)
        # Ls = first_derivative_operator_2d(nx, ny)
        L = sparse.kron(sparse.identity(nt), Ls)
        LV = L@V
    else:
        LV = L@V
    if (tqdm_ == True):
        range_ = tqdm(range(n_iter), desc='running MMGKS...')
    else:
        range_ = range(n_iter)
    for ii in range_:
        v = A @ x - b
        wf = (v**2 + epsilon**2)**(pnorm/2 - 1)
        AA = AV*(wf**power)
        (Q_A, R_A) = la.qr(AA, mode='economic') 
        u = L @ x
        if isoTV_option in ['isoTV', 'ISOTV', 'IsoTV']:
            if prob_dims == False:
                raise TypeError("For Isotropic TV you must enter the dimension of the dynamic problem! Example: (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 'gcv', x_true = None, isoTV = 'isoTV', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            Ls = first_derivative_operator_2d(nx, ny)
            nt = int((x.reshape((-1,1)).shape[0])/(nx*ny))
            spacen = int(Ls.shape[0] / 2)
            spacent = spacen * nt
            X = x.reshape(nx**2, nt)
            LsX = Ls @ X
            LsX1 = LsX[:spacen, :]
            LsX2 = LsX[spacen:2*spacen, :]
            weightx = (LsX1**2 + LsX2**2 + epsilon**2)**((qnorm-2) / 4)
            weightx = np.concatenate((weightx.flatten(), weightx.flatten()))
            weightt = (u[2*spacent:]**2 + epsilon**2)**((qnorm-2) / 4)
            wr = np.concatenate((weightx.reshape(-1,1), weightt))
        elif GS_option in  ['GS', 'gs', 'Gs']:
            if prob_dims == False:
                raise TypeError("For Isotropic Group Sparsity you must enter the dimension of the dynamic problem. (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 'gcv', x_true = None, GS = 'GS', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            nt = int((x.reshape((-1,1)).shape[0])/(nx*ny))
            utemp = np.reshape(x, (nx*ny, nt))
            Dutemp = Ls.dot(utemp)
            wr = np.exp(2) * np.ones((2*nx*(ny-1), 1))
            for i in range(2*nx*(ny-1)):
                wr[i] = (np.linalg.norm(Dutemp[i,:])**2 + wr[i])**(qnorm/2-1)
            wr = np.kron(np.ones((nt, 1)), wr)
        else:
            wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))

        LL = LV * (wr**power)
        (Q_L, R_L) = la.qr(LL, mode='economic') 
        if regparam == 'gcv':
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, (wf**power) *b, **kwargs)
        elif regparam == 'gcv_tol':
            lambdah = generalized_crossvalidation_tol(Q_A, R_A, R_L, (wf**power) *b, **kwargs)
        elif regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, (wf**power) *b, **kwargs)
        else:
            lambdah = regparam
        
        lambda_history.append(lambdah)
        y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), 
                        np.concatenate((Q_A.T@ ((wf**power)*b), np.zeros((R_L.shape[0],1)))),rcond=None)
        x = V @ y
        if (non_neg):
            x[x<0] = 0
        x_history.append(x)
        if ii >= R_L.shape[0]:
            break
        v = AV@y - b
        u = LV @ y
        ra = wf * (AV @ y - b)
        ra = A.T @ ra
        rb = wr * (LV @ y)
        rb = L.T @ rb
        r = ra + lambdah * rb
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)
        normed_r = r / la.norm(r) 
        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))
        Lvn = vn
        Lvn = L*vn
        LV = np.column_stack((LV, Lvn))
        residual_history.append(la.norm(r))
        
    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'Residual': residual_history, 'its': ii}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}
    
    return (x, info)

def MMGKS_OF(A, b, L_,I, t_end,shape,pnorm=2, qnorm=1, rnorm= 1, projection_dim=3, n_iter=5, n_iter_b=60,
regparam='gcv', regparam_of = "gcv", vs_true = None, v_primes_true = None, v_max = None,x0=None, x_true=None, pnorm_opt=1,qnorm_opt = 1, 
proj_dim_of = 1, interval=1,power=1,two_way=True,of_solver=solve_of,sigma_of=1,motion_mat="M_mat", centered_ut_of = True,
power_of=0.5,epsilon_of=0.01,v_scale_of ="cv2_pyrup",**kwargs):
    def M_mat(u, v):
        nx, ny = u.shape
        R = np.array(list(np.ndindex(nx, ny)))  # 2D array of pixel indices
        v_flat = v.reshape(-1, 2)  # Flatten v to match R

        # Calculate new indices based on v displacement
        new_ind = (v_flat + R).astype(int)
        new_ind[:, 0] = np.clip(new_ind[:, 0], 0, nx - 1)
        new_ind[:, 1] = np.clip(new_ind[:, 1], 0, ny - 1)

        # Create v_prime with displacements
        v_prime = np.zeros((nx, ny, 2))
        v_prime[new_ind[:, 0], new_ind[:, 1]] = -v_flat

        # Flatten indices to create sparse matrix
        inds = new_ind[:, 0] * ny + new_ind[:, 1]
        rows = np.arange(nx * ny)
        cols = inds
        data = np.ones(nx * ny)

    # Create sparse matrix M_ (CSR format)
        M_ = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()

        return M_, v_prime

    nx = shape[0];ny = shape[1]

    scale = kwargs['scale'] if ('scale' in kwargs) else 2
    reduction = kwargs['reduction'] if ('reduction' in kwargs) else False
    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    isoTV_option = kwargs['isoTV'] if ('isoTV' in kwargs) else False
    GS_option = kwargs['GS'] if ('GS' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    prob_dims = kwargs['prob_dims'] if ('prob_dims' in kwargs) else False
    parallel_of = kwargs['parallel_of'] if ('parallel_of' in kwargs) else False
    n_jobs = kwargs['n_jobs'] if ('n_jobs' in kwargs) else 20
    non_neg = kwargs['non_neg'] if ('non_neg' in kwargs) else False
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]
    (U, B, V) = golub_kahan_2(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    alpha_history = []
    residual_history = []
    e = 1

    if x0 is not None:
        x = x0
    else:
        x = A.T @ b # initialize x for reweighting


    if GS_option in  ['GS', 'gs', 'Gs']:
        nx = prob_dims[0]
        ny = prob_dims[1]
        nt = prob_dims[2]
        Ls = generate_first_derivative_operator_2d_matrix(nx, ny)
        # Ls = first_derivative_operator_2d(nx, ny)
        L_ = sparse.kron(sparse.identity(nt), Ls)

    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        # compute reweighting for p-norm approximation
        #print(la.norm(x-A.T @ b))
        v = A @ x - b
        x_ = x.reshape((-1,))
        #print(x_.shape)
        u = L_@x
        len_=nx*ny
        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]
        if ((ii%interval == 0) or ii==n_iter-1):
            if vs_true is not None: 
                v_ests = vs_true 
            else: 
                if (ii==1000):
                    v_ests = np.zeros((t_end-1, *shape,2))
                else:
                    _,v_ests,_ = of_solver(x_traj,shape=shape,t_end=t_end,v_trues=None,v_max=v_max,n_iter=n_iter_b,reduction = reduction, 
                                    scale = scale,pnorm=pnorm_opt,qnorm=qnorm_opt,proj_dim=proj_dim_of,regparam =regparam_of, parallel = parallel_of,n_jobs= n_jobs,
                                    sigma=sigma_of,centered_ut=centered_ut_of,power=power_of,epsilon=epsilon_of,v_scale =v_scale_of) #solve_opt_flow(x_traj,shape,t_end)
                    v_ests = np.array(v_ests)

                Ms = []
                v_primes_=[]
            if motion_mat == "M_mat":
                #v_ests = np.rint(np.array(v_ests))
                for t in range (t_end-1): 
                    M_v_prime =   M_mat(im_func(x_[(t_end - t -1)*(x_.shape[0]//t_end):(t_end - t)*(x_.shape[0]//t_end)],shape), np.rint(np.array(v_ests))[::-1][t].reshape((nx,ny,2)))
                    Ms.append(M_v_prime[0])
                    v_primes_.append(M_v_prime[1])
                Ms.reverse()

                x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]

                if v_primes_true is not None: 
                    v_inv_ests = v_primes_true
                else: 
                    v_inv_ests = v_primes_ #solve_opt_flow_b(x_inv_traj,shape,t_end,None,v_max,n_iter_b) #solve_opt_flow(x_inv_traj,shape,t_end)
                    v_inv_ests = np.rint(np.array(v_inv_ests))

                M_primes = [x_[0:1*(x_.shape[0]//t_end)]]
                M_primes = []
                for t in range (t_end-1):   
                    M_primes.append(M_mat(im_func(x_[t*(x_.shape[0]//t_end):(t+1)*(x_.shape[0]//t_end)],shape), v_inv_ests[::-1][t].reshape((nx,ny,2)))[0])

            elif motion_mat == "interp_mat":
                for t in range (t_end-1): 
                    M =  build_interpolation_matrix(v_ests[::-1][t].reshape((nx,ny,2)),shape)
                    Ms.append(M)
                    v_primes_.append(-v_ests[::-1][t].reshape((nx,ny,2)))
                Ms.reverse()

                x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]

                if v_primes_true is not None: 
                    v_inv_ests = v_primes_true
                else: 
                    v_inv_ests = v_primes_ #solve_opt_flow_b(x_inv_traj,shape,t_end,None,v_max,n_iter_b) #solve_opt_flow(x_inv_traj,shape,t_end)
                    v_inv_ests = np.array(v_inv_ests)

                M_primes = [x_[0:1*(x_.shape[0]//t_end)]]
                M_primes = []
                for t in range (t_end-1):   
                    M_primes.append(build_interpolation_matrix(v_inv_ests[::-1][t].reshape((nx,ny,2)),shape))

            I1 = scipy.sparse.identity(nx*ny)

            M_top = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [I1, -Ms[i]] + (len(Ms)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(Ms))]
            M_bottom = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [-M_primes[i],I1] + (len(M_primes)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(M_primes))]

            if (two_way == True):
                M = scipy.sparse.vstack((*M_top,*M_bottom))
            else:
                M = scipy.sparse.vstack(M_top)
            M = scipy.sparse.csr_matrix(M)

        #L = scipy.sparse.vstack((L_,M))

        if isoTV_option in ['isoTV', 'ISOTV', 'IsoTV']:
            if prob_dims == False:
                raise TypeError("For Isotropic TV you must enter the dimension of the dynamic problem! Example: (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 'gcv', x_true = None, isoTV = 'isoTV', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            Ls = first_derivative_operator_2d(nx, ny)
            nt = int((x.reshape((-1,1)).shape[0])/(nx*ny))
            spacen = int(Ls.shape[0] / 2)
            spacent = spacen * nt
            X = x.reshape(nx**2, nt)
            LsX = Ls @ X
            LsX1 = LsX[:spacen, :]
            LsX2 = LsX[spacen:2*spacen, :]
            weightx = (LsX1**2 + LsX2**2 + epsilon**2)**((qnorm-2) / 4)
            weightx = np.concatenate((weightx.flatten(), weightx.flatten()))
            weightt = (u[2*spacent:]**2 + epsilon**2)**((qnorm-2) / 4)
            wr = np.concatenate((weightx.reshape(-1,1), weightt))
        elif GS_option in  ['GS', 'gs', 'Gs']:
            if prob_dims == False:
                raise TypeError("For Isotropic Group Sparsity you must enter the dimension of the dynamic problem. (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 'gcv', x_true = None, GS = 'GS', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            nt = int((x.reshape((-1,1)).shape[0])/(nx*ny))
            utemp = np.reshape(x, (nx*ny, nt))
            Dutemp = Ls.dot(utemp)
            wr = np.exp(2) * np.ones((2*nx*(ny-1), 1))
            for i in range(2*nx*(ny-1)):
                wr[i] = (np.linalg.norm(Dutemp[i,:])**2 + wr[i])**(qnorm/2-1)
            wr = np.kron(np.ones((nt, 1)), wr)
        else:
            wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))
        AV = A@V
        LV_ = L_@V
        MV = M@V

        v = A @ x - b
        u = L_@x
        z= M@x
    
        wf = ((v**2 + epsilon**2)**(pnorm/2 - 1))
        AA = AV*(wf**power)
        (Q_A, R_A) = la.qr(AA, mode='economic')
        
        wm = ((z**2 + epsilon**2)**(rnorm/2 - 1))
        MM = MV * (wm**power)
        
        LL_ = LV_ * (wr**power)
        
        LL = np.concatenate((LL_,MM))
        (Q_L, R_L) = la.qr(LL, mode='economic') 
        #print(la.norm(wm),la.norm(wr))
            
        if regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, (wf**power) *b, **kwargs)
        elif regparam == 'gcv':
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, (wf**power) *b, **kwargs)
        elif regparam == 'gcv_tol':
            lambdah = generalized_crossvalidation_tol(Q_A, R_A, R_L, (wf**power) *b, **kwargs)
        else:
            lambdah = regparam
        lambda_history.append(lambdah)
        y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ ((wf**power)* b), np.zeros((R_L.shape[0],1)))),rcond=None)

        x = V @ y
        
        if (non_neg):
            x[x<0] = 0
        x_history.append(x)
        if ii >= R_L.shape[0]:
            break

        ra = wf * (AV @ y - b)
        ra = A.T @ ra
        rb = wr * (LV_ @ y)
        rb = L_.T @ rb
        rc = wm * (MV @ y)
        rc = M.T @ rc


        r = ra + lambdah * (rb + rc)
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)

        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))

        Lvn = L_*vn
        LV_ = np.column_stack((LV_, Lvn))
        Mvn = M*vn
        MV = np.column_stack((MV, Mvn))
        residual_history.append(la.norm(r))


    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history,'regParam2_history': alpha_history, 
                'relError': rre_history, 'Residual': residual_history, 'its': ii,'Ms':Ms,'M_primes':M_primes}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii,'Ms':Ms,'M_primes':M_primes}
    
    return (x, info, v_ests, v_inv_ests)

def build_interpolation_matrix(v, shape):
    v=v[...,[1,0]]
    h, w = shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Calculate the source coordinates
    x_src = x + v[:,:,0]
    y_src = y + v[:,:,1]
    
    # Clip the coordinates to be within the image bounds
    x_src = np.clip(x_src, 0, w-1)
    y_src = np.clip(y_src, 0, h-1)
    
    # Calculate the four nearest integer coordinate points
    x0 = np.floor(x_src).astype(int)
    x1 = np.ceil(x_src).astype(int)
    y0 = np.floor(y_src).astype(int)
    y1 = np.ceil(y_src).astype(int)
    
    # Calculate the weights for bilinear interpolation
    wx1 = x_src - x0
    wx0 = 1 - wx1
    wy1 = y_src - y0
    wy0 = 1 - wy1
    
    # Create the sparse matrix
    i = np.arange(h*w)
    j00 = y0 * w + x0
    j01 = y0 * w + x1
    j10 = y1 * w + x0
    j11 = y1 * w + x1
    
    data = np.concatenate([
        (wx0 * wy0).flatten(),
        (wx1 * wy0).flatten(),
        (wx0 * wy1).flatten(),
        (wx1 * wy1).flatten()
    ])
    row = np.concatenate([i, i, i, i])
    col = np.concatenate([j00.flatten(), j01.flatten(), j10.flatten(), j11.flatten()])
    
    M = csr_matrix((data, (row, col)), shape=(h*w, h*w))
    return M