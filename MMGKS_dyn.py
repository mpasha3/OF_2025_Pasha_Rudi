from imports import *
from optical_flow_solver import *
from optical_flow_solver import solve_opt_flow

def MMGKS_dyn_joint_old(A, b, L,I, t_end,shape,pnorm=2, qnorm=1, rnorm= 1, projection_dim=3, n_iter=5, n_iter_b=60,
regparam='gcv', vs_true = None, v_primes_true = None, v_max = None, x_true=None, **kwargs):

    def M(u,v):
        R = np.array(list(np.ndindex(*u.shape)))
        nx = u.shape[0]
        ny = u.shape[1]
        new_ind = (v.reshape(nx*ny,2)+R).astype(int)
        new_ind[new_ind>nx-1] = nx-1
        new_ind[new_ind<0] = 0
        v_prime=np.zeros((nx,ny,2))
        v_prime[new_ind[:,0],new_ind[:,1]] = -v.reshape(nx*ny,2)
        return np.array([u[tuple(r)]  for r in new_ind]),v_prime

    nx = shape[0];ny = shape[1]

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    isoTV_option = kwargs['isoTV'] if ('isoTV' in kwargs) else False
    GS_option = kwargs['GS'] if ('GS' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    prob_dims = kwargs['prob_dims'] if ('prob_dims' in kwargs) else False
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]
    scale = kwargs['scale'] if ('scale' in kwargs) else 2
    reduction = kwargs['reduction'] if ('reduction' in kwargs) else False
    (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    alpha_history = []
    residual_history = []
    e = 1
    x = A.T @ b # initialize x for reweighting

    AV = A@V
    LV = L@V
    IV = I@V

    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        # compute reweighting for p-norm approximation
        v = A @ x - b

        x_ = x.reshape((-1,))

        len_=nx*ny
        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]

        if vs_true is not None:
            v_ests = vs_true

        else:
            _,v_ests,_ = solve_opt_flow(x_traj,shape=shape,t_end=t_end,v_trues=None,v_max=v_max,n_iter=n_iter_b,reduction = reduction, scale = scale) #solve_opt_flow(x_traj,shape,t_end)
            v_ests = (np.array(v_ests))

        M_x=[x_[(t_end-1)*(x_.shape[0]//t_end):t_end*(x_.shape[0]//t_end)]]
        v_primes_=[]
        for t in range (t_end-1):
            M_x_v_prime =   M(im_func(x_[(t_end - t -1)*(x_.shape[0]//t_end):(t_end - t)*(x_.shape[0]//t_end)],shape), v_ests[::-1][t].reshape((nx,ny,2)))
            M_x.append(M_x_v_prime[0])
            v_primes_.append(M_x_v_prime[1])
        M_x.reverse()

        M_x = np.array(M_x)
        M_X = vectorize_func(M_x).reshape((-1,1))


        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]
        x_inv_traj = x_traj[::-1]
        if v_primes_true is not None:
            v_inv_ests = v_primes_true
        else:
            v_inv_ests = v_primes_ #solve_opt_flow_b(x_inv_traj,shape,t_end,None,v_max,n_iter_b) #solve_opt_flow(x_inv_traj,shape,t_end)
            v_inv_ests = (np.array(v_inv_ests))

        vel_shape = (nx,ny, 2)
        M_inv_x=[x_[0:1*(x_.shape[0]//t_end)]]
        for t in range (t_end-1):
            M_inv_x.append(M(im_func(x_[t*(x_.shape[0]//t_end):(t+1)*(x_.shape[0]//t_end)],shape), v_inv_ests[::-1][t].reshape((nx,ny,2)))[0])

        M_inv_x = np.array(M_inv_x)
        M_inv_X = vectorize_func(M_inv_x).reshape((-1,1))


        z = I@x - M_X

        z_inv = I@x - M_inv_X

        wf = ((v**2 + epsilon**2)**(pnorm/2 - 1))**(1/2)
        wm = ((z**2 + epsilon**2)**(rnorm/2 - 1))**(1/2)
        wm_inv = ((z_inv**2 + epsilon**2)**(rnorm/2 - 1))**(1/2)

        AA = AV*wf
        II = IV*wm
        II_inv = IV*wm_inv

        (Q_A, R_A) = la.qr(AA, mode='economic') # Project A into V, separate into Q and R
        (Q_I, R_I) = la.qr(II, mode='economic') # Project I into V, separate into Q and R
        (Q_I_inv, R_I_inv) = la.qr(II_inv, mode='economic') # Project I into V, separate into Q and R
        # Compute reweighting for q-norm approximation

        u = L @ x
        wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))**(1/2)
        LL = LV * wr
        (Q_L, R_L) = la.qr(LL, mode='economic') # Project L into V, separate into Q and R



        LLIIII = np.concatenate((LL,II,II_inv))
        (Q_LII, R_LII) = la.qr(LLIIII, mode='economic')

        b_ = np.concatenate((np.zeros((LL.shape[0],1)), wm* M_X, wm_inv* M_inv_X))

        if regparam == 'gcv':
            lambdah = generalized_crossvalidation_ext(Q_A, R_A, Q_LII, R_LII, b, b_, **kwargs)#['x'].item() # find ideal lambda by crossvalidation
            alpha = lambdah
            beta = lambdah

        lambda_history.append(lambdah)

        y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L, np.sqrt(alpha) * R_I
                                                , np.sqrt(beta) * R_I_inv)),
                  np.concatenate((Q_A.T@ (wf* b), np.zeros((R_L.shape[0],1)),np.sqrt(alpha)*Q_I.T@ (wm* M_X),
                                 np.sqrt(beta)*Q_I_inv.T@ (wm_inv* M_inv_X))),rcond=None)
        x = V @ y # project y back

        x_history.append(x)

        if (ii >= R_L.shape[0]):
            break
        v = AV@y
        v = v - b
        u = LV @ y

        z = IV@y
        z = z - M_X

        z_inv = IV@y
        z_inv = z_inv - M_inv_X

        ra = (wf**2) * (AV @ y - b)
        ra = A.T @ ra
        rb = (wr**2) * (LV @ y)
        rb = L.T @ rb
        rc = (wm**2) * (IV@y - M_X)
        rd = (wm_inv**2) * (IV@y - M_inv_X)
        r = ra + lambdah * rb + alpha*rc + beta*rd
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)

        normed_r = r / la.norm(r) # normalize residual
        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))

        Ivn = I @ vn
        IV = np.column_stack((IV, Ivn))

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

    return (x, info, v_ests, v_inv_ests)




def MMGKS_dyn(A, b, L_,I, t_end,shape,pnorm=2, qnorm=1, rnorm= 1, projection_dim=3, n_iter=5, n_iter_b=60,
regparam='gcv', vs_true = None, v_primes_true = None, v_max = None, x_true=None, qnorm_opt = 1, **kwargs):

    def M(u,v):
        R = np.array(list(np.ndindex(*u.shape)))
        nx = u.shape[0]
        ny = u.shape[1]
        new_ind = (v.reshape(nx*ny,2)+R).astype(int)
        new_ind[new_ind>nx-1] = nx-1
        new_ind[new_ind<0] = 0
        v_prime=np.zeros((nx,ny,2))
        v_prime[new_ind[:,0],new_ind[:,1]] = -v.reshape(nx*ny,2)
        return np.array([u[tuple(r)]  for r in new_ind]),v_prime

    nx = shape[0];ny = shape[1]

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    isoTV_option = kwargs['isoTV'] if ('isoTV' in kwargs) else False
    GS_option = kwargs['GS'] if ('GS' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    prob_dims = kwargs['prob_dims'] if ('prob_dims' in kwargs) else False
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]
    scale = kwargs['scale'] if ('scale' in kwargs) else 2
    reduction = kwargs['reduction'] if ('reduction' in kwargs) else False
    (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    alpha_history = []
    residual_history = []
    e = 1
    x = A.T @ b # initialize x for reweighting
    L = scipy.sparse.vstack((L_,I,I)) #np.concatenate((L_@I.toarray(),I.toarray(),I.toarray()))
    # L = scipy.sparse.csr_matrix(L)

    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        # compute reweighting for p-norm approximation
        v = A @ x - b

        x_ = x.reshape((-1,))

        len_=nx*ny
        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]

        if vs_true is not None:
            v_ests = vs_true

        else:
            _,v_ests,_ = solve_opt_flow(x_traj,shape=shape,t_end=t_end,v_trues=None,v_max=v_max,n_iter=n_iter_b,reduction = reduction, scale = scale, qnorm = qnorm_opt) #solve_opt_flow(x_traj,shape,t_end)
            v_ests = (np.array(v_ests))
        M_x=[x_[(t_end-1)*(x_.shape[0]//t_end):t_end*(x_.shape[0]//t_end)]]
        v_primes_=[]
        for t in range (t_end-1):
            M_x_v_prime =   M(im_func(x_[(t_end - t -1)*(x_.shape[0]//t_end):(t_end - t)*(x_.shape[0]//t_end)],shape), v_ests[::-1][t].reshape((nx,ny,2)))
            M_x.append(M_x_v_prime[0])
            v_primes_.append(M_x_v_prime[1])
        M_x.reverse()

        M_x = np.array(M_x)
        M_X = vectorize_func(M_x).reshape((-1,1))


        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]
        x_inv_traj = x_traj[::-1]
        if v_primes_true is not None:
            v_inv_ests = v_primes_true
        else:
            v_inv_ests = v_primes_ #solve_opt_flow_b(x_inv_traj,shape,t_end,None,v_max,n_iter_b) #solve_opt_flow(x_inv_traj,shape,t_end)
            v_inv_ests = (np.array(v_inv_ests))

        vel_shape = (nx,ny, 2)
        M_inv_x=[x_[0:1*(x_.shape[0]//t_end)]]
        for t in range (t_end-1):
            M_inv_x.append(M(im_func(x_[t*(x_.shape[0]//t_end):(t+1)*(x_.shape[0]//t_end)],shape), v_inv_ests[::-1][t].reshape((nx,ny,2)))[0])

        M_inv_x = np.array(M_inv_x)
        M_inv_X = vectorize_func(M_inv_x).reshape((-1,1))

        b_ = np.concatenate((np.zeros((L_.shape[0],1)),M_X,M_inv_X))

        AV = A@V
        LV = L@V

        v = A @ x - b

        # print(x.shape,L.shape,b_.shape)
        u = L@x - b_


        wf = ((v**2 + epsilon**2)**(pnorm/2 - 1))


        AA = AV*(wf**(1/2))


        (Q_A, R_A) = la.qr(AA, mode='economic') # Project A into V, separate into Q and R

        # Compute reweighting for q-norm approximation

        u = L @ x
        if isoTV_option in ['isoTV', 'ISOTV', 'IsoTV']:
            if prob_dims == False:
                raise TypeError("For Isotropic TV you must enter the dimension of the dynamic problem! Example: (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 0.005, x_true = None, isoTV = 'isoTV', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            #### This are the same weights as in utilities.weights
            nt = int((x.reshape((-1,1)).shape[0])/(nx*ny))
            Ls = gen_first_derivative_operator_2D(nx, ny)
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
            # print(wr.shape)
            ######
        elif GS_option in  ['GS', 'gs', 'Gs']:
            if prob_dims == False:
                raise TypeError("For Isotropic TV you must enter the dimension of the dynamic problem. (x_mmgks, info_mmgks) = MMGKS(A, data_vec, L, pnorm=2, qnorm=1, projection_dim=2, n_iter =3, regparam = 0.005, x_true = None, isoTV = 'isoTV', prob_dims = (nx,ny, nt))")
            else:
                nx = prob_dims[0]
                ny = prob_dims[1]
            wr = GS_weights(x, nx, ny, epsilon, qnorm)
        else:
            wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))

        LL = LV * (wr**1)
        (Q_L, R_L) = la.qr(LL, mode='economic') # Project L into V, separate into Q and R

        # Compute the projected rhs
        bhat = (Q_A.T @ b).reshape(-1,1)



        if regparam == 'gcv':
            lambdah = generalized_crossvalidation_ext(Q_A, R_A, Q_L, R_L, b, b_, **kwargs)#['x'].item() # find ideal lambda by crossvalidation


        lambda_history.append(lambdah)

        y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)),
                  np.concatenate((Q_A.T@ ((wf**1)* b), np.sqrt(lambdah)*Q_L.T@ ((wr**1)* b_))),rcond=None)

        x = V @ y # project y back

        x_history.append(x)

        if (ii >= R_L.shape[0]):
            break
        v = AV@y
        v = v - b
        u = LV @ y
        u = u - b_

        ra = wf * (AV @ y - b)
        ra = A.T @ ra
        rb = (wr**1) * (LV @ y - b_)
        rb = L.T @ rb

        r = ra + lambdah * rb

        normed_r = r / la.norm(r) # normalize residual
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

    return (x, info, v_ests, v_inv_ests)


def MMGKS2(A, b, L, pnorm=2, qnorm=1, projection_dim=3, n_iter=5, regparam='gcv', x_true=None, min_l = 0,max_l=1,opt='nonscaled', non_neg= True, **kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False

    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    (U, B, V) = golub_kahan_2(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    residual_history = []
    e = 1
    x = A.T @ b
    AV = A@V
    LV = L@V


    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        # y= la.pinv(V)@x
        v = A @ x - b
        wf = (v**2 + epsilon**2)**(pnorm/2 - 1)


        AA = AV*wf
        (Q_A, R_A) = la.qr(AA, mode='economic')
        # g= np.linalg.lstsq(R_A,Q_A.T@b)[0]
        u =(L @ x) #*(1/la.norm(L@(V@g),1))

        wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))
        LL = LV * wr
        (Q_L, R_L) = la.qr(LL, mode='economic')
        if regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            f_max = 1
            g_max = 1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'gcv':
            lambdah = generalized_crossvalidation_2(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            f_max = 1
            g_max = 1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'new':

            x2 = np.linalg.lstsq(np.concatenate(((1-min_l)*R_A, (min_l)* R_L)),
                            np.concatenate(((1-min_l)*Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None) [0]

            x1 = np.linalg.lstsq(np.concatenate(((min_l)*R_A, (1-min_l)* R_L)),
                            np.concatenate(((min_l)*Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None) [0]


            f_max = la.norm(AA@x1-b)**2
            g_max = la.norm(LL@x2)**2

            AA = AA/np.sqrt(f_max)
            LL = LL/np.sqrt(g_max)
            b_=b/np.sqrt(f_max)

            (Q_A, R_A) = la.qr(AA, mode='economic')
            (Q_L, R_L) = la.qr(LL, mode='economic')

            lambdah = gg(AA,Q_A,R_A,b,LL,Q_L,R_L,np.zeros((L.shape[0],1)))
            w1 = 1
            w2 = lambdah**2
            y,_,_,_ = np.linalg.lstsq(np.concatenate((w1*R_A, (lambdah) * R_L)),
                        np.concatenate((w1*Q_A.T@ b_, (lambdah) *np.zeros((R_L.shape[0],1)))),rcond=None)
        else:
            lambdah = regparam
        lambda_history.append(lambdah)


        x = V @ y

        if (non_neg):
            x[x<0] = 0
        x_history.append(x)
        if ii >= R_L.shape[0]:
            break

        ra = wf * (AV @ y - b)/f_max
        ra = A.T @ ra
        rb = wr * (LV @ y)/g_max
        rb = L.T @ rb


        w1 = 1


        r = w1**2* ra + w2 * rb
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)

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


def MMGKS_dyn_joint2(A, b, L_,I, t_end,shape,pnorm=2, qnorm=1, rnorm= 1, projection_dim=3, n_iter=5, n_iter_b=60,
regparam='gcv', vs_true = None, v_primes_true = None, v_max = None, x_true=None, min_l = 0,max_l=1,opt='nonscaled', non_neg= True, qnorm_opt = 1, proj_dim_opt = 1,**kwargs):
    def M_mat(u,v):
        R = np.array(list(np.ndindex(*u.shape)))
        nx = u.shape[0]
        ny = u.shape[1]
        new_ind = (v.reshape(nx*ny,2)+R).astype(int)
        new_ind[new_ind>nx-1] = nx-1
        new_ind[new_ind<0] = 0
        v_prime=np.zeros((nx,ny,2))
        v_prime[new_ind[:,0],new_ind[:,1]] = -v.reshape(nx*ny,2)
        inds = [ind[0]*nx+ind[1] for ind in new_ind]
        rows = [i for i in range(len(inds))]
        cols = [inds[i] for i in range(len(inds))]
        data = [1 for i in range(len(inds))]
        M_ = scipy.sparse.coo_matrix((data, (rows,cols)),shape = (nx*ny,nx*ny))
        M_ = M_.tocsr()
        return M_,v_prime
    nx = shape[0];ny = shape[1]

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    scale = kwargs['scale'] if ('scale' in kwargs) else 2
    reduction = kwargs['reduction'] if ('reduction' in kwargs) else False
    (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    alpha_history = []
    residual_history = []
    e = 1
    x = A.T @ b # initialize x for reweighting

    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        # compute reweighting for p-norm approximation
        v = A @ x - b
        x_ = x.reshape((-1,))

        len_=nx*ny
        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]
        if vs_true is not None:
            v_ests = vs_true
        else:
            _,v_ests,_ = solve_opt_flow(x_traj,shape=shape,t_end=t_end,v_trues=None,v_max=v_max,n_iter=n_iter_b,reduction = reduction, scale = scale,qnorm=qnorm_opt,proj_dim=proj_dim_opt) #solve_opt_flow(x_traj,shape,t_end)
            v_ests = (np.array(v_ests))

        Ms = []
        v_primes_=[]
        for t in range (t_end-1):
            M_v_prime =   M_mat(im_func(x_[(t_end - t -1)*(x_.shape[0]//t_end):(t_end - t)*(x_.shape[0]//t_end)],shape), v_ests[::-1][t].reshape((nx,ny,2)))
            Ms.append(M_v_prime[0])
            v_primes_.append(M_v_prime[1])
        Ms.reverse()

        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]

        if v_primes_true is not None:
            v_inv_ests = v_primes_true
        else:
            v_inv_ests = v_primes_ #solve_opt_flow_b(x_inv_traj,shape,t_end,None,v_max,n_iter_b) #solve_opt_flow(x_inv_traj,shape,t_end)
            v_inv_ests = (np.array(v_inv_ests))

        M_primes = [x_[0:1*(x_.shape[0]//t_end)]]
        M_primes = []
        for t in range (t_end-1):
            M_primes.append(M_mat(im_func(x_[t*(x_.shape[0]//t_end):(t+1)*(x_.shape[0]//t_end)],shape), v_inv_ests[::-1][t].reshape((nx,ny,2)))[0])

        I1 = scipy.sparse.identity(nx*ny)

        M_top = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [I1, -Ms[i]] + (len(Ms)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(Ms))]
        M_bottom = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [-M_primes[i],I1] + (len(M_primes)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(M_primes))]

        M = scipy.sparse.vstack((*M_top,*M_bottom))
        M = scipy.sparse.csr_matrix(M)

        #L = scipy.sparse.vstack((L_,M))

        AV = A@V
        LV_ = L_@V
        MV = M@V

        v = A @ x - b
        u = L_@x
        z= M@x


        wf = ((v**2 + epsilon**2)**(pnorm/2 - 1))
        AA = AV*(wf**(1/2))
        (Q_A, R_A) = la.qr(AA, mode='economic')

        wm = ((z**2 + epsilon**2)**(rnorm/2 - 1))
        MM = MV * wm


        wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))
        LL_ = LV_ * wr

        LL = np.concatenate((LL_,MM))
        (Q_L, R_L) = la.qr(LL, mode='economic')


        if regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            f_max = 1
            g_max = 1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'gcv':
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            f_max = 1
            g_max = 1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'new':

            x2 = np.linalg.lstsq(np.concatenate(((1-min_l)*R_A, (min_l)* R_L)),
                            np.concatenate(((1-min_l)*Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None) [0]

            x1 = np.linalg.lstsq(np.concatenate(((min_l)*R_A, (1-min_l)* R_L)),
                            np.concatenate(((min_l)*Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None) [0]


            f_max = la.norm(AA@x1-b)**2
            g_max = la.norm(LL@x2)**2

            AA = AA/np.sqrt(f_max)
            LL = LL/np.sqrt(g_max)
            b_=b/np.sqrt(f_max)

            (Q_A, R_A) = la.qr(AA, mode='economic')
            (Q_L, R_L) = la.qr(LL, mode='economic')

            lambdah = gg(AA,Q_A,R_A,b,LL,Q_L,R_L,np.zeros((LL.shape[0],1)))
            w1 = 1
            w2 = lambdah**2
            y,_,_,_ = np.linalg.lstsq(np.concatenate((w1*R_A, (lambdah) * R_L)),
                        np.concatenate((w1*Q_A.T@ b_, (lambdah) *np.zeros((R_L.shape[0],1)))),rcond=None)
        else:
            lambdah = regparam
        lambda_history.append(lambdah)


        x = V @ y

        if (non_neg):
            x[x<0] = 0
        x_history.append(x)
        if ii >= R_L.shape[0]:
            break

        ra = wf * (AV @ y - b)/f_max
        ra = A.T @ ra
        rb = wr * (LV_ @ y)/g_max
        rb = L_.T @ rb
        rc = wm * (MV @ y)/g_max
        rc = M.T @ rc

        w1 = 1

        r = w1**2* ra + w2 * (rb + rc)
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)

        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))

        Lvn = L_*vn
        LV = np.column_stack((LV_, Lvn))
        Mvn = M*vn
        MV = np.column_stack((MV, Mvn))
        residual_history.append(la.norm(r))

    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history,'regParam2_history': alpha_history, 'relError': rre_history, 'Residual': residual_history, 'its': ii,'Ms':Ms}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}

    return (x, info, v_ests, v_inv_ests)


def MMGKS_dyn(A, b, L_,I, t_end,shape,pnorm=2, qnorm=1, rnorm= 1, projection_dim=3, n_iter=5, n_iter_b=60,
regparam='gcv', vs_true = None, v_primes_true = None, v_max = None, x_true=None, min_l = 0,max_l=1,opt='nonscaled', non_neg= True, qnorm_opt = 1, proj_dim_opt = 1,**kwargs):
    def M_mat(u,v):
        R = np.array(list(np.ndindex(*u.shape)))
        nx = u.shape[0]
        ny = u.shape[1]
        new_ind = (v.reshape(nx*ny,2)+R).astype(int)
        new_ind[new_ind>nx-1] = nx-1
        new_ind[new_ind<0] = 0
        v_prime=np.zeros((nx,ny,2))
        v_prime[new_ind[:,0],new_ind[:,1]] = -v.reshape(nx*ny,2)
        inds = [ind[0]*nx+ind[1] for ind in new_ind]
        rows = [i for i in range(len(inds))]
        cols = [inds[i] for i in range(len(inds))]
        data = [1 for i in range(len(inds))]
        M_ = scipy.sparse.coo_matrix((data, (rows,cols)),shape = (nx*ny,nx*ny))
        M_ = M_.tocsr()
        return M_,v_prime
    nx = shape[0];ny = shape[1]

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    scale = kwargs['scale'] if ('scale' in kwargs) else 2
    reduction = kwargs['reduction'] if ('reduction' in kwargs) else False
    (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    alpha_history = []
    residual_history = []
    e = 1
    x = A.T @ b # initialize x for reweighting

    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        # compute reweighting for p-norm approximation
        v = A @ x - b
        x_ = x.reshape((-1,))

        len_=nx*ny
        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]
        if vs_true is not None:
            v_ests = vs_true
        else:
            _,v_ests,_ = solve_opt_flow(x_traj,shape=shape,t_end=t_end,v_trues=None,v_max=v_max,n_iter=n_iter_b,reduction = reduction, scale = scale,qnorm=qnorm_opt,proj_dim=proj_dim_opt) #solve_opt_flow(x_traj,shape,t_end)
            v_ests = (np.array(v_ests))

        Ms = []
        v_primes_=[]
        for t in range (t_end-1):
            M_v_prime =   M_mat(im_func(x_[(t_end - t -1)*(x_.shape[0]//t_end):(t_end - t)*(x_.shape[0]//t_end)],shape), v_ests[::-1][t].reshape((nx,ny,2)))
            Ms.append(M_v_prime[0])
            v_primes_.append(M_v_prime[1])
        Ms.reverse()

        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]

        if v_primes_true is not None:
            v_inv_ests = v_primes_true
        else:
            v_inv_ests = v_primes_ #solve_opt_flow_b(x_inv_traj,shape,t_end,None,v_max,n_iter_b) #solve_opt_flow(x_inv_traj,shape,t_end)
            v_inv_ests = (np.array(v_inv_ests))

        M_primes = [x_[0:1*(x_.shape[0]//t_end)]]
        M_primes = []
        for t in range (t_end-1):
            M_primes.append(M_mat(im_func(x_[t*(x_.shape[0]//t_end):(t+1)*(x_.shape[0]//t_end)],shape), v_inv_ests[::-1][t].reshape((nx,ny,2)))[0])

        I1 = scipy.sparse.identity(nx*ny)

        M_top = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [I1, -Ms[i]] + (len(Ms)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(Ms))]
        M_bottom = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [-M_primes[i],I1] + (len(M_primes)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(M_primes))]

        M = scipy.sparse.vstack((*M_top,*M_bottom))
        M = scipy.sparse.csr_matrix(M)

        #L = scipy.sparse.vstack((L_,M))
        # if (ii==0):
        AV = A@V
        LV_ = L_@V
        MV = M@V

        v = A @ x - b
        u = L_@x
        z= M@x


        wf = ((v**2 + epsilon**2)**(pnorm/2 - 1))
        AA = AV*(wf**(1/2))
        (Q_A, R_A) = la.qr(AA, mode='economic')

        wm = ((z**2 + epsilon**2)**(rnorm/2 - 1))
        MM = MV * wm
        (Q_M, R_M) = la.qr(MM, mode='economic')

        wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))
        LL_ = LV_ * wr
        (Q_L_, R_L_) = la.qr(LL_, mode='economic')

        LL = np.concatenate((LL_,MM))
        (Q_L, R_L) = la.qr(LL, mode='economic')


        if regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            f_max = 1
            g_max = 1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'gcv':
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            w3= lambdah
            f_max = 1
            g_max = 1
            h_max =1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'new':

            x2 = np.linalg.lstsq(np.concatenate(((1-min_l)*R_A, (min_l)* R_L_)),
                            np.concatenate(((1-min_l)*Q_A.T@ b, np.zeros((R_L_.shape[0],1)))),rcond=None) [0]

            x3 = np.linalg.lstsq(np.concatenate(((1-min_l)*R_A, (min_l)* R_M)),
                            np.concatenate(((1-min_l)*Q_A.T@ b, np.zeros((R_M.shape[0],1)))),rcond=None) [0]

            x1 = np.linalg.lstsq(np.concatenate(((min_l)*R_A, (1-min_l)* R_L)),
                            np.concatenate(((min_l)*Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None) [0]

            f_max = la.norm(AA@x1-b)**2
            g_max = la.norm(LL_@x2)**2

            h_max = la.norm(MM@x3)**2

            AA = AA/np.sqrt(f_max)
            LL_ = LL_/np.sqrt(g_max)
            MM = MM/np.sqrt(h_max)
            b_=b/np.sqrt(f_max)

            (Q_A, R_A) = la.qr(AA, mode='economic')
            (Q_L_, R_L_) = la.qr(LL_, mode='economic')
            (Q_M, R_M) = la.qr(MM, mode='economic')

            lambdah = gg(AA,Q_A,R_A,b,LL_,Q_L_,R_L_,np.zeros((LL_.shape[0],1)))

            alpha = gg(AA,Q_A,R_A,b,MM,Q_M,R_M,np.zeros((MM.shape[0],1)))

            alpha_history.append(alpha)

            w1 = 1
            w2 = lambdah**2
            w3 = alpha**2


            y,_,_,_ = np.linalg.lstsq(np.concatenate((w1*R_A, (lambdah) * R_L_, alpha*R_M)),
                        np.concatenate((w1*Q_A.T@ b_, (lambdah) *np.zeros((R_L_.shape[0],1)), alpha*np.zeros((R_M.shape[0],1)))),rcond=None)
        else:
            lambdah = regparam
        lambda_history.append(lambdah)



        x = V @ y

        if (non_neg):
            x[x<0] = 0
        x_history.append(x)
        if ii >= R_L.shape[0]:
            break

        ra = wf * (AV @ y - b)/f_max
        ra = A.T @ ra
        rb = wr * (LV_ @ y)/g_max
        rb = L_.T @ rb
        rc = wm * (MV @ y)/h_max
        rc = M.T @ rc

        w1 = 1

        r = w1**2* ra + w2 * rb + w3*rc
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
        if(ii==n_iter-1):
            print(la.norm(LV_),la.norm(L_@V),la.norm(AV),la.norm(A@V),la.norm(MV),la.norm(M@V))
        residual_history.append(la.norm(r))

    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history,'regParam2_history': alpha_history, 'relError': rre_history, 'Residual': residual_history, 'its': ii,'Ms':Ms}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}



    return (x, info, v_ests, v_inv_ests)


def MMGKS_dyn_joint(A, b, L_,I, t_end,shape,pnorm=2, qnorm=1, rnorm= 1, projection_dim=3, n_iter=5, n_iter_b=60,
regparam='gcv', vs_true = None, v_primes_true = None, v_max = None, x_true=None, min_l = 0,max_l=1,opt='nonscaled', non_neg= True, pnorm_opt=1,qnorm_opt = 1, proj_dim_opt = 1,**kwargs):
    def M_mat(u,v):
        R = np.array(list(np.ndindex(*u.shape)))
        nx = u.shape[0]
        ny = u.shape[1]
        new_ind = (v.reshape(nx*ny,2)+R).astype(int)
        new_ind[new_ind>nx-1] = nx-1
        new_ind[new_ind<0] = 0
        v_prime=np.zeros((nx,ny,2))
        v_prime[new_ind[:,0],new_ind[:,1]] = -v.reshape(nx*ny,2)
        inds = [ind[0]*nx+ind[1] for ind in new_ind]
        rows = [i for i in range(len(inds))]
        cols = [inds[i] for i in range(len(inds))]
        data = [1 for i in range(len(inds))]
        M_ = scipy.sparse.coo_matrix((data, (rows,cols)),shape = (nx*ny,nx*ny))
        M_ = M_.tocsr()
        return M_,v_prime
    nx = shape[0];ny = shape[1]

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    scale = kwargs['scale'] if ('scale' in kwargs) else 2
    reduction = kwargs['reduction'] if ('reduction' in kwargs) else False
    (U, B, V) = golub_kahan_2(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    alpha_history = []
    residual_history = []
    e = 1
    x = A.T @ b # initialize x for reweighting

    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        # compute reweighting for p-norm approximation
        v = A @ x - b
        x_ = x.reshape((-1,))

        len_=nx*ny
        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]
        if vs_true is not None:
            v_ests = vs_true
        else:
            _,v_ests,_ = solve_opt_flow(x_traj,shape=shape,t_end=t_end,v_trues=None,v_max=v_max,n_iter=n_iter_b,reduction = reduction,
                                        scale = scale,pnorm=pnorm_opt,qnorm=qnorm_opt,proj_dim=proj_dim_opt) #solve_opt_flow(x_traj,shape,t_end)
            v_ests = np.rint(np.array(v_ests))

        Ms = []
        v_primes_=[]
        for t in range (t_end-1):
            M_v_prime =   M_mat(im_func(x_[(t_end - t -1)*(x_.shape[0]//t_end):(t_end - t)*(x_.shape[0]//t_end)],shape), v_ests[::-1][t].reshape((nx,ny,2)))
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

        I1 = scipy.sparse.identity(nx*ny)

        M_top = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [I1, -Ms[i]] + (len(Ms)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(Ms))]
        M_bottom = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [-M_primes[i],I1] + (len(M_primes)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(M_primes))]

        M = scipy.sparse.vstack((*M_top,*M_bottom))
        M = scipy.sparse.csr_matrix(M)

        #L = scipy.sparse.vstack((L_,M))

        AV = A@V
        LV_ = L_@V
        MV = M@V

        v = A @ x - b
        u = L_@x
        z= M@x


        wf = ((v**2 + epsilon**2)**(pnorm/2 - 1))
        AA = AV*(wf**(1/2))
        (Q_A, R_A) = la.qr(AA, mode='economic')

        wm = ((z**2 + epsilon**2)**(rnorm/2 - 1))
        MM = MV * wm


        wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))
        LL_ = LV_ * wr

        LL = np.concatenate((LL_,MM))
        (Q_L, R_L) = la.qr(LL, mode='economic')


        if regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            f_max = 1
            g_max = 1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'gcv':
            lambdah = generalized_crossvalidation_2(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            f_max = 1
            g_max = 1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'new':

            x2 = np.linalg.lstsq(np.concatenate(((1-min_l)*R_A, (min_l)* R_L)),
                            np.concatenate(((1-min_l)*Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None) [0]

            x1 = np.linalg.lstsq(np.concatenate(((min_l)*R_A, (1-min_l)* R_L)),
                            np.concatenate(((min_l)*Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None) [0]


            f_max = la.norm(AA@x1-b)**2
            g_max = la.norm(LL@x2)**2

            AA = AA/np.sqrt(f_max)
            LL = LL/np.sqrt(g_max)
            b_=b/np.sqrt(f_max)

            (Q_A, R_A) = la.qr(AA, mode='economic')
            (Q_L, R_L) = la.qr(LL, mode='economic')

            lambdah = gg(AA,Q_A,R_A,b_,LL,Q_L,R_L,np.zeros((LL.shape[0],1)))
            w1 = 1
            w2 = lambdah**2
            y,_,_,_ = np.linalg.lstsq(np.concatenate((w1*R_A, (lambdah) * R_L)),
                        np.concatenate((w1*Q_A.T@ b_, (lambdah) *np.zeros((R_L.shape[0],1)))),rcond=None)
        else:
            lambdah = regparam
        lambda_history.append(lambdah)


        x = V @ y

        if (non_neg):
            x[x<0] = 0
        x_history.append(x)
        if ii >= R_L.shape[0]:
            break

        ra = wf * (AV @ y - b)/f_max
        ra = A.T @ ra
        rb = wr * (LV_ @ y)/g_max
        rb = L_.T @ rb
        rc = wm * (MV @ y)/g_max
        rc = M.T @ rc

        w1 = 1

        r = w1**2* ra + w2 * (rb + rc)
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)

        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))

        Lvn = L_*vn
        LV = np.column_stack((LV_, Lvn))
        Mvn = M*vn
        MV = np.column_stack((MV, Mvn))
        residual_history.append(la.norm(r))

    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history,'regParam2_history': alpha_history, 'relError': rre_history, 'Residual': residual_history, 'its': ii,'Ms':Ms}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}

    return (x, info, v_ests, v_inv_ests)

def MMGKS_dyn2(A, b, L_,I, t_end,shape,pnorm=2, qnorm=1, rnorm= 1, projection_dim=3, n_iter=5, n_iter_b=60,
regparam='gcv', vs_true = None, v_primes_true = None, v_max = None, x_true=None, min_l = 0,max_l=1,opt='nonscaled', non_neg= True, qnorm_opt = 1, proj_dim_opt = 3,**kwargs):
    def M_mat(u,v):
        R = np.array(list(np.ndindex(*u.shape)))
        nx = u.shape[0]
        ny = u.shape[1]
        new_ind = (v.reshape(nx*ny,2)+R).astype(int)
        new_ind[new_ind>nx-1] = nx-1
        new_ind[new_ind<0] = 0
        v_prime=np.zeros((nx,ny,2))
        v_prime[new_ind[:,0],new_ind[:,1]] = -v.reshape(nx*ny,2)
        inds = [ind[0]*nx+ind[1] for ind in new_ind]
        rows = [i for i in range(len(inds))]
        cols = [inds[i] for i in range(len(inds))]
        data = [1 for i in range(len(inds))]
        M_ = scipy.sparse.coo_matrix((data, (rows,cols)),shape = (nx*ny,nx*ny))
        M_ = M_.tocsr()
        return M_,v_prime
    nx = shape[0];ny = shape[1]

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    scale = kwargs['scale'] if ('scale' in kwargs) else 2
    reduction = kwargs['reduction'] if ('reduction' in kwargs) else False
    (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    alpha_history = []
    residual_history = []
    e = 1
    x = A.T @ b # initialize x for reweighting

    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        # compute reweighting for p-norm approximation
        v = A @ x - b
        x_ = x.reshape((-1,))

        len_=nx*ny
        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]
        if vs_true is not None:
            v_ests = vs_true
        else:
            _,v_ests,_ = solve_opt_flow(x_traj,shape=shape,t_end=t_end,v_trues=None,v_max=v_max,n_iter=n_iter_b,reduction = reduction, scale = scale,qnorm=qnorm_opt,proj_dim=proj_dim_opt) #solve_opt_flow(x_traj,shape,t_end)
            v_ests = (np.array(v_ests))

        Ms = []
        v_primes_=[]
        for t in range (t_end-1):
            M_v_prime =   M_mat(im_func(x_[(t_end - t -1)*(x_.shape[0]//t_end):(t_end - t)*(x_.shape[0]//t_end)],shape), v_ests[::-1][t].reshape((nx,ny,2)))
            Ms.append(M_v_prime[0])
            v_primes_.append(M_v_prime[1])
        Ms.reverse()

        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]

        if v_primes_true is not None:
            v_inv_ests = v_primes_true
        else:
            v_inv_ests = v_primes_ #solve_opt_flow_b(x_inv_traj,shape,t_end,None,v_max,n_iter_b) #solve_opt_flow(x_inv_traj,shape,t_end)
            v_inv_ests = (np.array(v_inv_ests))

        M_primes = [x_[0:1*(x_.shape[0]//t_end)]]
        M_primes = []
        for t in range (t_end-1):
            M_primes.append(M_mat(im_func(x_[t*(x_.shape[0]//t_end):(t+1)*(x_.shape[0]//t_end)],shape), v_inv_ests[::-1][t].reshape((nx,ny,2)))[0])

        I1 = scipy.sparse.identity(nx*ny)

        M_top = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [I1, -Ms[i]] + (len(Ms)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(Ms))]
        M_bottom = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [-M_primes[i],I1] + (len(M_primes)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(M_primes))]

        M = scipy.sparse.vstack((*M_top,*M_bottom))
        M = scipy.sparse.csr_matrix(M)

        #L = scipy.sparse.vstack((L_,M))
        # if (ii==0):
        AV = A@V
        LV_ = L_@V
        MV = M@V

        v = A @ x - b
        u = L_@x
        z= M@x


        wf = ((v**2 + epsilon**2)**(pnorm/2 - 1))
        AA = AV*(wf**(1/2))
        (Q_A, R_A) = la.qr(AA, mode='economic')

        wm = ((z**2 + epsilon**2)**(rnorm/2 - 1))
        MM = MV * wm
        (Q_M, R_M) = la.qr(MM, mode='economic')

        wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))
        LL_ = LV_ * wr
        (Q_L_, R_L_) = la.qr(LL_, mode='economic')

        LL = np.concatenate((LL_,MM))
        (Q_L, R_L) = la.qr(LL, mode='economic')


        if regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            f_max = 1
            g_max = 1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'gcv':
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            w3= lambdah
            f_max = 1
            g_max = 1
            h_max =1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'new':

            x2 = np.linalg.lstsq(np.concatenate(((1-min_l)*R_A, (min_l)* R_L_)),
                            np.concatenate(((1-min_l)*Q_A.T@ b, np.zeros((R_L_.shape[0],1)))),rcond=None) [0]

            x1 = np.linalg.lstsq(np.concatenate(((min_l)*R_A, (1-min_l)* R_L_)),
                            np.concatenate(((min_l)*Q_A.T@ b, np.zeros((R_L_.shape[0],1)))),rcond=None) [0]

            f_max = la.norm(AA@x1-b)**2
            g_max = la.norm(LL_@x2)**2


            AA = AA/np.sqrt(f_max)
            LL_ = LL_/np.sqrt(g_max)
            b_=b/np.sqrt(f_max)

            (Q_A, R_A) = la.qr(AA, mode='economic')
            (Q_L_, R_L_) = la.qr(LL_, mode='economic')


            lambdah = gg(AA,Q_A,R_A,b_,LL_,Q_L_,R_L_,np.zeros((LL_.shape[0],1)))


            AALL_ = np.concatenate((AA,lambdah*LL_))
            (Q_AL_, R_AL_) = la.qr(AALL_, mode='economic')

            b_2 = np.concatenate((b_,np.zeros((LL_.shape[0],1))))

            x2_2 = np.linalg.lstsq(np.concatenate(((1-min_l)*R_AL_, (min_l)* R_M)),
                            np.concatenate(((1-min_l)*Q_AL_.T@ b_2, np.zeros((R_M.shape[0],1)))),rcond=None) [0]

            x1_2 = np.linalg.lstsq(np.concatenate(((min_l)*R_AL_, (1-min_l)* R_M)),
                            np.concatenate(((min_l)*Q_AL_.T@ b_2, np.zeros((R_M.shape[0],1)))),rcond=None) [0]

            f_max_2 = la.norm(AALL_@x1_2-b_2)**2
            g_max_2 = la.norm(MM@x2_2)**2


            AALL_ = AALL_/np.sqrt(f_max_2)
            MM = MM/np.sqrt(g_max_2)
            b_2_=b_2/np.sqrt(f_max_2)

            (Q_AL_, R_AL_) = la.qr(AALL_, mode='economic')
            (Q_M, R_M) = la.qr(MM, mode='economic')


            alpha = gg(AALL_,Q_AL_,R_AL_,b_2_,MM,Q_M,R_M,np.zeros((MM.shape[0],1)))

            alpha_history.append(alpha)

            w1 = 1
            w2 = lambdah**2
            w3 = alpha**2


            y,_,_,_ = np.linalg.lstsq(np.concatenate((w1*R_AL_, alpha*R_M)),
                        np.concatenate((w1*Q_AL_.T@ b_2_, alpha*np.zeros((R_M.shape[0],1)))),rcond=None)
        else:
            lambdah = regparam
        lambda_history.append(lambdah)



        x = V @ y

        if (non_neg):
            x[x<0] = 0
        x_history.append(x)
        if ii >= R_L.shape[0]:
            break

        ra = wf * (AV @ y - b)/(f_max*f_max_2)
        ra = A.T @ ra
        rb = wr * (LV_ @ y)/(g_max*f_max_2)
        rb = L_.T @ rb
        rc = wm * (MV @ y)/g_max_2
        rc = M.T @ rc

        w1 = 1

        r = w1**2* ra + w2 * rb + w3*rc
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
        if(ii==n_iter-1):
            print(la.norm(LV_),la.norm(L_@V),la.norm(AV),la.norm(A@V),la.norm(MV),la.norm(M@V))
        residual_history.append(la.norm(r))

    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history,'regParam2_history': alpha_history, 'relError': rre_history, 'Residual': residual_history, 'its': ii,'Ms':Ms}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}



    return (x, info, v_ests, v_inv_ests)


def MMGKS_a(A, b, L, pnorm=2, qnorm=1, projection_dim=3, n_iter=5, regparam='gcv', x_true=None, **kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    isoTV_option = kwargs['isoTV'] if ('isoTV' in kwargs) else False
    GS_option = kwargs['GS'] if ('GS' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    prob_dims = kwargs['prob_dims'] if ('prob_dims' in kwargs) else False
    non_neg = kwargs['non_neg'] if ('non_neg' in kwargs) else False
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]
    (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    residual_history = []
    e = 1
    x = A.T @ b
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
    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        v = A @ x - b
        wf = (v**2 + epsilon**2)**(pnorm/2 - 1)
        AA = AV*wf
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
        LL = LV * wr
        (Q_L, R_L) = la.qr(LL, mode='economic')
        if regparam == 'gcv':
            lambdah = generalized_crossvalidation(Q_A, R_A, R_L, wf *b, **kwargs)
        elif regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, wf *b, **kwargs)

        else:
            lambdah = regparam

        lambda_history.append(lambdah)
        y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)),
                        np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        x = V @ y
        x[x<0] = 0
        x_history.append(x)
        if ii >= R_L.shape[0]:
            break
        v = AV@y
        v = v - b
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


def MMGKS_dyn_joint_(A, b, L_,I, t_end,shape,pnorm=2, qnorm=1, rnorm= 1, projection_dim=3, n_iter=5, n_iter_b=60,
regparam='gcv', vs_true = None, v_primes_true = None, v_max = None, x_true=None, min_l = 0,max_l=1,opt='nonscaled', non_neg= True, pnorm_opt=1,qnorm_opt = 1, proj_dim_opt = 1, interval=1,**kwargs):
    def M_mat(u,v):
        R = np.array(list(np.ndindex(*u.shape)))
        nx = u.shape[0]
        ny = u.shape[1]
        new_ind = (v.reshape(nx*ny,2)+R).astype(int)
        new_ind[new_ind>nx-1] = nx-1
        new_ind[new_ind<0] = 0
        v_prime=np.zeros((nx,ny,2))
        v_prime[new_ind[:,0],new_ind[:,1]] = -v.reshape(nx*ny,2)
        inds = [ind[0]*nx+ind[1] for ind in new_ind]
        rows = [i for i in range(len(inds))]
        cols = [inds[i] for i in range(len(inds))]
        data = [1 for i in range(len(inds))]
        M_ = scipy.sparse.coo_matrix((data, (rows,cols)),shape = (nx*ny,nx*ny))
        M_ = M_.tocsr()
        return M_,v_prime
    nx = shape[0];ny = shape[1]

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    scale = kwargs['scale'] if ('scale' in kwargs) else 2
    reduction = kwargs['reduction'] if ('reduction' in kwargs) else False
    (U, B, V) = golub_kahan_2(A, b, projection_dim, dp_stop, **kwargs)

    x_history = []
    lambda_history = []
    alpha_history = []
    residual_history = []
    M_history = []
    v_ests_history = []
    ux_history = []
    uy_history = []
    ux_uy_history = []
    ut_history = []
    e = 1
    x = A.T @ b # initialize x for reweighting
    x_history.append(x)

    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        # compute reweighting for p-norm approximation
        v = A @ x - b
        x_ = x.reshape((-1,))
        len_=nx*ny
        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]
        if ((ii%interval == 0)):
            if vs_true is not None:
                v_ests = vs_true
            else:
                # (v_ests, v_larges, info, ux_uy_history, ut_history)
                v_ests_small, v_ests, _, ux_, uy_, ux_uy_, ut_ = solve_opt_flow(x_traj,shape=shape,t_end=t_end,v_trues=None,v_max=v_max,n_iter=n_iter_b,reduction = reduction,
                                            scale = scale,pnorm=pnorm_opt,qnorm=qnorm_opt,proj_dim=proj_dim_opt) #solve_opt_flow(x_traj,shape,t_end)
                v_ests = np.rint(np.array(v_ests))
                v_ests_history.append(v_ests_small)
                ux_history.append(ux_)
                uy_history.append(uy_)
                ux_uy_history.append(ux_uy_)
                ut_history.append(ut_)
            Ms = []
            v_primes_=[]
            for t in range (t_end-1):
                M_v_prime =   M_mat(im_func(x_[(t_end - t -1)*(x_.shape[0]//t_end):(t_end - t)*(x_.shape[0]//t_end)],shape), v_ests[::-1][t].reshape((nx,ny,2)))
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

            I1 = scipy.sparse.identity(nx*ny)

            M_top = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [I1, -Ms[i]] + (len(Ms)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(Ms))]
            M_bottom = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [-M_primes[i],I1] + (len(M_primes)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(M_primes))]

            M = scipy.sparse.vstack((*M_top,*M_bottom))
            M = scipy.sparse.csr_matrix(M)
            M_history.append(M)
        #L = scipy.sparse.vstack((L_,M))

        AV = A@V
        LV_ = L_@V
        MV = M@V

        v = A @ x - b
        u = L_@x
        z= M@x


        wf = ((v**2 + epsilon**2)**(pnorm/2 - 1))
        AA = AV*(wf**(1/2))
        (Q_A, R_A) = la.qr(AA, mode='economic')

        wm = ((z**2 + epsilon**2)**(rnorm/2 - 1))
        MM = MV * wm


        wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))
        LL_ = LV_ * wr

        LL = np.concatenate((LL_,MM))
        (Q_L, R_L) = la.qr(LL, mode='economic')


        if regparam == 'dp':
            lambdah = discrepancy_principle(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            f_max = 1
            g_max = 1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'gcv':
            lambdah = generalized_crossvalidation_2(Q_A, R_A, R_L, wf *b, **kwargs)
            w2 = lambdah
            f_max = 1
            g_max = 1
            y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None)
        elif regparam == 'new':

            x2 = np.linalg.lstsq(np.concatenate(((1-min_l)*R_A, (min_l)* R_L)),
                            np.concatenate(((1-min_l)*Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None) [0]

            x1 = np.linalg.lstsq(np.concatenate(((min_l)*R_A, (1-min_l)* R_L)),
                            np.concatenate(((min_l)*Q_A.T@ b, np.zeros((R_L.shape[0],1)))),rcond=None) [0]


            f_max = la.norm(AA@x1-b)**2
            g_max = la.norm(LL@x2)**2

            AA = AA/np.sqrt(f_max)
            LL = LL/np.sqrt(g_max)
            b_=b/np.sqrt(f_max)

            (Q_A, R_A) = la.qr(AA, mode='economic')
            (Q_L, R_L) = la.qr(LL, mode='economic')

            lambdah = gg(AA,Q_A,R_A,b_,LL,Q_L,R_L,np.zeros((LL.shape[0],1)))
            w1 = 1
            w2 = lambdah**2
            y,_,_,_ = np.linalg.lstsq(np.concatenate((w1*R_A, (lambdah) * R_L)),
                        np.concatenate((w1*Q_A.T@ b_, (lambdah) *np.zeros((R_L.shape[0],1)))),rcond=None)
        else:
            lambdah = regparam
        lambda_history.append(lambdah)


        x = V @ y

        if (non_neg):
            x[x<0] = 0
        x_history.append(x)
        if ii >= R_L.shape[0]:
            break

        ra = wf * (AV @ y - b)/f_max
        ra = A.T @ ra
        rb = wr * (LV_ @ y)/g_max
        rb = L_.T @ rb
        rc = wm * (MV @ y)/g_max
        rc = M.T @ rc

        w1 = 1

        r = w1**2* ra + w2 * (rb + rc)
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)

        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))
        Avn = A @ vn
        AV = np.column_stack((AV, Avn))

        Lvn = L_*vn
        LV = np.column_stack((LV_, Lvn))
        Mvn = M*vn
        MV = np.column_stack((MV, Mvn))
        residual_history.append(la.norm(r))

    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history,'regParam2_history': alpha_history,
                'relError': rre_history, 'Residual': residual_history, 'its': ii,'Ms':Ms,'M_primes':M_primes}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}

    return (x, info, v_ests, v_inv_ests, M_history, x_history, v_ests_history, ux_history, uy_history, ux_uy_history, ut_history)
