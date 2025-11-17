
from imports import *
from OF import *

import concurrent
def build_Ms(xs):
    return [calc_M([xs[i], xs[i+1]]) for i in range(len(xs)-1)]
def MMGKS_OF(A, b, L_, t_end,shape,pnorm=2, qnorm=1, rnorm= 1, projection_dim=3, n_iter=5,
regparam='gcv', x0=None, V0 = None,v_ests_0=None,x_true=None,power=1,two_way=True,kmin = 3, l_max = 1,interval = 10, M0s = None, M_prime0s = None,  start_of=0,l_curve_plot=False,**kwargs):
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

    if V0 is not None:
        V = V0
    else:
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
        spacen = int(Ls.shape[0] / 2)
    rp = regparam
    for ii in tqdm(range(n_iter), desc='running MMGKS...'):
        if rp == 'dp_l_curve':
            if ((ii<n_iter//2)):
                regparam = 'l_curve'
            else:
                regparam = 'dp'
        # compute reweighting for p-norm approximation
        #print(la.norm(x-A.T @ b))
        v = A @ x - b
        x_ = x.reshape((-1,))
        #print(x_.shape)
        u = L_@x
        len_=nx*ny
        x_traj = [x[len_*i:len_*(i+1)] for i in range(t_end)]
        xs = x.reshape(t_end,nx,ny)
        xs = [np.clip(x,0,None) for x in xs]
        if (ii%interval == 0): # or ii==n_iter-1):
            
            if (ii<start_of) and (M0s is None):
                Ms = [scipy.sparse.identity(nx * ny) for _ in range(t_end-1)]
                M_primes = [scipy.sparse.identity(nx * ny) for _ in range(t_end-1)]
            elif (ii<start_of) and (M0s is not None):
                Ms = M0s
                M_primes = M_prime0s
            else:
                st = time.time()
                
                def _compute_M_forward(args):
                    xs, i = args
                    return calc_M([np.clip(xs[i], 0, None), np.clip(xs[i+1], 0, None)])

                def _compute_M_backward(args):
                    xs, i = args
                    return calc_M([np.clip(xs[i], 0, None), np.clip(xs[i-1], 0, None)])

                # In your function:
                if parallel_of:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                        Ms = list(executor.map(_compute_M_forward, 
                                            [(xs, i) for i in range(t_end - 1)]))
                        M_primes = list(executor.map(_compute_M_backward, 
                                                    [(xs, i) for i in range(1, t_end)]))
                else:
                    
                    Ms = [
                        calc_M([xs[i], xs[i+1]])
                        for i in range(t_end - 1)
                    ]
                    M_primes = [
                        calc_M([xs[i], xs[i-1]])
                        for i in range(1, t_end)
                    ]

            I1 = scipy.sparse.identity(nx*ny)

            st = time.time()
            M_top = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [I1, -Ms[i]] + (len(Ms)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(Ms))]
            M_bottom = [scipy.sparse.hstack(i*[scipy.sparse.csr_matrix((nx*ny,nx*ny))] + [-M_primes[i],I1] + (len(M_primes)-i-1)*[scipy.sparse.csr_matrix((nx*ny,nx*ny))]) for i in range(len(M_primes))]
            et = time.time()
            #print('T:', et-st)
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
            wr = smoothed_holder_weights(u, epsilon=epsilon, p=qnorm).reshape((-1,1))#**0.5
        AV = A@V
        LV_ = L_@V
        MV = M@V

        v = A @ x - b
        u = L_@x
        z= M@x
    
        wf = ((v**2 + epsilon**2)**(pnorm/2 - 1))
        AA = AV*(wf**power)
        (Q_A, R_A) = la.qr(AA, mode='economic')
        
        wm = ((z**2 + epsilon**2)**(rnorm/2 - 1))#**0.5
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
        elif regparam == 'l_curve':
            lambdah = l_curve(R_A, R_L,Q_A.T@ ((wf**power)*b),plot=l_curve_plot)
        else:
            lambdah = regparam
        # if (lambdah <1e-2):
        #     lambdah = lambdah_prev if ii>0 else 1e-2 #l_curve(R_A, R_L,Q_A.T@ ((wf**power)*b))
        # lambdah_prev=lambdah

        #lambdah2 = l_curve(R_A, R_L,Q_A.T@ ((wf**power)*b), **kwargs)
        lambda_history.append(lambdah)
        y,_,_,_ = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_L)), np.concatenate((Q_A.T@ ((wf**power)* b), np.zeros((R_L.shape[0],1)))),rcond=None)

        x = V @ y
        
        if (non_neg):
            x[x<0] = 0
        x_history.append(x)
        # if ii >= R_L.shape[0]:
        #     break

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

        if ((V.shape[1] == l_max) or (ii == (n_iter -1))):

            # Compute truncated SVD with k singular values
            _,_ ,Wt = np.linalg.svd(np.vstack((R_A, np.sqrt(lambdah) * R_L)))

            # Sorting in descending order
            #Wt = Wt[::-1,:]


            W = Wt.T
            W = W[:, :kmin-1]

            V_tilde = V[:,:-1]@W

            V_tilde, _ = np.linalg.qr(V_tilde)
            assert np.linalg.norm( (V_tilde.T @ V_tilde) - np.eye(V_tilde.shape[1]) ) < 1e-10, "New basis is not a basis"
            x_new =  x - V_tilde @ (V_tilde.T @ x)
            x_new /= la.norm(x_new)
            V = np.column_stack((V_tilde, x_new))
            assert np.linalg.norm( (V.T @ V) - np.eye(V.shape[1]) ) < 1e-10, "New basis is not a basis"
            V, _ = np.linalg.qr(V)

            AV = A@V #np.column_stack((AV, Avn))

            LV = L_@V #np.column_stack((LV, Lvn))

            MV = M@V


    if x_true is not None:
        x_true_norm = la.norm(x_true)
        rre_history = [la.norm(x - x_true)/x_true_norm for x in x_history]
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history,'regParam2_history': alpha_history, 
                'relError': rre_history, 'Residual': residual_history, 'its': ii,'Ms':Ms,'M_primes':M_primes,'V':V}
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii,'Ms':Ms,'M_primes':M_primes,'V':V}
    
    return (x, info,V,lambdah)
