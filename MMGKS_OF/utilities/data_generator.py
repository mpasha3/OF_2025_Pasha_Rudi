import numpy as np
from utilities.imports import *
from utilities.operators import *
from utilities.tomo_class import *
from trips.test_problems.Deblurring2D import *
from PIL import Image
from trips.utilities.helpers import convert_image_for_trips

Tomo = Tomography()
Tomo2 = Tomography2()

def gen_im_seq(shape,t_end=3,v_max = 2,v_min = 1,padding = 4):
    v_mag_= v_min
    v_mag = v_max
    us = []

    v_max = padding
    v_min = padding
    for t in range(t_end):
        u = np.zeros(shape)
        scale = shape[0]//8
        size = u.shape[0]

        
        
        u[v_max+v_mag*t:2*scale+v_max+v_mag*t,v_max:2*scale+v_max]=1

        u[size-v_min-1*scale - v_mag_*t :size-v_min - v_mag_*t,size-v_min-2*scale:size-v_min]=0.99
        u[size-v_min-1*scale-scale - v_mag_*t:size-v_min-scale-v_mag_*t,size-v_min-2*scale:size-v_min-scale]=0.99

        u[v_min + v_mag_*t: v_min+1*scale+ v_mag_*t,size-v_min-2*scale-v_mag_*t:size-v_min- v_mag_*t]=0.98
        u[v_min+scale+ v_mag_*t: v_min+2*scale+ v_mag_*t,size-v_min-2*scale- v_mag_*t:size-v_min-scale-v_mag_*t]=0.98

        u[size-v_max-2*scale:size-v_max-1*scale,v_max+ v_mag*t:v_max+3*scale+ v_mag*t]=0.97
        u[size-v_max-1*scale:size-v_max,v_max+scale+ v_mag*t:v_max+2*scale+ v_mag*t]=0.97

        us.append(u)
    
    def gen_v(u):
        v =np.zeros(shape + (2,))
        v[np.where(u==1)[0].min()-v_mag:np.where(u==1)[0].max()+v_mag+1,np.where(u==1)[1].min()-v_mag:np.where(u==1)[1].max()+v_mag+1]  = np.array([1,0])*v_mag
        v[np.where(u==0.99)[0].min()-v_mag_:np.where(u==0.99)[0].max()+v_mag_+1,np.where(u==0.99)[1].min()-v_mag_:np.where(u==0.99)[1].max()+v_mag_+1]  = np.array([-1,0])*v_mag_
        v[np.where(u==0.98)[0].min()-v_mag_:np.where(u==0.98)[0].max()+v_mag_+1,np.where(u==0.98)[1].min()-v_mag_:np.where(u==0.98)[1].max()+v_mag_+1]  = np.array([1,-1])*v_mag_
        v[np.where(u==0.97)[0].min()-v_mag:np.where(u==0.97)[0].max()+v_mag+1,np.where(u==0.97)[1].min()-v_mag:np.where(u==0.97)[1].max()+v_mag+1]  = np.array([0,1])*v_mag
        return v

    def gen_v_prime(u):
        v =np.zeros(shape + (2,))
        v[np.where(u==1)[0].min()-v_mag:np.where(u==1)[0].max()+v_mag+1,np.where(u==1)[1].min()-v_mag:np.where(u==1)[1].max()+v_mag+1]  = -np.array([1,0])*v_mag
        v[np.where(u==0.99)[0].min()-v_mag_:np.where(u==0.99)[0].max()+v_mag_+1,np.where(u==0.99)[1].min()-v_mag_:np.where(u==0.99)[1].max()+v_mag_+1]  = -np.array([-1,0])*v_mag_
        v[np.where(u==0.98)[0].min()-v_mag_:np.where(u==0.98)[0].max()+v_mag_+1,np.where(u==0.98)[1].min()-v_mag_:np.where(u==0.98)[1].max()+v_mag_+1]  = -np.array([1,-1])*v_mag_
        v[np.where(u==0.97)[0].min()-v_mag:np.where(u==0.97)[0].max()+v_mag+1,np.where(u==0.97)[1].min()-v_mag:np.where(u==0.97)[1].max()+v_mag+1]  = -np.array([0,1])*v_mag

        return v
    us_rev = us[::-1]
    vs = []
    v_primes = []
    for t in range(0,t_end-1):
        vs.append(gen_v(us[t]))
        v_primes.append(gen_v_prime(us_rev[t]))

    u_traj=[vec(u) for u in us] 
    u_inv_traj = [vec(u_inv) for u_inv in us_rev]
    class test_sequence:
    # init method or constructor
        def __init__(self, u_traj,u_inv_traj,vs,v_primes):
            self.u_traj = u_traj
            self.u_inv_traj = u_inv_traj
            self.vs = vs
            self.v_primes = v_primes
        
    return test_sequence(u_traj,u_inv_traj,vs,v_primes)

def gen_joint_blur_op_and_data(x_traj, t_end, nx,ny, spread,dim= (1,1),noise_level=1e-2,CommitCrime=False):
    Deblur = Deblurring2D(CommitCrime =  CommitCrime)
    A = Deblur.forward_Op(dim, spread, nx, ny)#[6,6] for numbers

    As = [A]*t_end
    b_trues = []
    x_trues = []
    j=0
    for x_true in x_traj:
        im = plt.imsave("data/my_image_data/x.png",x_true.reshape(nx,ny))
        convert_image_for_trips(imag = 'x', image_type= 'png')
        x_true = Deblur.gen_true_mydata('x', nx = nx, ny = ny)
        b_true = Deblur.gen_data(x_true)
        b_trues.append(b_true)
        x_trues.append(vec(x_true))
        j+=1
    b_traj_and_deltas = [Deblur.add_noise(b_true, opt = 'Gaussian', noise_level = noise_level) for b_true in b_trues]
    b_traj = np.array(b_traj_and_deltas,dtype=object)[:,0]
    deltas = np.array(b_traj_and_deltas,dtype=object)[:,1]
    L = gen_first_derivative_operator_2D(nx, ny)
    data_vec = [b.reshape((-1,1)) for b in b_traj]
    data_vec_true = [b.reshape((-1,1)) for b in b_trues]
    A_bar = pylops.BlockDiag([As[i] for i in range(t_end)])
    L_bar = scipy.sparse.block_diag([L for i in range(t_end)])
    X_bar = vectorize_func(np.array(x_trues))
    data_vec_bar = vectorize_func(np.array(data_vec)).reshape((-1,1))
    data_vec_true_bar = vectorize_func(np.array(data_vec_true)).reshape((-1,1))
    I = sparse.identity(A.shape[1])#np.eye(A.shape[1])
    I_bar = scipy.sparse.block_diag([I for i in range(t_end)]) 

    return (X_bar, A_bar,data_vec_bar,data_vec_true_bar,L_bar,I_bar,deltas)

def gen_joint_tomo_op_and_data(x_traj, t_end, nx,ny, views,noise_level=1e-2,case='a',CommitCrime=False):
    Tomo2 = Tomography2(CommitCrime=CommitCrime)
    np.random.seed(0)
    '''Generates the diagonalized foward operator and vectorized output for the joint problem'''
    if (case == 'c'):
        As = []
        b_trues = []
        j=0
        for x_true in x_traj:
            b = np.pi #np.linspace(0,np.pi,views,endpoint=False)[-1]

            min_angle = ((b/(views))/t_end)*j

            max_angle = b + ((b/(views))/t_end)*j
            print(np.linspace(min_angle*360/(2*np.pi),max_angle*360/(2*np.pi), views, endpoint=False))
            
            (A, b_true, p, q, AforMatrixOperation) = Tomo2.gen_data(x_true, nx, ny, views,min_angle,max_angle)
            As.append(A)
            b_trues.append(b_true)
            j+=1

 
    elif (case == 'b'):
        As = []
        b_trues = []
        j=0
        for x_true in x_traj:
            b = np.linspace(0,np.pi,t_end,endpoint=False)[-1]

            min_angle = (np.pi/t_end)*j

            max_angle = (np.pi/t_end)*(j+1)
            print(np.linspace(min_angle*360/(2*np.pi),max_angle*360/(2*np.pi), views, endpoint=False), views, type(views))
            (A, b_true, p, q, AforMatrixOperation) = Tomo2.gen_data(x_true, nx, ny, views,min_angle,max_angle)
            As.append(A)
            b_trues.append(b_true)
            j+=1 
    elif (case == 'a'):
        As = []
        b_trues = []
        j=0
        for x_true in x_traj:
            min_angle = 0
            max_angle = np.pi
            print(np.linspace(min_angle*360/(2*np.pi),max_angle*360/(2*np.pi), views, endpoint=False))
            (A, b_true, p, q, AforMatrixOperation) = Tomo2.gen_data(x_true, nx, ny, views,min_angle,max_angle)
            As.append(A)
            b_trues.append(b_true)
            j+=1
    elif (case == 'd'):
        As = []
        b_trues = []
        rowshift = 11
        j=0
        for x_true in x_traj:
            min_angle = (rowshift*j)*(2*np.pi/360)
            max_angle = (14*(views-1)+ rowshift*j)*(2*np.pi/360)
            print(np.linspace(min_angle*360/(2*np.pi),max_angle*360/(2*np.pi), views, endpoint=False))
            (A, b_true, p, q, AforMatrixOperation) = Tomo2.gen_data(x_true, nx, ny, views,min_angle,max_angle)
            As.append(A)
            b_trues.append(b_true)
            j+=1

    b_traj_and_deltas = [Tomo2.add_noise(b_true, opt = 'Gaussian', noise_level=noise_level) for b_true in b_trues]
    b_traj = np.array(b_traj_and_deltas,dtype=object)[:,0]
    deltas = np.array(b_traj_and_deltas,dtype=object)[:,1]
    delta = deltas[0]
    L = gen_first_derivative_operator_2D(nx, ny)
    data_vec = [b.reshape((-1,1)) for b in b_traj]
    data_vec_true = [b.reshape((-1,1)) for b in b_trues]
    A_bar = pylops.BlockDiag([As[i] for i in range(t_end)])
    # L_bar = pylops.BlockDiag([L for i in range(t_end)])
    L_bar = scipy.sparse.block_diag([L for i in range(t_end)])
    X_bar = vectorize_func(np.array(x_traj))
    data_vec_bar = vectorize_func(np.array(data_vec)).reshape((-1,1))
    data_vec_true_bar = vectorize_func(np.array(data_vec_true)).reshape((-1,1))
    I = sparse.identity(A.shape[1])#np.eye(A.shape[1])
    I_bar = scipy.sparse.block_diag([I for i in range(t_end)]) 
    b_shape=(p,q)
    
    return (X_bar, A_bar,As,data_vec_bar,b_traj,data_vec_true_bar,b_trues,b_shape,L_bar,I_bar,deltas)


