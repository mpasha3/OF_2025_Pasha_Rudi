from utilities.imports import *
class Tomography():
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
    def define_proj_id(self, sizex, sizey, views, min_angle=0, max_angle=2*np.pi, **kwargs):
        self.dataset = kwargs['dataset'] if ('dataset' in kwargs) else False
        self.nx = sizex
        self.ny = sizey
        self.p = int(np.sqrt(2)*self.nx)    # number of detector pixels
        self.q = views           # number of projection angles
        self.views = views
        self.theta = np.linspace(min_angle, max_angle, self.q, endpoint=False) #np.linspace(0, 2*np.pi, self.q, endpoint=False)   # in rad
        self.source_origin = 3*self.nx                     # source origin distance [cm]
        self.detector_origin = self.nx                      # origin detector distance [cm]
        self.detector_pixel_size = (self.source_origin + self.detector_origin)/self.source_origin
        self.detector_length = self.detector_pixel_size*self.p   # detector length
        self.vol_geom = astra.create_vol_geom(self.nx,self.nx)
        self.proj_geom = astra.create_proj_geom('fanflat', self.detector_pixel_size, self.p, self.theta, self.source_origin, self.detector_origin)
        self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)
        return self.proj_id

    def define_A(self, sizex, sizey, views, min_angle=0, max_angle=2*np.pi): 
        proj_id = self.define_proj_id(sizex, sizey, views,min_angle, max_angle)  
        self.A = astra.OpTomo(self.proj_id)    
        return self.A

    def forward_Op(self, x, sizex, sizey, views,min_angle=0, max_angle=2*np.pi):
        A = self.define_A(sizex, sizey, views,min_angle, max_angle)
        operatorf = lambda X: (A*X.reshape((sizex, sizey))).reshape(-1,1)
        operatorb = lambda B: A.T*B.reshape((self.p, self.q))
        OP = pylops.FunctionOperator(operatorf, operatorb, self.p*self.q, sizex*sizey)
        
        return OP, A

    def gen_true(self, sizex, sizey, test_problem):
        if test_problem == 'grains':
            N_fine = sizex
            numGrains = int(round(4*np.sqrt(N_fine)))
            x_true = phantom.grains(N_fine, numGrains) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'smooth':
            N_fine = sizex
            x_true = phantom.smooth(N_fine) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'tectonic':
            N_fine = sizex
            x_true = phantom.tectonic(N_fine)
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1] 
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'threephases':
            N_fine = sizex
            x_true = phantom.threephases(N_fine) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'ppower':
            N_fine = sizex
            x_true = phantom.ppower(N_fine) 
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        elif test_problem == 'simple':
            N_fine = sizex
            x_true = im_func(x_traj[0],shape)
            tmp_shape = x_true.shape
            self.nx = tmp_shape[0]
            self.ny = tmp_shape[1]
            x_truef = x_true.reshape((-1,1)) 
        else:
            raise TypeError("You must enter a valid test problem! Options are: grains, smooth, tectonic, threephases, ppower, CT60, CT90, head.")
        return (x_truef, self.nx, self.ny)

    def gen_data(self, x, nx, ny, views, min_angle=0, max_angle=2*np.pi):
        self.nx = nx
        self.ny = ny
        self.views = views
        self.max_angle = max_angle
        self.min_angle = min_angle
        proj_id = self.define_proj_id(self.nx, self.ny, self.views, self.min_angle, self.max_angle)
        (A, AforMatrixOperation) = self.forward_Op(x, self.nx, self.ny, self.views,self.min_angle,self.max_angle)
        b = A@x.reshape((-1,1))
        bshape = b.shape
        self.p = self.views
        self.q = int(bshape[0]/self.views)
        bimage = b.reshape((self.p, self.q))
        return A, b, self.p, self.q, AforMatrixOperation
    
    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            # mu_obs = np.zeros((self.p*self.q,1))      # mean of noise
            noise = np.random.randn(b_true.shape[0]).reshape((-1,1))
            e = noise_level * np.linalg.norm(b_true) / np.linalg.norm(noise) * noise
            e = e.reshape((-1,1))
            b_true = b_true.reshape((-1,1))
            delta = la.norm(e)
            b = b_true + e # add noise
            b_meas = b_true + e
            b_meas_i = b_meas.reshape((self.p, self.q))
        elif (opt == 'Poisson'):
            # Add Poisson Noise 
            gamma = 1 # background counts assumed known
            b_meas = np.random.poisson(lam=b_true+gamma) 
            b_meas_i = b_meas.reshape((self.p, self.q))
            delta = 0
        else:
            mu_obs = np.zeros(self.p*self.q)      # mean of noise
            e = np.random.laplace(self.p*self.q)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            delta = la.norm(sig_obs*e)
            b_meas_i = b_meas.reshape((self.p, self.q))
        return (b_meas_i , delta)

class Tomography2():
    def __init__(self,**kwargs):
        seed = kwargs.pop('seed',2022)
        self.CommitCrime = kwargs['CommitCrime'] if ('CommitCrime' in kwargs) else False
    def define_proj_id(self, sizex, sizey, views, min_angle=0, max_angle=2*np.pi, **kwargs):
        self.dataset = kwargs['dataset'] if ('dataset' in kwargs) else False
        self.nx = sizex
        self.ny = sizey
        self.p = int(np.sqrt(2)*self.nx)    # number of detector pixels
        self.q = views           # number of projection angles
        self.views = views
        #print(sizex,sizey,self.nx,self.ny)
        #print(self.q,type(self.q))
        self.theta = np.linspace(min_angle, max_angle, self.q, endpoint=False) #np.linspace(0, 2*np.pi, self.q, endpoint=False)   # in rad
        self.source_origin = 3*self.nx                     # source origin distance [cm]
        self.detector_origin = self.nx                      # origin detector distance [cm]
        self.detector_pixel_size = (self.source_origin + self.detector_origin)/self.source_origin
        self.detector_length = self.detector_pixel_size*self.p   # detector length
        self.vol_geom = astra.create_vol_geom(self.nx,self.nx)

        if self. CommitCrime == False:
            self.theta_mis = self.theta + 1e-8
            self.proj_geom_mis = astra.create_proj_geom('fanflat', self.detector_pixel_size, self.p, self.theta_mis, self.source_origin, self.detector_origin)
            self.proj_id_mis = astra.create_projector('strip_fanflat', self.proj_geom_mis, self.vol_geom) #astra.create_projector('line_fanflat', self.proj_geom_mis, self.vol_geom)

        self.proj_geom = astra.create_proj_geom('fanflat', self.detector_pixel_size, self.p, self.theta, self.source_origin, self.detector_origin)
        self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)
        return self.proj_id

    def define_A(self, sizex, sizey, views, min_angle=0, max_angle=2*np.pi): 
        #proj_id = self.define_proj_id(sizex, sizey, views,min_angle, max_angle)  
        self.A = astra.OpTomo(self.proj_id)    
        if self. CommitCrime == False:
            self.A_mis = astra.OpTomo(self.proj_id_mis)

        #print((self.A).shape,(self.A_mis).shape)
        return self.A

    def forward_Op(self, x, sizex, sizey, views,min_angle=0, max_angle=2*np.pi):
        print(sizex,sizey,self.p,self.q)
        A = self.define_A(sizex, sizey, views,min_angle, max_angle)
        operatorf = lambda X: (A*X.reshape((sizex, sizey))).reshape(-1,1)
        operatorb = lambda B: A.T*B.reshape((self.p, self.q))
        OP = pylops.FunctionOperator(operatorf, operatorb, self.p*self.q, sizex*sizey)
        if self. CommitCrime == False:
            A_mis = self.A_mis
            operatorf_mis = lambda X: (A_mis*X.reshape((sizex, sizey))).reshape(-1,1)
            operatorb_mis = lambda B: A_mis.T*B.reshape((self.p, self.q))
            OP_mis = pylops.FunctionOperator(operatorf_mis, operatorb_mis, self.p*self.q, sizex*sizey)
            #print(OP.shape,OP_mis.shape,sizex,sizey,self.p,self.q)
            return OP, A, OP_mis
        else:
            #print(OP.shape,sizex,sizey,self.p,self.q)
            return OP, A


    def gen_data(self, x, nx, ny, views, min_angle=0, max_angle=2*np.pi):
        self.nx = nx
        self.ny = ny
        #print(self.nx,self.ny)
        self.views = views
        self.max_angle = max_angle
        self.min_angle = min_angle
        proj_id = self.define_proj_id(self.nx, self.ny, self.views, self.min_angle, self.max_angle)
        #print(self.nx,self.ny)
        if self. CommitCrime == False:
            (A, AforMatrixOperation, A_mis) = self.forward_Op(x,self.nx, self.ny, self.views, self.min_angle, self.max_angle)
            
            b = A_mis@x.reshape((-1,1))
            print('no crime')
        else:
            (A, AforMatrixOperation) = self.forward_Op(x,self.nx, self.ny, self.views, self.min_angle, self.max_angle)
            print(self.nx,self.ny)
            b = A@x.reshape((-1,1))
            print('crime')

        bshape = b.shape
        self.p = self.views
        self.q = int(bshape[0]/self.views)
        bimage = b.reshape((self.p, self.q))
        return A, b, self.p, self.q, AforMatrixOperation
    
    def add_noise(self, b_true, opt, noise_level):
        if (opt == 'Gaussian'):
            # mu_obs = np.zeros((self.p*self.q,1))      # mean of noise
            noise = np.random.randn(b_true.shape[0]).reshape((-1,1))
            e = noise_level * np.linalg.norm(b_true) / np.linalg.norm(noise) * noise
            e = e.reshape((-1,1))
            b_true = b_true.reshape((-1,1))
            delta = la.norm(e)
            b = b_true + e # add noise
            b_meas = b_true + e
            b_meas_i = b_meas.reshape((self.p, self.q))
        elif (opt == 'Poisson'):
            # Add Poisson Noise 
            gamma = 1 # background counts assumed known
            b_meas = np.random.poisson(lam=b_true+gamma) 
            b_meas_i = b_meas.reshape((self.p, self.q))
            delta = 0
        else:
            mu_obs = np.zeros(self.p*self.q)      # mean of noise
            e = np.random.laplace(self.p*self.q)
            sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
            b_meas = b_true + sig_obs*e
            delta = la.norm(sig_obs*e)
            b_meas_i = b_meas.reshape((self.p, self.q))
        return (b_meas_i , delta)
