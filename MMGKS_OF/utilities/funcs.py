from utilities.imports import *

def vec(img):
    vector = img.reshape(img.shape[0]*img.shape[1])
    return vector

def vectorize_func(img):
    vector = img.reshape(img.shape[0]*img.shape[1])
    return vector

def im_func(vector,shape):
    img = vector.reshape(shape)
    return img

def display_img_traj(img_list,shape,title=''):
    t_end = len(img_list)
    fig,axs = plt.subplots(1,t_end,dpi = 400,figsize=(20,5),sharey=True)  
    for t in range(t_end):
        img = img_list[t]
        img = im_func(img,shape)
        axs[t].imshow(img,vmin=0,vmax=1*np.max(img),cmap='gray')
        axs[t].axis('off')
        axs[t].set_title(f'$t = $ {t}')
        axs[t].invert_yaxis()   
    fig.suptitle(title,va='bottom',y=0.22)

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