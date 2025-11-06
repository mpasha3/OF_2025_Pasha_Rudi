from imports import *

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

def M(u,v):
    R = np.array(list(np.ndindex(*u.shape)))
    nx = u.shape[0]
    ny = u.shape[1]
    new_ind = (v.reshape(nx*ny,2)+R).astype(int)
    new_ind[new_ind>=nx] = nx-1
    return np.array([u[tuple(r)]  for r in new_ind])