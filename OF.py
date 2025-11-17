import cv2
from scipy.sparse import csr_matrix, eye
import numpy as np
from copy import deepcopy
from scipy.ndimage import gaussian_filter

def gen_of(img1, img2, shape, smooth_kernel_size=5, smooth_sigma=20):
    processed = []
    for img in [img1, img2]:
        img = deepcopy(img).reshape(shape)
        img = np.asarray(img)
        # kernel_t = np.ones((1,1))
        # kernel_t /= kernel_t.size   # normalize so it's an average
        # # smoothed version of the image
        # img = signal.convolve2d(img, kernel_t, boundary='symm', mode='same')   
        img_min = np.min([img1,img2])
        img_max = np.max([img1,img2])
        if img_max == img_min:
            img_scaled = np.zeros_like(img)
        else:
            img_scaled = 255 * (img - img_min) / (img_max - img_min)
        img_uint8 = img_scaled.astype(np.uint8)
        # Smooth the image
        img_smoothed = gaussian_filter(img_uint8, sigma=2) # img_uint8 #img_uint8 #cv2.GaussianBlur(img_uint8,ksize=(smooth_kernel_size, smooth_kernel_size), sigmaX=smooth_sigma) #img_uint8 #cv2.GaussianBlur(img_uint8,ksize=(smooth_kernel_size, smooth_kernel_size), sigmaX=smooth_sigma)
        processed.append(img_smoothed)

    img1, img2 = processed
    nx = shape[1]

    tv_l1 = cv2.optflow.DualTVL1OpticalFlow.create()#cv2.cuda_OpticalFlowDual_TVL1.create() #
    tv_l1.setLambda(.001)
    #tv_l1.setTheta(0.1)
    tv_l1.setScalesNumber(10)
    tv_l1.setScaleStep(0.9)
    tv_l1.setWarpingsNumber(1)
    tv_l1.setInnerIterations(20)
    tv_l1.setOuterIterations(20)
    # tv_l1.setMedianFiltering(5)
    tv_l1.setUseInitialFlow(True)
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2, None,
        pyr_scale=0.9,
        levels=10,
        winsize =5 * (nx // 50) if nx >= 50 else 5,
        iterations=10,
        poly_n=7,
        poly_sigma=1.2,
        flags=0
    )
    
    flow = tv_l1.calc(img1, img2, flow)
    return flow
def build_M2_safe(flow, img_shape):
    """
    Same as build_M2, but preserves original pixel values where flow is invalid.
    """
    h, w = img_shape
    assert flow.shape[:2] == (h, w), "Flow shape must match image shape"

    # Create coordinate grid
    y, x = np.mgrid[0:h, 0:w]
    
    # Compute source (warped) coordinates
    src_x = x + flow[:, :, 0]
    src_y = y + flow[:, :, 1]

    # Identify valid interpolation range
    valid = (src_x >= 0) & (src_x <= w - 2) & (src_y >= 0) & (src_y <= h - 2)

    # Flatten everything
    src_x = src_x[valid]
    src_y = src_y[valid]
    dst_flat_indices = np.flatnonzero(valid)

    # Integer and fractional parts for bilinear interpolation
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = src_x - x0
    wy = src_y - y0
    wx0 = 1 - wx
    wy0 = 1 - wy

    # Interpolation weights
    w00 = wx0 * wy0
    w10 = wx  * wy0
    w01 = wx0 * wy
    w11 = wx  * wy

    # Flattened source pixel indices
    def idx(y, x): return y * w + x
    src00 = idx(y0, x0)
    src10 = idx(y0, x1)
    src01 = idx(y1, x0)
    src11 = idx(y1, x1)

    # All row indices are from destination
    rows = np.concatenate([dst_flat_indices]*4)
    cols = np.concatenate([src00, src10, src01, src11])
    data = np.concatenate([w00, w10, w01, w11])

    # Filter out-of-bound indices (in case rounding went wrong)
    in_bounds = (cols >= 0) & (cols < h*w)
    M_valid = csr_matrix((data[in_bounds], (rows[in_bounds], cols[in_bounds])), shape=(h*w, h*w))

    # Identify invalid destinations — assign identity
    all_indices = np.arange(h*w)
    invalid_dst_indices = np.setdiff1d(all_indices, dst_flat_indices)

    M_identity = csr_matrix((np.ones_like(invalid_dst_indices), 
                             (invalid_dst_indices, invalid_dst_indices)), shape=(h*w, h*w))

    # Combine valid warping and fallback identity
    M = M_valid + M_identity

    return M


def calc_M(xs):
    nx,ny = [int(np.sqrt(np.size(xs[0])))]*2
    flows = []
    for i in range(len(xs)-1):
        flow = gen_of(xs[i].reshape(nx, ny), xs[i+1].reshape(nx, ny),(nx,ny))
        flows.append(flow)
    
    if (len(xs)>2):
        Ms = [build_M2_safe(flow, (nx, ny)) for flow in flows]
    else:
        Ms = build_M2_safe(flow, (nx, ny)) # M_mat(xs[1],np.rint(flow))  #
    return Ms

