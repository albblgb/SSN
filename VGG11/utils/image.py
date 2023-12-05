import numpy as np
import cv2
import math
import torch.nn as nn
import torch

def quantization(tensor):
    return torch.round(torch.clamp(tensor*255, min=0., max=255.))/255

# def quantization_v2(tensor):
#     return torch.round(255 * (tensor - tensor.min()) / (tensor.max() - tensor.min()))/255

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def Fibonacci(n):
   if n == 1 or n == 2:
       return 1
   else:
       return(Fibonacci(n-1) + Fibonacci(n-2))
   

def cat_map(tensor, p=1, q=1, epoch=10, obfuscate=True):
    '''
    key k is composed by p, q and epoch
    tensor: batch x c x h x w
    '''
    matric = torch.tensor([[Fibonacci(2*epoch-1), Fibonacci(2*epoch)], [Fibonacci(2*epoch), Fibonacci(2*epoch+1)]])
    matric_inverse = torch.tensor([[Fibonacci(2*epoch+1), -1*Fibonacci(2*epoch)], [-1*Fibonacci(2*epoch), Fibonacci(2*epoch-1)]])
    # print(torch.mm(matric, matric_inverse))
    
    if obfuscate != True: # restore pert
        matric = matric_inverse

    h = tensor.shape[2]; w = tensor.shape[3]

    processed_tensor = torch.zeros_like(tensor)
    for x in range(0, h):
        for y in range(0, w):
            new_x, new_y = torch.mm(matric, torch.tensor(([x], [y]))) % h
            processed_tensor[:, :, new_x.item(), new_y.item()] = tensor[:, :, x, y]
    
    return processed_tensor



def calculate_rmse(img1, img2):
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    rmse = np.sqrt(mse)

    return np.mean(rmse)

def calculate_rmses(t1, t2):
    rmes_list = []
    for i in range(t1.shape[0]):
        rmes_list.append(calculate_rmse(t1[i], t2[i]))
    return np.mean(np.array(rmes_list))


def calculate_mae(img1, img2):

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    apd = np.mean(np.abs(img1 - img2))
    if apd == 0:
        return float('inf')

    return np.mean(apd)

def calculate_maes(t1, t2):
    mae_list = []
    for i in range(t1.shape[0]):
        mae_list.append(calculate_mae(t1[i], t2[i]))
    return np.mean(np.array(mae_list))

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnrs(t1, t2):
    psnr_list = []
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]//3):
            psnr_list.append(calculate_psnr(t1[i][3*j:3*(j+1)], t2[i][3*j:3*(j+1)]))
    return np.mean(np.array(psnr_list))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssims(t1, t2):
    ssim_list = []
    for i in range(t1.shape[0]):
        for j in range(t1.shape[1]//3):
            # print(t1[i][3*j:3*(j+1)].shape)
            ssim_list.append(calculate_ssim(t1[i][3*j:3*(j+1)], t2[i][3*j:3*(j+1)]))
    return np.mean(np.array(ssim_list))


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = img1.transpose((1, 2, 0))
    img2 = img2.transpose((1, 2, 0))

    # print(img1.shape)
    # print(img2.shape)
    # bk
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    

# def calculate_ssim(numpy1, numpy2): # for 4-dim numpy
#     if numpy1.ndim== 4:
#         sum = 0
#         for i in range(numpy1.shape[0]):
#             sum += calculate_ssim_img(numpy1[i, :, :, :], numpy2[i, :, :, :])
        
#         return sum/numpy1.shape[0]
#     return calculate_ssim_img(numpy1, numpy2)
    


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)
    

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2


    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
    


############################################################################
#                        For secret image compression                      #
############################################################################
def pixel_unshuffle(input, upscale_factor=2):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.
    written by: Zhaoyi Yan, https://github.com/Zhaoyi-Yan
    and Kai Zhang, https://github.com/cszn/FFDNet
    01/01/2019
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)
 

class rgb_to_ycbcr(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Output:
        result(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(rgb_to_ycbcr, self).__init__()
        matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ], dtype=np.float32).T

        self.shift = nn.Parameter(torch.tensor([0., 0.5, 0.5]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        result.view(image.shape)
        return result


class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        # batch x 6 x height/2 x width/2 
    """

    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image = image.permute(0, 3, 1, 2).clone()   # batch x 3 x height x width 
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        cb = avg_pool(image[:, 1, :, :].unsqueeze(1))   # batch x 1 x height/2 x width/2 
        cr = avg_pool(image[:, 2, :, :].unsqueeze(1))   # batch x 1 x height/2 x width/2 

        y = pixel_unshuffle(image[:, 0, :, :].unsqueeze(1))          # batch x 4 x height/2 x width/2 

        return torch.concat((y, cb, cr), dim=1)         # batch x 6 x height/2 x width/2 


class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input:
        image(tensor): # batch x 6 x height/2 x width/2  ycbcr
    Output:
        image(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, ycbcr):
        
        ps = torch.nn.PixelShuffle(2)
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x
        
        y = ps(ycbcr[:, :4, :, :])   # batch x 1 x height x width
        cb = repeat(ycbcr[:, 4, :, :]).unsqueeze(1)# batch x 1 x height x width
        cr = repeat(ycbcr[:, 5, :, :]).unsqueeze(1) # batch x 1 x height x width

        return torch.cat([y, cb, cr], dim=1).permute(0, 2, 3, 1)


class ycbcr_to_rgb(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Output:
        result(tensor): batch x 3 x height x width
    """

    def __init__(self):
        super(ycbcr_to_rgb, self).__init__()
        matrix = np.array([
            [1., 0., 1.402],
            [1, -0.344136, -0.714136],
            [1, 1.772, 0]
        ], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -0.5, -0.5]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)
    
    
rgb2ycbcr = rgb_to_ycbcr()
c_down = chroma_subsampling()
c_up = chroma_upsampling()
ycbcr2rgb = ycbcr_to_rgb()


def image_downsampling(tensor):
    # input:4-dim tensor(batch x 3 x height x width) rgb
    # output: 4-dim tensor(batch x 6 x height/2 x width/2) ycbcr
    return c_down(rgb2ycbcr(tensor))


def image_upsampling(tensor):
    # input: 4-dim tensor(batch x 6 x height/2 x width/2) ycbcr
    # output: 4-dim tensor(batch x 3 x height x width)  rgb
    return ycbcr2rgb(c_up(tensor))


