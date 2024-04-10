import torch
import matplotlib.pyplot as plt

def generate_permutation(y):
    shuffled_array = y.clone().cuda()

    while torch.equal(shuffled_array, y):
        shuffled_array = y[torch.randperm(y.shape[0])]

    return shuffled_array

def overlay_y_on_x(x, y, num_classes):
    x_ = x.clone()
    y=torch.squeeze(y)
    if y.dim() == 2:
        x_[:, :num_classes] *= 0.0
        y=torch.argmax(y, dim=1)
        x_[range(x.shape[0]), y] = x.max()
    else:
        x_[:, :num_classes] *= 0.0
        x_[range(x.shape[0]), y] = x.max()
    return x_

def concat_y_and_x(x, y, num_classes):
    x_ = torch.ones_like(x[:,:1,:,:])
    y = torch.squeeze(y)
    if y.dim() == 2:
        y=torch.argmax(y, dim=1)

    for i in range(y.shape[0]):
        x_[i] *= y[i]*0.1

    concat = torch.concat([x,x_],1)
    return concat


def generate_contrastive_pairs_y(x, y, num_classes, dims,  labelling=True, random=False):
    if labelling:
        if x.dim() == 2:
            x_pos = overlay_y_on_x(x, y, num_classes)
            if random == True:
                rnd = torch.randperm(x.size(0))
                x_neg = overlay_y_on_x(x, y[rnd], num_classes)
            else:
                y_shuffled = generate_permutation(y)
                x_neg = overlay_y_on_x(x, y_shuffled, num_classes)
        else:
            x_pos = concat_y_and_x(x, y, num_classes)
            if random == True:
                rnd = torch.randperm(x.size(0))
                x_neg = concat_y_and_x(x, y[rnd], num_classes)
            else:
                y_shuffled = generate_permutation(y)
                x_neg = concat_y_and_x(x, y_shuffled, num_classes)
    else:
        x_pos = x
        # if labelling False, the pairs are still never the matching ones
        if random == True:
            rnd = torch.randperm(x.size(0))
            x_neg = x[rnd]
        else:
            x_neg = generate_permutation(x)
    return x_pos, x_neg

def generate_contrastive_pairs_x(x, y, num_classes, dims,  labelling=True, random=False):
    if labelling:
        if x.dim() == 2:
            x_pos = overlay_y_on_x(x, y, num_classes)
            if random == True:
                rnd = torch.randperm(x.size(0))
                x_neg = overlay_y_on_x(x[rnd], y, num_classes)
            else:
                x_shuffled = generate_permutation(x)
                x_neg = overlay_y_on_x(x_shuffled, y, num_classes)
        else:
            x_pos = concat_y_and_x(x, y, num_classes)
            if random == True:
                rnd = torch.randperm(x.size(0))
                x_neg = concat_y_and_x(x[rnd], y, num_classes)
            else:
                x_shuffled = generate_permutation(x)
                x_neg = concat_y_and_x(x_shuffled, y, num_classes)
    else:
        x_pos = x
        # if labelling False, the pairs are still never the matching ones
        if random == True:
            rnd = torch.randperm(x.size(0))
            x_neg = x[rnd]
        else:
            x_neg = generate_permutation(x)
    return x_pos, x_neg


def generate_hybrid_contrastive_pairs(x,y,num_classes, dims, labelling=True):
    if x.dim() == 2:
        noise = torch.rand(x.shape).view(-1, dims[0], dims[1], dims[2])
    else:
        noise = torch.rand(x.shape)

    if dims[0] == 1:
        blur_kernel = torch.tensor([0.25, 0.5, 0.25])
    elif dims[0] == 3:
        blur_kernel = torch.tensor([0.25, 0.5, 0.25]).repeat(3, 1, 1, 1)
        blur_kernel = blur_kernel.transpose(2, 3)
    
    blurred_noise = noise
    for i in range(5):
        if dims[0] == 1:
            blurred_noise = torch.nn.functional.conv2d(blurred_noise, blur_kernel.view(1, 1, 3, 1).transpose(2, 3), padding=(1, 0))
            blurred_noise = torch.nn.functional.conv2d(blurred_noise, blur_kernel.view(1, 1, 1, 3).transpose(2, 3), padding=(0, 1))
        elif dims[0] == 3:
            blurred_noise = torch.nn.functional.conv2d(blurred_noise, blur_kernel, padding=(1, 0), groups=dims[0])
            blurred_noise = torch.nn.functional.conv2d(blurred_noise, blur_kernel.transpose(2, 3), padding=(0, 1), groups=dims[0])
    blurred_noise_binary =  (blurred_noise > 0.5).float().cuda()
    #blurred_noise_binary =  blurred_noise.float().cuda()
    
    blurred_noise_inverted = 1 - blurred_noise_binary.cuda()

    if x.dim() == 2:
        hybrid_samples = x * blurred_noise_binary.flatten(start_dim=1) + generate_permutation(x).cuda() * blurred_noise_inverted.flatten(start_dim=1)
    else:
        hybrid_samples = x * blurred_noise_binary + generate_permutation(x).cuda() * blurred_noise_inverted

    if labelling:
        if x.dim() == 2:
            x_pos = overlay_y_on_x(x, y, num_classes)
            x_neg = overlay_y_on_x(hybrid_samples, y, num_classes)
        else:
            x_pos = concat_y_and_x(x, y, num_classes)
            x_neg = concat_y_and_x(hybrid_samples, y, num_classes)
    else:
        x_pos = x
        x_neg = hybrid_samples
    
    return x_pos, x_neg

def mask_random_quadratic_area(image, max_mask_size=8):
    
    C, H, W = image.shape
    
    max_mask_size = min(max_mask_size, H, W)
    
    mask_size = torch.randint(1, max_mask_size + 1, (1,)).item()

    top_left_y = torch.randint(0, H - mask_size + 1, (1,)).item()
    top_left_x = torch.randint(0, W - mask_size + 1, (1,)).item()

    image[:, top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = 0
    
    return image

def add_gaussian_noise(image):
    mean = 0.0
    std = 0.05
    
    device = image.device
    noise = torch.randn(image.size(), device=device) * std + mean
    noisy_image = image + noise
    
    return noisy_image