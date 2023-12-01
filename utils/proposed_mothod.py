import imp
import torch
import torch.nn as nn
import numpy as np
import math
import os
import sys
import copy
import torch.nn.init as init


def get_gradient(model, data_loader, criterion, device):
    model.eval()
    gradients=[]
    model = model.to(device)
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        l = 0
        for k, m in list(model.named_modules()):
            if (isinstance(m, nn.Conv2d) and 'shortcut' not in k) or isinstance(m, nn.Linear):
                if batch_idx == 0:
                    gradients.append(m.weight.grad.data.abs()) 
                else:
                    gradients[l] = gradients[l] + m.weight.grad.data.abs()
                    l += 1          

    return gradients


def filter_selection(secret_model ,secret_dataloader, stego_dataloader, prop, device, lambda_g=1e-2, criterion=nn.CrossEntropyLoss()):
    ''' 
    Return the B_stream which records the position of the selected filters and the sparse_masks which records the position of the selected neurons.
    A filters consists of c x s x s neurons, wheres c is the channel and s is the kernel size.
    '''
    # calculating the gradients of weights on the secret dataset
    secret_grad = get_gradient(secret_model, secret_dataloader, criterion, device)   # GoS
    # calculating the gradients of weights on the stego dataset
    stego_grad = get_gradient(secret_model, stego_dataloader, criterion, device)   # GoC
    L = len(secret_grad) # the total number of the conv and fc layers

    # merging the gardients of the two task
    Alpha = [] # for recording the score of the filters.
    for l in range(L):
        nof = secret_grad[l].shape[0] # the number of filters in l-th conv layer   
        Alpha_l = []  # for recording the score of the filters in l-th conv layer.
        if l < L-2:   #  for secret model, its 0~(L-2)-th layers are conv layers, whose weights' gradients are 4-dim tensors
            for d in range(nof): 
                GOE_ld = avg(secret_grad[l][d, :, :, :]) + avg(secret_grad[l+1][:, d, :, :])
                GDT_ld = avg(stego_grad[l][d, :, :, :]) + avg(stego_grad[l+1][:, d, :, :])
                Alpha_l.append(GOE_ld - lambda_g * GDT_ld) 
        if l == L-2:   #  for secret model, its (L-1)-th layer (final layer) is fc layer, whose weights' gradients are 2-dim tensors
            for d in range(nof): 
                GOE_ld = avg(secret_grad[l][d, :, :, :]) + avg(secret_grad[l+1][:, d])
                GDT_ld = avg(stego_grad[l][d, :, :, :]) + avg(stego_grad[l+1][:, d])
                Alpha_l.append(GOE_ld - lambda_g * GDT_ld) 
        if Alpha_l:
            Alpha.append(Alpha_l)

    # Selecting filters
    B_stream = [] # for recoding the position of the selected filters
    for l in range(len(Alpha)):
        nof = len(Alpha[l])
        nosf = int(prop * nof) # the number of the selected filters
        idxs_sf = top_K(Alpha[l], nosf) # the indexs of the selected filters in l-th layer
        b_l = torch.zeros(nof)
        for idx in idxs_sf:
            b_l[idx] = 1
        B_stream.append(b_l)

    sparse_masks = {}
    conv_masks = B_stream_to_sparse_masks(secret_grad, B_stream)
    bn_masks = copy.deepcopy(B_stream)
    sparse_masks['conv'] = conv_masks
    sparse_masks['bn'] = bn_masks
    # processing the shortcuts in resnet18.
    if has_short_connection(secret_model):
        connection_points = [[4, 6], [8, 10], [12, 14]] # resnet18 has 3 shortcut which consists of conv and bn layers， the first shortcut skips from the (4+1)-th to the (6+1)-th layers of backbone.
        sc_conv_masks, sc_bn_masks = B_stream_to_sparse_masks_sc(B_stream, connection_points)
        sparse_masks['sc_conv'] = sc_conv_masks
        sparse_masks['sc_bn'] = sc_bn_masks

    return B_stream, sparse_masks


def B_stream_to_sparse_masks(gradients, B_stream):
    Sparse_masks = []
    nol = len(gradients) # the number of the conv and fc layers in the model
    for l in range(len(B_stream)):
        mask = torch.ones_like(gradients[l]) 
        idxs_f = np.squeeze(np.argwhere((1-B_stream[l]).numpy())).tolist() # the indexs of the selected filters in current layer
        if l == 0: # for the first conv layer
            mask[idxs_f, :, :, :] = 0
        else: # for the mid conv layers
            idxs_c = np.squeeze(np.argwhere((1-B_stream[l-1]).numpy())).tolist() # the indexs of the selected channels in l-th layer, which is euqal to the indexs of the selected filters in (l-1)-th layer.
            mask[:, idxs_c, :, :] = 0
            mask[idxs_f, :, :, :] = 0
        Sparse_masks.append(mask)

    # for the last fc layer 
    idxs_c = np.squeeze(np.argwhere((1-B_stream[len(B_stream)-1]).numpy())).tolist()
    mask = torch.ones_like(gradients[len(gradients)-1]) 
    mask[:, idxs_c] = 0
    Sparse_masks.append(mask)

    return Sparse_masks


def B_stream_to_sparse_masks_sc(B_stream, connection_points):
    '''
    generating the sparse masks for the conv and bn layers in the shortcut connection
    '''
    conv_masks = []
    bn_masks = []
    for l in range(len(connection_points)):
        nof = len(B_stream[connection_points[l][1]]) # the number of filters of the conv layer in the l-th shortcut.
        noc = len(B_stream[connection_points[l][0]]) # the number of channels of the conv layer in the l-th shortcut.
        sc_conv_kernels = 1
        mask = torch.ones(nof, noc, sc_conv_kernels, sc_conv_kernels) 
        idxs_f = np.squeeze(np.argwhere((1-B_stream[connection_points[l][1]]).numpy())).tolist() # the indexs of the selected filters in current layer
        idxs_c = np.squeeze(np.argwhere((1-B_stream[connection_points[l][0]]).numpy())).tolist() # the indexs of the selected channels in l-th layer, which is euqal to the indexs of the selected filters in (l-1)-th layer.
        mask[:, idxs_c, :, :] = 0
        mask[idxs_f, :, :, :] = 0
        conv_masks.append(mask)
        bn_masks.append(B_stream[connection_points[l][1]])
    
    return conv_masks, bn_masks


def top_K(arraylist, K):
    '''
    Return the index of the first k max_number
    '''
    maxlist=[]
    maxlist_id=list(range(0,K))
    m=[maxlist,maxlist_id]
    for i in maxlist_id:
        maxlist.append(arraylist[i])

    for i in range(K,len(arraylist)):
        if arraylist[i]>min(maxlist):
            mm=maxlist.index(min(maxlist))
            del m[0][mm]
            del m[1][mm]
            m[0].append(arraylist[i])
            m[1].append(i)
    return maxlist_id


def avg(tensor):
    # get mean of tensor
    return tensor.data.cpu().mean().numpy().tolist()


def has_bn_layer(model):
    for m in list(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            return True
    return False


def has_short_connection(model):
    for k, m in list(model.named_modules()):
        if 'shortcut' in k and isinstance(m, nn.Conv2d):
            return True
    return False


def secret_model_extraction(model, B_stream):

    model = model.cpu()
    sub_model = copy.deepcopy(model)
    l_c = 0 # for conv and fc layers in backbone
    l_b = 0 # for bn layers in backbone
    l_c_sc = 0 # for conv and fc layers in shortcut
    l_b_sc = 0 # for bn layers in shortcut
    connection_points = [[4, 6], [8, 10], [12, 14]] # for resnet18 which has 3 shortcuts that consists of conv and bn layers， the first shortcut skips from the (4+1)-th to the (6+1)-th layers of backbone.
    for [(k, m), (k_s, m_s)] in zip(model.named_modules(), sub_model.named_modules()):

        if isinstance(m, nn.Conv2d):
            if 'shortcut' not in k:
                idxs_f = np.squeeze(np.argwhere(B_stream[l_c].numpy())).tolist() # the indexs of the selected filters in l-th layer
                if l_c == 0: # for the first conv layer
                    extracted_filters = m.weight.data[idxs_f, :, :, :]
                else: # for other conv layers
                    idxs_c = np.squeeze(np.argwhere(B_stream[l_c-1].numpy())).tolist() # the indexs of the selected channels in l-th layer, which is euqal to the indexs of the selected filters in (l-1)-th layer.
                    extracted_filters = m.weight.data[:, idxs_c, :, :]
                    extracted_filters = extracted_filters[idxs_f, :, :, :]
                m_s.weight = nn.Parameter(extracted_filters)
                l_c += 1
            else:
                idxs_c = np.squeeze(np.argwhere(B_stream[connection_points[l_c_sc][0]].numpy())).tolist() # the indexs of the selected filters in l-th layer
                idxs_f = np.squeeze(np.argwhere(B_stream[connection_points[l_c_sc][1]].numpy())).tolist() # the indexs of the selected filters in l-th layer
                extracted_filters = m.weight.data[:, idxs_c, :, :]
                extracted_filters = extracted_filters[idxs_f, :, :, :]
                m_s.weight = nn.Parameter(extracted_filters)
                l_c_sc += 1

        elif isinstance(m, nn.BatchNorm2d):
            if 'shortcut' not in k:
                # for weight, bias, running_means and running vars in bn layers
                idxs_f = np.squeeze(np.argwhere(B_stream[l_b].numpy())).tolist() # the indexs of the selected filters in l-th layer
                weight_tmp = m.weight.data[idxs_f]
                m_s.weight = nn.Parameter(weight_tmp)
                bias_tmp = m.bias.data[idxs_f]
                m_s.bias = nn.Parameter(bias_tmp)
                running_mean_tmp = m.running_mean[idxs_f]
                m_s.running_mean = running_mean_tmp
                running_var_tmp = m.running_var[idxs_f]
                m_s.running_var = running_var_tmp
                l_b += 1
            else:
                # for weight, bias, running_means and running vars in bn layers
                idxs_f = np.squeeze(np.argwhere(B_stream[connection_points[l_b_sc][1]].numpy())).tolist() # the indexs of the selected filters in l-th layer
                weight_tmp = m.weight.data[idxs_f]
                m_s.weight = nn.Parameter(weight_tmp)
                bias_tmp = m.bias.data[idxs_f]
                m_s.bias = nn.Parameter(bias_tmp)
                running_mean_tmp = m.running_mean[idxs_f]
                m_s.running_mean = running_mean_tmp
                running_var_tmp = m.running_var[idxs_f]
                m_s.running_var = running_var_tmp
                l_b_sc += 1

        elif isinstance(m, nn.Linear):
            idxs_c = np.squeeze(np.argwhere(B_stream[l_c-1].numpy())).tolist() # the indexs of the selected channels in the last fc layer.
            extracted_filters = m.weight.data[:, idxs_c]
            m_s.weight = nn.Parameter(extracted_filters)

    return sub_model


def secret_model_embedding(model, tuned_sub_model, B_stream, sparse_masks):

    for masks in sparse_masks.values():
        for i in range(len(masks)):
            masks[i] = masks[i].cpu()    
    tuned_sub_model = tuned_sub_model.cpu()
    l_c = 0 # for conv and fc layers in backbone
    l_b = 0 # for bn layers in backbone
    l_c_sc = 0 # for conv and fc layers in shortcut
    l_b_sc = 0 # for bn layers in shortcut
    connection_points = [[4, 6], [8, 10], [12, 14]] # for resnet18 which has 3 shortcuts that consists of conv and bn layers， the first shortcut skips from the (4+1)-th to the (6+1)-th layers of backbone.
    for [(k, m), (k_s, m_s)] in zip(model.named_modules(), tuned_sub_model.named_modules()):
        if isinstance(m, nn.Conv2d):
            if 'shortcut' not in k:
                idxs_f = np.squeeze(np.argwhere(B_stream[l_c].numpy())).tolist() # the indexs of the selected filters in l-th layer
                if l_c == 0: # for the first conv layer
                    m.weight.data[idxs_f, :, :, :] = m_s.weight.data
                else: # for other conv layers
                    idxs_c = np.squeeze(np.argwhere(B_stream[l_c-1].numpy())).tolist() # the indexs of the selected channels in l-th layer, which is euqal to the indexs of the selected filters in (l-1)-th layer.
                    zero_padding_weight_1 = torch.zeros(m_s.weight.shape[0], m.weight.shape[1], m.weight.shape[2], m.weight.shape[3])
                    zero_padding_weight_1[:, idxs_c, :, :] = m_s.weight.data
                    zero_padding_weight_2 = torch.zeros_like(m.weight)
                    zero_padding_weight_2[idxs_f, :, :, :] = zero_padding_weight_1
                    m.weight.data = zero_padding_weight_2 + m.weight.data.mul_(1-sparse_masks['conv'][l_c])
                l_c += 1
            else:
                idxs_c = np.squeeze(np.argwhere(B_stream[connection_points[l_c_sc][0]].numpy())).tolist() # the indexs of the selected filters in l-th layer
                idxs_f = np.squeeze(np.argwhere(B_stream[connection_points[l_c_sc][1]].numpy())).tolist() # the indexs of the selected filters in l-th layer
                zero_padding_weight_1 = torch.zeros(m_s.weight.shape[0], m.weight.shape[1], m.weight.shape[2], m.weight.shape[3])
                zero_padding_weight_1[:, idxs_c, :, :] = m_s.weight.data
                zero_padding_weight_2 = torch.zeros_like(m.weight)
                zero_padding_weight_2[idxs_f, :, :, :] = zero_padding_weight_1
                m.weight.data = zero_padding_weight_2 + m.weight.data.mul_(1-sparse_masks['sc_conv'][l_c_sc])
                l_c_sc += 1

        
        elif isinstance(m, nn.BatchNorm2d):
            if 'shortcut' not in k:
                # for weight and bias in bn layers
                idxs_f = np.squeeze(np.argwhere(B_stream[l_b].numpy())).tolist() # the indexs of the selected filters in l-th layer
                m.weight.data[idxs_f] = m_s.weight.data
                m.bias.data[idxs_f] = m_s.bias.data
                l_b += 1
            else:
                idxs_f = np.squeeze(np.argwhere(B_stream[connection_points[l_b_sc][1]].numpy())).tolist()
                m.weight.data[idxs_f] = m_s.weight.data
                m.bias.data[idxs_f] = m_s.bias.data
                l_b_sc += 1
        
        elif isinstance(m, nn.Linear):
            idxs_c = np.squeeze(np.argwhere(B_stream[l_c-1].numpy())).tolist() # the indexs of the selected channels in the last fc layer.
            m.weight.data[:, idxs_c] = m_s.weight.data

    return model


def side_info_embedding(model, B_stream, key):
    '''
    embedding B_stream into the stego-model
    '''
    # concat b_1, b_2..., b_L in B_stream
    One_dim_B_stream = torch.Tensor([])
    for b_l in B_stream:
        One_dim_B_stream = torch.concat((One_dim_B_stream, b_l), dim=0)
    nob = One_dim_B_stream.shape[0] # the number of bits

    weight_values_stream = host_weight_extraction(model)
    # select key-th to (key+nob)-th weights for hosting B_stream
    host_weights = weight_values_stream[key:key+nob]

    # each value in the host_weights will be embedded with a bit, and the LSB algorithm is used to embed the bit into the fourth decimal place of the weight value.
    weights_resi = torch.zeros_like(host_weights)
    for i in range(len(host_weights)):
        # extract the sixth decimal place (sdp)
        sdp = int(str(host_weights[i].item()).split('.')[-1][3])
        if (One_dim_B_stream[i] == 0 and sdp % 2 == 1) or (One_dim_B_stream[i] == 1 and sdp % 2 == 0):
            weights_resi[i] += 1e-4

    stego_weights = host_weights + weights_resi

    weight_values_stream[key:key+nob] = stego_weights
    
    # Feedback the changes of weights to model's parameters to complete embedding.
    start_idx = 0
    for k, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d) and 'shortcut' not in k:
            num_weights_in_m = m.weight.numel()
            m.weight.data = weight_values_stream[start_idx:start_idx+num_weights_in_m].view(m.weight.shape)
            start_idx += num_weights_in_m

    return model


def side_info_extraction(model, key):
    '''
    extracting B_stream from the stego-model
    '''
    
    len_B = [] # a list that records the lenght of b_1, b_2..., b_L in B_stream
    for m in list(model.modules()):
        if isinstance(m, nn.Conv2d):
            len_l = m.weight.shape[0]
            len_B.append(len_l)
    
    nob = np.sum(np.array(len_B)) # the number of bits
    
    weight_values_stream = host_weight_extraction(model)

    # select key-th to (key+nob)-th weights for extracting B_stream
    stego_weights = weight_values_stream[key:key+nob]

    One_dim_B_stream = torch.zeros(nob)
    for i in range(len(stego_weights)):
        # extract the sixth decimal place (sdp)
        sdp = int(str(stego_weights[i].item()).split('.')[-1][3])
        if sdp % 2 == 1:
            One_dim_B_stream[i] = 1.

    B_stream = []
    len_B.append(0)
    start = 0; end = len_B[0]
    # split One_dim_B_stream to b_1, b_2..., b_L in B_stream
    for l in range(len(len_B)-1):
        b_l = One_dim_B_stream[start:end]
        B_stream.append(b_l)
        start += len_B[l]; end += len_B[l+1]

    return B_stream


def host_weight_extraction(model):
    # model: secret model
    weight_values_stream = torch.Tensor([])
    for k, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d) and 'shortcut' not in k:
            weight_values_stream = torch.concat((weight_values_stream, m.weight.data.clone().view(-1).cpu()))
            if m.bias != None:
                weight_values_stream = torch.concat((weight_values_stream, m.bias.data.clone().view(-1).cpu()))
    return weight_values_stream








