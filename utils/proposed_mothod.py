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


def side_info_embedding(key, model, B_stream, bn_running_binary):
    '''
    embedding B_stream into the stego-model
    '''
    # concat b_1, b_2..., b_L in B_stream
    One_dim_B_stream = torch.Tensor([])
    for b_l in B_stream:
        One_dim_B_stream = torch.concat((One_dim_B_stream, b_l), dim=0)
    
    One_dim_B_stream_str = ''
    for elem in One_dim_B_stream:
        One_dim_B_stream_str += str(int(elem.item()))
    One_dim_B_stream_str += bn_running_binary
    nob = len(One_dim_B_stream_str) # the number of bits

    weight_values_stream = host_weight_extraction(model)
    # select key-th to (key+nob)-th weights for hosting B_stream
    host_weights = weight_values_stream[key:key+3*nob]

    # each value in the host_weights will be embedded with a bit, and the LSB algorithm is used to embed the bit into the forth decimal place of the weight value.
    weights_resi = torch.zeros_like(host_weights)
    for i in range(nob):
        for j in range(3):
            # sdp = ((host_weights[3*i+j].abs() * 10000).floor().item()) % 10
            sdp = ((host_weights[3*i+j].abs() * 10000).floor().item()) % 10
            if (One_dim_B_stream_str[i] == '0' and sdp % 2 == 1) or (One_dim_B_stream_str[i] == '1' and sdp % 2 == 0):
                weights_resi[3*i+j] += 1e-4

    stego_weights = host_weights + weights_resi

    weight_values_stream[key:key+3*nob] = stego_weights
    
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
    for k, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d) and 'shortcut' not in k:
            len_l = m.weight.shape[0]
            len_B.append(len_l)

    nob = np.sum(np.array(len_B)) # the number of bits in B_stream

    weight_values_stream = host_weight_extraction(model)

    # select key-th to (key+nob)-th weights for extracting B_stream
    stego_weights = weight_values_stream[key:key+3*nob]

    One_dim_B_stream = torch.zeros(nob)
    for i in range(nob):
        # # extract the sixth decimal place (sdp)
        # sdp = int(str(stego_weights[i].item()).split('.')[-1][3])
        # if sdp % 2 == 1:
        #     One_dim_B_stream[i] = 1.
        count = 0
        for j in range(3):
            # sdp = int(str(stego_weights[3*i+j].item()).split('.')[-1][3])
            sdp = ((stego_weights[3*i+j].abs() * 10000).floor().item()) % 10
            if sdp % 2 == 1:
                count += 1
        if count >= 2:
            One_dim_B_stream[i] = 1
    
    B_stream = []
    len_B.append(0)
    start = 0; end = len_B[0]
    # split One_dim_B_stream to b_1, b_2..., b_L in B_stream
    for l in range(len(len_B)-1):
        b_l = One_dim_B_stream[start:end]
        B_stream.append(b_l)
        start += len_B[l]; end += len_B[l+1]

    # extract the running_mean and running_var of the bn layers
    nob_bnr = int(One_dim_B_stream.sum().item() * 24 * 2) # the number of bits in bn running_mean and running_var

    
    connection_points = [[4, 6], [8, 10], [12, 14]] # for resnet18 which has 3 shortcuts that consists of conv and bn layers， the first shortcut skips from the (4+1)-th to the (6+1)-th layers of backbone.
    for i in range(len(connection_points)):
        nob_bnr_sc = int(B_stream[connection_points[i][1]].sum().item() * 24 * 2)
        nob_bnr += nob_bnr_sc

    stego_weights = weight_values_stream[(key+3*nob):(key+3*(nob+nob_bnr))]
    bn_running_binary = ''
    for i in range(nob_bnr):
        # extract the sixth decimal place (sdp)
        count = 0
        for j in range(3):
            sdp = int(str(stego_weights[3*i+j].item()).split('.')[-1][3])
            if sdp % 2 == 1:
                count += 1
        if count >= 2:
            bn_running_binary += '1'
        else:
            bn_running_binary += '0'
    
    return B_stream, bn_running_binary



def host_weight_extraction(model):
    # model: secret model
    weight_values_stream = torch.Tensor([])
    for k, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d) and 'shortcut' not in k:
            weight_values_stream = torch.concat((weight_values_stream, m.weight.data.clone().view(-1).cpu()))
            # if m.bias != None:
            #     weight_values_stream = torch.concat((weight_values_stream, m.bias.data.clone().view(-1).cpu()))
    return weight_values_stream




def bn_running_extraction(model):
    bn_running = torch.tensor([])
    for k, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_running = torch.concat((bn_running, m.running_mean.clone().cpu()))
            bn_running = torch.concat((bn_running, m.running_var.clone().cpu()))

    non = bn_running.shape[0] # number of neurons in bn_running_mean and bn_running_var

    bn_running_binary = ''
    for i in range(non):
        # We discard the last 8 bits of the binary mantissa [26:34], introducing negligible errors that do not affect the network's performance.
        binary = (ConvertFloatToBinary(bn_running[i].item()))[2:26]
        
        binary_list = list(binary)
        for j in range(len(binary_list)): 
            if binary_list[j] != '0' and binary_list[j] != '1':
                binary_list[j] = '0'
                binary = ''.join(binary_list)
        
        bn_running_binary = bn_running_binary + binary

    return bn_running_binary


def bn_running_embedding(model, bn_running_binary):

    one_dim_bn_running = []
    for i in range(len(bn_running_binary)//24):
        binary = bn_running_binary[24*i:24*(i+1)] + '00000000'
        br_value = ConvertBinartToFloat(binary)
        one_dim_bn_running.append(br_value) 
    one_dim_bn_running = torch.from_numpy(np.array(one_dim_bn_running)).float()

    start_idx = 0
    for m in list(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            num_weights_in_m = m.weight.numel()
            m.running_mean = one_dim_bn_running[start_idx:start_idx+num_weights_in_m].view(m.weight.shape)
            start_idx += num_weights_in_m
            m.running_var = one_dim_bn_running[start_idx:start_idx+num_weights_in_m].view(m.weight.shape)
            start_idx += num_weights_in_m

    return model




''' float32 to IEEE 754 binary string '''
def ConvertFixedIntegerToComplement(fixedInterger) :#浮点数整数部分转换成补码(整数全部为正)
    return bin(fixedInterger)[2:]
 
def ConvertFixedDecimalToComplement(fixedDecimal) :#浮点数小数部分转换成补码
    fixedpoint = int(float(fixedDecimal)) / (10.0**len(fixedDecimal))
    s = ''
    while fixedDecimal != 1.0 and len(s) < 23 :
        fixedpoint = fixedpoint * 2.0
        s += str(fixedpoint)[0]
        fixedpoint = fixedpoint if str(fixedpoint)[0] == '0' else fixedpoint - 1.0
    return s
 
def ConvertToExponentMarker(number) : #阶码生成
    return bin(number + 127)[2:].zfill(8)
 
 
def ConvertFloatToBinary(floatingPoint) :#转换成IEEE754标准的数
    floatingPointString = str(floatingPoint)
    if floatingPointString.find('-') != -1 :#判断符号位
        sign = '1'
        floatingPointString = floatingPointString[1:]
    else :
        sign = '0'
    l = floatingPointString.split('.')#将整数和小数分离
    # front = ConvertFixedIntegerToComplement(int(float(l[0])))#返回整数补码
    front = ConvertFixedIntegerToComplement(int(l[0]))#返回整数补码
    rear  = ConvertFixedDecimalToComplement(l[1])#返回小数补码
    floatingPointString = front + '.' + rear #整合
    relativePos =   floatingPointString.find('.') - floatingPointString.find('1')#获得字符1的开始位置
    if relativePos > 0 :#若小数点在第一个1之后
        exponet = ConvertToExponentMarker(relativePos-1)#获得阶码
        mantissa =  floatingPointString[floatingPointString.find('1')+1 : floatingPointString.find('.')]  + floatingPointString[floatingPointString.find('.') + 1 :] # 获得尾数
    else :
        exponet = ConvertToExponentMarker(relativePos)#获得阶码
        mantissa = floatingPointString[floatingPointString.find('1') + 1: ]  # 获得尾数
    mantissa =  mantissa[:23] + '0' * (23 - len(mantissa))
    floatingPointString = '0b' + sign + exponet + mantissa
    return floatingPointString
    # return hex( int( floatingPointString , 2 ) )
 
''' IEEE 754 binary string to float32 '''
def ConvertExponent(strData):#阶码转整数
    return int(strData,2)-127

def ConverComplementToFixedDecimal(fixedStr):#字符串转小数
    count=1
    num=0
    for ch in fixedStr:
        if ch=="1":
            num+=2**(-count)
        count+=1
    return num

def ConverComplementToInteger(fixedStr):#字符串转整数
    return int(fixedStr,2)
    
def ConvertBinartToFloat(binStr): #IEEE754 浮点字符串转float
	# if strData=="00000000":
    #     return 0.0
    # binStr="".join(hex2bin_map[i] for i in strData)
    sign = binStr[0]
    exponet=binStr[1:9]#阶码
    mantissa="1"+binStr[9:]#尾数
    fixedPos=ConvertExponent(exponet)
    if fixedPos>=0: #小数点在1后面
        fixedDec=ConverComplementToFixedDecimal(mantissa[fixedPos+1:])#小数转换
        fixedInt=ConverComplementToInteger(mantissa[:fixedPos+1])#整数转换
    else: #小数点在1前面（原数在[-0.99,0.99]范围内)
        mantissa="".zfill(-fixedPos)+mantissa
        fixedDec=ConverComplementToFixedDecimal(mantissa[1:])#小数转换
        fixedInt=ConverComplementToInteger(mantissa[0])#整数转换
    fixed=fixedInt+fixedDec
    if sign=="1":
        fixed=-fixed
    return fixed



