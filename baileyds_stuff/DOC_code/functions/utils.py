import torch
import numpy as np

#works with get grads from adam
def n_params(model, verbose=False):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num = sum([np.prod(p.size()) for p in model_parameters])
    
    if verbose == True:
        for name, p in model.named_parameters():
            print(name, 'size:', p.numel())
        print('Total num. parameters:', num) 

    return num

def get_param_state(model):
    state = np.zeros((n_params(model)))
    idx = 0
    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    with torch.no_grad():
        #used to be model.parameters()
        for param in model.parameters():
            this_param = param.cpu().numpy().flatten()
            for i in range(len(this_param)):
                state[idx] = this_param[i]
                idx += 1
    return state

def state_dict_to_vector(state_dict):
    num_params = sum([len(t.flatten()) for t in state_dict.values()])
    state = np.zeros((num_params))
    idx = 0
    for tensor in state_dict.values():
        this_param = tensor.cpu().numpy().flatten()
        #for i in range(len(this_param)):
        state[idx:idx+len(this_param)] = this_param
        idx += len(this_param)
    return state
    
def vector_to_state_dict(param_state, model):
    #print('help')
    param_info = [(item[0], list(item[1].shape)) for item in model.named_parameters()]
    idx = 0
    dict = {}
    #print('param state shape:', param_state.shape)
    for name,shape in param_info:

        this_num = 1
        for i in range(len(shape)):
            this_num *= shape[i]
        #print('test', name, shape, this_num)
        this_param = torch.from_numpy(param_state[idx:idx+this_num].reshape(shape))
        dict[name] = this_param
        idx += this_num
        
    for name,p in model.state_dict().items():
        if name not in dict:
            dict[name] = p
    
    return dict
