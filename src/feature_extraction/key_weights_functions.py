import torch
from torch import Tensor


def weights_function_v1(distances: Tensor, bias = 4, scale = 1.8) -> Tensor:
    """
    Arguments:
    ----------
    distances: Tensor
        A 2d tensor where i-th element is a vector of distances 
        for i-th swipe dot and each of keyboard keys. It's supposed
        that distances are measured in half_key_diagonals 
        (distances = raw_distances / half_key_diagonal)

    $$f(x) = \frac{1}{1+e^{\frac{s \cdot x}{key\_radius} - b}}$$
    b = bias = 4
    s = scale = 1.8
    """
    
    # return 1 / (1 + torch.exp(1.8 * distances - 4))

    #! It may be a good idea to move division by half_key_diag outside
    # this function.  The division is just a scaling of distances
    # so that they are not in pixels but use half_key_diag as a unit. 
    sigmoid_input = distances * (-scale) + bias
    return torch.nn.functional.sigmoid(sigmoid_input)


def weights_function_v1_softmax(distances: Tensor, bias = 4, scale = 1.8) -> Tensor:
    """
    Arguments:
    ----------
    distances: Tensor
        A 2d tensor where i-th element is a vector of distances 
        for i-th swipe dot and each of keyboard keys. It's supposed
        that distances are measured in half_key_diagonals 
        (distances = raw_distances / half_key_diagonal)
    """
    mask = torch.isinf(distances)
    sigmoid_input = distances * (-scale) + bias
    weights = torch.nn.functional.sigmoid(sigmoid_input)
    # -inf to zero out unpresent values and have a sum of one 
    weights.masked_fill_(mask, float('-inf'))
    return torch.nn.functional.softmax(weights, dim=1)



def weights_function_sigmoid_normalized_v1(distances: Tensor, 
                                           bias = 4, scale = 1.8) -> Tensor:
    """
    Arguments:
    ----------
    distances: Tensor
        A 2d tensor where i-th element is a vector of distances 
        for i-th swipe dot and each of keyboard keys. It's supposed
        that distances are measured in half_key_diagonals 
        (distances = raw_distances / half_key_diagonal)
    
    $$f(x) = \frac{1}{1+e^{\frac{s \cdot x}{key\_radius} - b}}$$
    b = bias = 4
    s = scale = 1.8
    """
    #! It may be a good idea to move division by half_key_diag outside
    # this function.  The division is just a scaling of distances
    # so that they are not in pixels but use half_key_diag as a unit. 
    sigmoid_input = distances * (-scale) + bias
    sigmoidal_weights = torch.nn.functional.sigmoid(sigmoid_input)
    weights = sigmoidal_weights / sigmoidal_weights.sum(dim=1, keepdim=True)
    return weights

