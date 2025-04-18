import torch

# Helper function, makes circulant matrix of input signal.
def circulant(tensor, dim):
    """get a circulant version of the tensor along the {dim} dimension.
    The additional axis is appended as the last dimension.
    E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat(
        [
            tensor.flip((dim,)), 
            torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1,)
        ], 
        dim=dim,
    )
    return tmp.unfold(dim, S, 1).flip((-1,))