import numpy as np

def circulant(x):
    
    dim = np.size(x, axis=0)
    
    assert np.shape(x) == (dim,)
    
    circ = np.zeros(shape=(dim, dim), dtype=x.dtype)
    
    for k in range(dim):
        circ[k, :] += np.roll(x, k, axis=0)
        
    return circ


if __name__ == '__main__':
    
    a = np.array([1, 2, 3])
    mat = circulant(a)
    print('Given vector a = \n')
    print(a)
    print('\nThen circulant(a) = \n')
    print(mat)