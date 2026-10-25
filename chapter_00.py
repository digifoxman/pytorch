import time

import torch

print(torch.__version__)


def print_tensor_info(tensor):
    print(tensor)
    print("ndim: ", tensor.ndim)
    print("shape: ", tensor.shape, type(tensor.shape))
    print("itemsize: ", tensor.itemsize)
    print("dtype: ", tensor.dtype)

    print(f"Device tensor is stored on: {tensor.device}")


def p00_basic():
    # Scalar
    scalar = torch.tensor(7)
    print_tensor_info(scalar)

    # Vector
    vector = torch.tensor([7, 7, 8])
    print_tensor_info(vector)

    # Matrix
    MATRIX = torch.tensor([[7, 8], [9, 10]])
    print_tensor_info(MATRIX)

    # Tensor
    TENSOR = torch.tensor([[[1, 2, 3], [3, 6, 9], [2, 4, 5]]])
    print_tensor_info(TENSOR)

    TENSOR2 = torch.tensor(
        [[[1, 2, 3], [3, 6, 9], [2, 4, 5]], [[1, 2, 3], [3, 6, 9], [2, 4, 5]]]
    )
    print_tensor_info(TENSOR2)


def p00_fill_tensor():
    # Create a random tensor of size (3, 4)
    random_tensor = torch.rand(size=(3, 4))
    print_tensor_info(random_tensor)

    # Create a random tensor of size (224, 224, 3)
    random_image_size_tensor = torch.rand(size=(224, 224, 3))
    print_tensor_info(random_image_size_tensor)

    # Create a tensor of all zeros, goto integers
    zeros = torch.zeros(size=(3, 4), dtype=torch.int64)
    print_tensor_info(zeros)

    ones = torch.ones(size=(3, 4), dtype=torch.int64)
    print_tensor_info(ones)

    # Can also create a tensor of zeros similar to another tensor
    ten_zeros = torch.ones(size=(10,))  # must be a tuple
    ten_zeros_again = torch.zeros_like(input=ten_zeros)  # will have same shape
    print_tensor_info(ten_zeros_again)


def p00_basic_ops():
    # Create a tensor of values, add a number to it, multiply
    tensor = torch.tensor([1, 2, 3])
    print(tensor)
    print(tensor + 10)  # Tensor unchanged!
    print(tensor * 10)
    # Can also use torch functions but * and + are more common
    print(torch.multiply(tensor, 10))  # mul != matmul
    print(torch.tensor([1, 2, 3]) * torch.tensor([1, 2, 3]))


def p00_matmul():
    tensor = torch.tensor([2, 4, 8])
    print(tensor * tensor)
    print(torch.matmul(tensor, tensor))  # @ not recommended

    t1 = torch.tensor([[5, 6, 3], [4, 2, 3]])
    t2 = torch.tensor([[5, 6], [4, 2], [4, 5]])
    print(t1.shape, t2.shape)
    MT = torch.matmul(t1, t2)
    print(MT, MT.shape)  
    
def p00_looptest():
    rt1 = torch.rand(size=(10, 10))
    rt2 = torch.rand(size=(10, 10))

    # Matrix multiplication by hand with nested loops
    start = time.perf_counter()
    loop_result = torch.zeros(size=(rt1.shape[0], rt2.shape[1]))
    for i in range(rt1.shape[0]):
        for j in range(rt2.shape[1]):
            for k in range(rt1.shape[1]):
                loop_result[i, j] += rt1[i, k] * rt2[k, j]
    loop_time = time.perf_counter() - start

    # Same with torch.matmul
    start = time.perf_counter()
    matmul_result = torch.matmul(rt1, rt2)
    matmul_time = time.perf_counter() - start

    print("Dims: ", rt1.shape, rt2.shape)
    print("Results equal: ", torch.allclose(loop_result, matmul_result))
    print(f"Loop:   {loop_time * 1e6:.1f} µs")
    print(f"Matmul: {matmul_time * 1e6:.1f} µs")
    print(f"Matmul is {loop_time / matmul_time:.0f}x faster")
    
def p00_agg():
        rt = torch.rand(size=(1000000,))
        print(f"Minimum: {rt.min():.5f}")
        print(f"Maximum: {rt.max():.5f}")
        print(f"Mean: {rt.mean():.5f}")
        print(f"Sum: {rt.sum():.5f}")
        print(f"Index where max value occurs: {rt.argmax()}")
        print(f"Index where min value occurs: {rt.argmin()}")
        
def p00_reshape():
    x = torch.arange(1., 8.)
    print(x, x.shape)
    
    # Add an extra dimension
    x_reshaped = x.reshape(1, 7)
    print(x_reshaped, x_reshaped.shape)
        
    # Stack tensors on top of each other
    x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
    print_tensor_info(x_stacked)         
    
    # Let's index bracket by bracket
    print(f"First square bracket:\n{x_stacked[0]}") 
    print(f"Second square bracket: {x_stacked[0][0]}") 
            
# p00_basic()
# p00_fill_tensor()
# p00_basic_ops()
# p00_matmul()
# p00_looptest()
# p00_agg()
p00_reshape()
