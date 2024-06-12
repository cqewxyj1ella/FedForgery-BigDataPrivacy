import numpy as np
from flwr.common.typing import NDArrayFloat, NDArrayInt


def _stochastic_round(arr: NDArrayFloat) -> NDArrayInt:
    ret: NDArrayInt = np.ceil(arr).astype(np.int32)
    # rand_arr: NDArrayFloat = np.random.rand(*ret.shape)
    # traverse ret and change value -= 1 if rand_arr < arr - ret
    shape = ret.shape
    ret2 = ret.flatten()
    arr2 = arr.flatten()
    # AttributeError: 'float' object has no attribute 'flatten'
    for i in range(len(ret2)):
        rand_num = np.random.rand()
        if rand_num < arr2[i] - ret2[i]:
            ret2[i] -= 1
    ret = ret2.reshape(shape)
    ret = ret.astype(np.int32)
    # np.int32 is not suitable for assignment
    return ret


# Your array
# Your array
arr = np.array(
    [
        [
            [
                [2098884.2, 2102235.0, 2096822.2, 2097799.8],
                [2097728.2, 2097256.8, 2098911.5, 2097513.0],
                [2097376.0, 2097219.2, 2096526.1, 2097810.0],
                [2096310.8, 2095806.9, 2096166.6, 2097359.8],
            ],
            [
                [2102639.5, 2097853.8, 2097098.4, 2097614.8],
                [2094858.5, 2099358.8, 2095797.0, 2096412.2],
                [2095598.9, 2098816.8, 2094789.2, 2098113.8],
                [2096482.2, 2101477.0, 2097801.8, 2097794.5],
            ],
            [
                [2095895.9, 2096452.8, 2096554.6, 2096558.4],
                [2099296.8, 2097045.6, 2095824.1, 2097978.8],
                [2098187.8, 2099481.2, 2096880.6, 2097045.1],
                [2097702.2, 2098180.2, 2097460.2, 2097748.1],
            ],
        ]
    ]
)

# Repeat the array 8 times along a new axis
print(arr.shape)  # Outputs: (1, 3, 4, 4)
repeated_arr = np.repeat(arr, 64, axis=0)
print(repeated_arr.shape)  # Outputs: (64, 3, 4, 4)
result = _stochastic_round(repeated_arr)
print(result)
