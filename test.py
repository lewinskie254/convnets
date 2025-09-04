import numpy as np

arr = np.array([[1, 2],
                [3, 4]])

print(arr)
# [[1 2]
#  [3 4]]

# Pad with 1 zero on all sides
padded = np.pad(arr, pad_width=1, mode='constant', constant_values=0)

print(padded)
