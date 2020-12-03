import numpy as np
from matplotlib import pyplot as plt

img_array = np.load('data/full_numpy_bitmap_square.npy')
idx = 0

print(img_array.shape)

print(img_array[idx].shape)

# print(img_array[0])

first = np.reshape(img_array[idx], (-1, 28))
plt.imshow(first, cmap='gray')
plt.show()
