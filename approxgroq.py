#!/usr/bin/env python

from skimage.io import imread, imsave
import numpy as np
import sys

images = np.load('testdata.npz')['images']
d = np.array([images[0]])
data = np.reshape(d, (d.shape[0], d.shape[1]*d.shape[2]))
print(f'd.shape={d.shape}')
print(f'data.shape={data.shape}')
print(f'data.dtype={data.dtype}')
S = 3000

reduced_encoding_matrix = np.load('enc%d.npz'%S)['mat']
print(f'reduced_encoding_matrix.shape={reduced_encoding_matrix.shape}')
inv_encoding_matrix = np.load('inv%d.npz'%S)['mat']
print(f'inv_encoding_matrix.shape={inv_encoding_matrix.shape}')

#
#

step = d.shape[1]*8
weighting_matrix = np.zeros((1, S)).astype(np.float64)
idx=0
for i in range(0,d.shape[1]*d.shape[2],step):
    tmp = np.load('S%d-res-%d.npz'%(S,idx))['mat']
    idx += 1
    #print(f'i={i} {tmp.shape}')
    weighting_matrix = np.add(weighting_matrix, tmp)
print(f'weighting_matrix.dtype={weighting_matrix.dtype}')
weighting_matrix_ref = np.matmul(data, inv_encoding_matrix[:, :S])
print(f'weighting_matrix_ref.shape={weighting_matrix_ref.shape}')

absdiff = np.absolute(weighting_matrix_ref-weighting_matrix)
print(f'diff: min={np.min(absdiff)}  max={np.max(absdiff)}')

print(weighting_matrix_ref)
print(weighting_matrix)

#
# recovery
# 
data_approx = np.matmul(weighting_matrix, reduced_encoding_matrix)
data_approx = np.clip(data_approx, 0, np.inf)
mse = np.sum((data - data_approx)**2) / (d.shape[1]*d.shape[2])
print(f'mse={mse}')

data_approx_img = np.reshape(data_approx, (1, d.shape[1], d.shape[2]))
imsave('groq_data_approx%d.png'%S, data_approx_img[0].astype(np.uint16))

sys.exit(0)
