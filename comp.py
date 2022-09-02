#!/usr/bin/env python

from skimage.io import imread, imsave
import numpy as np
import sys

images = np.load('data/sampleimages.npz')['images']
d = np.array([images[0]])
data = np.reshape(d, (d.shape[0], d.shape[1]*d.shape[2]))
print(f'd.shape={d.shape}')
print(f'data.shape={data.shape}')
print(f'data.dtype={data.dtype}')

S = 3000

reduced_encoding_matrix = np.load('data/enc%d.npz'%S)['mat']
print(f'reduced_encoding_matrix.shape={reduced_encoding_matrix.shape}')
inv_encoding_matrix = np.load('data/inv%d.npz'%S)['mat']
print(f'inv_encoding_matrix.shape={inv_encoding_matrix.shape}')

#
#

step = d.shape[1]*8
weighting_matrix = np.zeros((1, S))
idx=0
for i in range(0,d.shape[1]*d.shape[2],step):
    m1 = data[:,i:(i+step)]
    m2 = inv_encoding_matrix[i:(i+step),:S]
    #print(f'm2: {np.min(m2)} {np.max(m2)}')
    tmp = np.matmul(m1, m2)
    np.savez('output/S%d-mat1-%d.npz' % (S, idx), mat=m1)
    np.savez('output/S%d-mat2-%d.npz' % (S, idx), mat=m2)
    np.savez('output/S%d-orac-%d.npz' % (S, idx), mat=tmp)
    idx += 1
    #print(f'i={i} {tmp.shape}')
    weighting_matrix = np.add(weighting_matrix, tmp)

weighting_matrix_ref = np.matmul(data, inv_encoding_matrix[:, :S])
print(f'weighting_matrix_ref.shape={weighting_matrix_ref.shape}')

absdiff = np.absolute(weighting_matrix_ref-weighting_matrix)
print(f'diff: min={np.min(absdiff)}  max={np.max(absdiff)}')



#
# recovery
# 
data_approx = np.matmul(weighting_matrix, reduced_encoding_matrix)
data_approx = np.clip(data_approx, 0, np.inf)
mse = np.sum((data - data_approx)**2) / (d.shape[1]*d.shape[2])
print(f'mse={mse}')

data_approx_img = np.reshape(data_approx, (1, d.shape[1], d.shape[2]))
imsave('data_approx%d.png'%S, data_approx_img[0].astype(np.uint16))

datalow = data
invlow = inv_encoding_matrix[:, :S]
weighting_matrix_low = np.matmul(datalow, invlow, dtype=np.float32)
data_approx_low = np.matmul(weighting_matrix_low, reduced_encoding_matrix, dtype=np.float64)
data_approx_low = np.clip(data_approx_low, 0, np.inf)
mselow = np.sum((data - data_approx_low)**2) / (d.shape[1]*d.shape[2])
print(f'mselow={mselow}')


datat = data.transpose().astype(np.float16)
ss = []
for i in range(0,S):
    invi = inv_encoding_matrix[:, i:i+1].astype(np.float32)
    tmpm = np.multiply(invi, datat, dtype=np.float32)
    ss.append(np.sum(tmpm, dtype=np.float32))

weighting_matrix_mix = np.array([ss])
data_approx_mix = np.matmul(weighting_matrix_mix, reduced_encoding_matrix, dtype=np.float64)
data_approx_mix = np.clip(data_approx_mix, 0, np.inf)
msemix = np.sum((data - data_approx_mix)**2) / (d.shape[1]*d.shape[2])
print(f'msemix={msemix}')

sys.exit(0)
