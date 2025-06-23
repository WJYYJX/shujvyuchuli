import numpy as np

#

# a1 = np.load('C:/Users/111/Desktop/numpydata/origin/x_val0.npy')
# a2 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val1.npy')
# a3 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val2.npy')
# a4 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val3.npy')
# a5 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val4.npy')
# a6 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val5.npy')
# a7 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val6.npy')
# a8 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val7.npy')
# a9 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val8.npy')
# a10 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val9.npy')
# a11 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val10.npy')
# a12 = np.load('C:/Users/111/Desktop/zhongzhuan/x_val11.npy')

# a1 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a2 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a3 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a4 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a5 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a6 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a7 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a8 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a9 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a10 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a11 = np.load('F:/weights/train_label.npy',allow_pickle=True)
# a12 = np.load('F:/weights/train_label.npy',allow_pickle=True)

a = np.load('F:/goubangzi_1.2.4.6.10.16/train/label/0.00/0a1a1072b67d4496bbe299a3bbf4e23d.npy',allow_pickle=True)
#a1 = a1.astype(np.float)
b = np.load('F:/jinzhou-liangge/y_data_uint8_choose_test.npy',allow_pickle=True)
#a = a.transpose(0,3,1,2)
a1 = a[:,:,:,0]
a2 = a[:,:,:,1]
a3 = a[:,:,:,2]
a4 = a[:,:,:,3]
a5 = a[:,:,:,4]
a6 = a[:,:,:,5]
# a7 = a[:,:,:,6]
# a8 = a[:,:,:,7]
# a9 = a[:,:,:,8]
# a10 = a[:,:,:,9]
# a11 = a[:,:,:,10]
# a12 = a[:,:,:,11]
a1 = a1[:,:,:,np.newaxis]
a2 = a2[:,:,:,np.newaxis]
a3 = a3[:,:,:,np.newaxis]
a4 = a4[:,:,:,np.newaxis]
a5 = a5[:,:,:,np.newaxis]
a6 = a6[:,:,:,np.newaxis]
# a7 = a7[:,:,:,np.newaxis]
# a8 = a8[:,:,:,np.newaxis]
# a9 = a9[:,:,:,np.newaxis]
# a10 = a10[:,:,:,np.newaxis]
# a11 = a11[:,:,:,np.newaxis]
# a12 = a12[:,:,:,np.newaxis]
b1 = b[:,0]
aa =np.concatenate((a5, a6, a9, a10, a11, a12), axis=3)
#aa = aa.transpose(0,3,1,2)


m=a.shape[0]

for i in range(m):
    m1 = aa[i]
    m2 = b1[i]
    i = str(i)
    np.save('C:/Users/111/Desktop/numpydata/image/x_jinzhou_'+i+'.npy', m1)
    np.save('C:/Users/111/Desktop/numpydata/label/y_jinzhou_'+i+'.npy', m2)

# b1 = np.unique(a2)

#
#
# c = []
#c = np.vstack((a1,a2))#合并npy
# c = np.vstack((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12))
# a1 = a1.astype('float64')
# a2 = a2.astype('float64')
# m1 = np.multiply(a1,100)
# m2 = np.multiply(a2,100)
# m1 = m1.astype('str')
# m2 = m2.astype('str')

# m1 = np.load('C:/Users/111/Desktop/numpydata/x_train_mix.npy',allow_pickle=True)
# m2 = np.load('C:/Users/111/Desktop/numpydata/y_train_mix.npy',allow_pickle=True)
# m3 = np.load('C:/Users/111/Desktop/numpydata/x_val_mix.npy',allow_pickle=True)
# m4 = np.load('C:/Users/111/Desktop/numpydata/y_val_mix.npy',allow_pickle=True)
#
#
#
# print(m1.shape)
# print(m2.shape)
# print(m3.shape)
# print(m4.shape)


