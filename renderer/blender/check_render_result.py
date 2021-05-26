# check silhouette hdf5 content
import h5py
import glob
import os
import matplotlib.pyplot as mlt

for housename in glob.glob('/hdd/suncg/renderings_sil/*'):
    for file_name in glob.glob(os.path.join(housename, '*.hdf5')):
        print(file_name)
        # with open(file_name)
        f = h5py.File(file_name, 'r')
        # List all groups
        # print file_name
        print("Keys: %s" % f.keys())
        # for i in f.keys()d:
        #     print('key {}, data {}'.format(i,f[i][...].shape))
        # for j in range(f['sil_images'][...].shape[0]):
        #     mlt.imshow(f['sil_images'][...][j, :, :])
        if 'sil_images' not in f.keys():
            os.remove(file_name)
            # print(file_name)
print 'done'

# file_name = '/hdd/suncg/renderings_sil/857e73060214a25ca8a2871c4b843db7/000020_sil.hdf5'
# f = h5py.File(file_name, 'r')
# print 'done'
