import numpy as np
import os
import torch
from torch.autograd import Variable
from ..utils import transformations
import scipy.io as sio
import glob
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

curr_path = os.path.dirname(os.path.abspath(__file__))
blender_dir = os.path.join(curr_path, 'blender')

colormap = np.array([[0.000000, 0.000000, 0.515625], [0.000000, 0.000000, 0.531250], [0.000000, 0.000000, 0.546875],
                     [0.000000, 0.000000, 0.562500], [0.000000, 0.000000, 0.578125], [0.000000, 0.000000, 0.593750],
                     [0.000000, 0.000000, 0.609375], [0.000000, 0.000000, 0.625000], [0.000000, 0.000000, 0.640625],
                     [0.000000, 0.000000, 0.656250], [0.000000, 0.000000, 0.671875], [0.000000, 0.000000, 0.687500],
                     [0.000000, 0.000000, 0.703125], [0.000000, 0.000000, 0.718750], [0.000000, 0.000000, 0.734375],
                     [0.000000, 0.000000, 0.750000], [0.000000, 0.000000, 0.765625], [0.000000, 0.000000, 0.781250],
                     [0.000000, 0.000000, 0.796875], [0.000000, 0.000000, 0.812500], [0.000000, 0.000000, 0.828125],
                     [0.000000, 0.000000, 0.843750], [0.000000, 0.000000, 0.859375], [0.000000, 0.000000, 0.875000],
                     [0.000000, 0.000000, 0.890625], [0.000000, 0.000000, 0.906250], [0.000000, 0.000000, 0.921875],
                     [0.000000, 0.000000, 0.937500], [0.000000, 0.000000, 0.953125], [0.000000, 0.000000, 0.968750],
                     [0.000000, 0.000000, 0.984375], [0.000000, 0.000000, 1.000000], [0.000000, 0.015625, 1.000000],
                     [0.000000, 0.031250, 1.000000], [0.000000, 0.046875, 1.000000], [0.000000, 0.062500, 1.000000],
                     [0.000000, 0.078125, 1.000000], [0.000000, 0.093750, 1.000000], [0.000000, 0.109375, 1.000000],
                     [0.000000, 0.125000, 1.000000], [0.000000, 0.140625, 1.000000], [0.000000, 0.156250, 1.000000],
                     [0.000000, 0.171875, 1.000000], [0.000000, 0.187500, 1.000000], [0.000000, 0.203125, 1.000000],
                     [0.000000, 0.218750, 1.000000], [0.000000, 0.234375, 1.000000], [0.000000, 0.250000, 1.000000],
                     [0.000000, 0.265625, 1.000000], [0.000000, 0.281250, 1.000000], [0.000000, 0.296875, 1.000000],
                     [0.000000, 0.312500, 1.000000], [0.000000, 0.328125, 1.000000], [0.000000, 0.343750, 1.000000],
                     [0.000000, 0.359375, 1.000000], [0.000000, 0.375000, 1.000000], [0.000000, 0.390625, 1.000000],
                     [0.000000, 0.406250, 1.000000], [0.000000, 0.421875, 1.000000], [0.000000, 0.437500, 1.000000],
                     [0.000000, 0.453125, 1.000000], [0.000000, 0.468750, 1.000000], [0.000000, 0.484375, 1.000000],
                     [0.000000, 0.500000, 1.000000], [0.000000, 0.515625, 1.000000], [0.000000, 0.531250, 1.000000],
                     [0.000000, 0.546875, 1.000000], [0.000000, 0.562500, 1.000000], [0.000000, 0.578125, 1.000000],
                     [0.000000, 0.593750, 1.000000], [0.000000, 0.609375, 1.000000], [0.000000, 0.625000, 1.000000],
                     [0.000000, 0.640625, 1.000000], [0.000000, 0.656250, 1.000000], [0.000000, 0.671875, 1.000000],
                     [0.000000, 0.687500, 1.000000], [0.000000, 0.703125, 1.000000], [0.000000, 0.718750, 1.000000],
                     [0.000000, 0.734375, 1.000000], [0.000000, 0.750000, 1.000000], [0.000000, 0.765625, 1.000000],
                     [0.000000, 0.781250, 1.000000], [0.000000, 0.796875, 1.000000], [0.000000, 0.812500, 1.000000],
                     [0.000000, 0.828125, 1.000000], [0.000000, 0.843750, 1.000000], [0.000000, 0.859375, 1.000000],
                     [0.000000, 0.875000, 1.000000], [0.000000, 0.890625, 1.000000], [0.000000, 0.906250, 1.000000],
                     [0.000000, 0.921875, 1.000000], [0.000000, 0.937500, 1.000000], [0.000000, 0.953125, 1.000000],
                     [0.000000, 0.968750, 1.000000], [0.000000, 0.984375, 1.000000], [0.000000, 1.000000, 1.000000],
                     [0.015625, 1.000000, 0.984375], [0.031250, 1.000000, 0.968750], [0.046875, 1.000000, 0.953125],
                     [0.062500, 1.000000, 0.937500], [0.078125, 1.000000, 0.921875], [0.093750, 1.000000, 0.906250],
                     [0.109375, 1.000000, 0.890625], [0.125000, 1.000000, 0.875000], [0.140625, 1.000000, 0.859375],
                     [0.156250, 1.000000, 0.843750], [0.171875, 1.000000, 0.828125], [0.187500, 1.000000, 0.812500],
                     [0.203125, 1.000000, 0.796875], [0.218750, 1.000000, 0.781250], [0.234375, 1.000000, 0.765625],
                     [0.250000, 1.000000, 0.750000], [0.265625, 1.000000, 0.734375], [0.281250, 1.000000, 0.718750],
                     [0.296875, 1.000000, 0.703125], [0.312500, 1.000000, 0.687500], [0.328125, 1.000000, 0.671875],
                     [0.343750, 1.000000, 0.656250], [0.359375, 1.000000, 0.640625], [0.375000, 1.000000, 0.625000],
                     [0.390625, 1.000000, 0.609375], [0.406250, 1.000000, 0.593750], [0.421875, 1.000000, 0.578125],
                     [0.437500, 1.000000, 0.562500], [0.453125, 1.000000, 0.546875], [0.468750, 1.000000, 0.531250],
                     [0.484375, 1.000000, 0.515625], [0.500000, 1.000000, 0.500000], [0.515625, 1.000000, 0.484375],
                     [0.531250, 1.000000, 0.468750], [0.546875, 1.000000, 0.453125], [0.562500, 1.000000, 0.437500],
                     [0.578125, 1.000000, 0.421875], [0.593750, 1.000000, 0.406250], [0.609375, 1.000000, 0.390625],
                     [0.625000, 1.000000, 0.375000], [0.640625, 1.000000, 0.359375], [0.656250, 1.000000, 0.343750],
                     [0.671875, 1.000000, 0.328125], [0.687500, 1.000000, 0.312500], [0.703125, 1.000000, 0.296875],
                     [0.718750, 1.000000, 0.281250], [0.734375, 1.000000, 0.265625], [0.750000, 1.000000, 0.250000],
                     [0.765625, 1.000000, 0.234375], [0.781250, 1.000000, 0.218750], [0.796875, 1.000000, 0.203125],
                     [0.812500, 1.000000, 0.187500], [0.828125, 1.000000, 0.171875], [0.843750, 1.000000, 0.156250],
                     [0.859375, 1.000000, 0.140625], [0.875000, 1.000000, 0.125000], [0.890625, 1.000000, 0.109375],
                     [0.906250, 1.000000, 0.093750], [0.921875, 1.000000, 0.078125], [0.937500, 1.000000, 0.062500],
                     [0.953125, 1.000000, 0.046875], [0.968750, 1.000000, 0.031250], [0.984375, 1.000000, 0.015625],
                     [1.000000, 1.000000, 0.000000], [1.000000, 0.984375, 0.000000], [1.000000, 0.968750, 0.000000],
                     [1.000000, 0.953125, 0.000000], [1.000000, 0.937500, 0.000000], [1.000000, 0.921875, 0.000000],
                     [1.000000, 0.906250, 0.000000], [1.000000, 0.890625, 0.000000], [1.000000, 0.875000, 0.000000],
                     [1.000000, 0.859375, 0.000000], [1.000000, 0.843750, 0.000000], [1.000000, 0.828125, 0.000000],
                     [1.000000, 0.812500, 0.000000], [1.000000, 0.796875, 0.000000], [1.000000, 0.781250, 0.000000],
                     [1.000000, 0.765625, 0.000000], [1.000000, 0.750000, 0.000000], [1.000000, 0.734375, 0.000000],
                     [1.000000, 0.718750, 0.000000], [1.000000, 0.703125, 0.000000], [1.000000, 0.687500, 0.000000],
                     [1.000000, 0.671875, 0.000000], [1.000000, 0.656250, 0.000000], [1.000000, 0.640625, 0.000000],
                     [1.000000, 0.625000, 0.000000], [1.000000, 0.609375, 0.000000], [1.000000, 0.593750, 0.000000],
                     [1.000000, 0.578125, 0.000000], [1.000000, 0.562500, 0.000000], [1.000000, 0.546875, 0.000000],
                     [1.000000, 0.531250, 0.000000], [1.000000, 0.515625, 0.000000], [1.000000, 0.500000, 0.000000],
                     [1.000000, 0.484375, 0.000000], [1.000000, 0.468750, 0.000000], [1.000000, 0.453125, 0.000000],
                     [1.000000, 0.437500, 0.000000], [1.000000, 0.421875, 0.000000], [1.000000, 0.406250, 0.000000],
                     [1.000000, 0.390625, 0.000000], [1.000000, 0.375000, 0.000000], [1.000000, 0.359375, 0.000000],
                     [1.000000, 0.343750, 0.000000], [1.000000, 0.328125, 0.000000], [1.000000, 0.312500, 0.000000],
                     [1.000000, 0.296875, 0.000000], [1.000000, 0.281250, 0.000000], [1.000000, 0.265625, 0.000000],
                     [1.000000, 0.250000, 0.000000], [1.000000, 0.234375, 0.000000], [1.000000, 0.218750, 0.000000],
                     [1.000000, 0.203125, 0.000000], [1.000000, 0.187500, 0.000000], [1.000000, 0.171875, 0.000000],
                     [1.000000, 0.156250, 0.000000], [1.000000, 0.140625, 0.000000], [1.000000, 0.125000, 0.000000],
                     [1.000000, 0.109375, 0.000000], [1.000000, 0.093750, 0.000000], [1.000000, 0.078125, 0.000000],
                     [1.000000, 0.062500, 0.000000], [1.000000, 0.046875, 0.000000], [1.000000, 0.031250, 0.000000],
                     [1.000000, 0.015625, 0.000000], [1.000000, 0.000000, 0.000000], [0.984375, 0.000000, 0.000000],
                     [0.968750, 0.000000, 0.000000], [0.953125, 0.000000, 0.000000], [0.937500, 0.000000, 0.000000],
                     [0.921875, 0.000000, 0.000000], [0.906250, 0.000000, 0.000000], [0.890625, 0.000000, 0.000000],
                     [0.875000, 0.000000, 0.000000], [0.859375, 0.000000, 0.000000], [0.843750, 0.000000, 0.000000],
                     [0.828125, 0.000000, 0.000000], [0.812500, 0.000000, 0.000000], [0.796875, 0.000000, 0.000000],
                     [0.781250, 0.000000, 0.000000], [0.765625, 0.000000, 0.000000], [0.750000, 0.000000, 0.000000],
                     [0.734375, 0.000000, 0.000000], [0.718750, 0.000000, 0.000000], [0.703125, 0.000000, 0.000000],
                     [0.687500, 0.000000, 0.000000], [0.671875, 0.000000, 0.000000], [0.656250, 0.000000, 0.000000],
                     [0.640625, 0.000000, 0.000000], [0.625000, 0.000000, 0.000000], [0.609375, 0.000000, 0.000000],
                     [0.593750, 0.000000, 0.000000], [0.578125, 0.000000, 0.000000], [0.562500, 0.000000, 0.000000],
                     [0.546875, 0.000000, 0.000000], [0.531250, 0.000000, 0.000000], [0.515625, 0.000000, 0.000000],
                     [0.500000, 0.000000, 0.000000]])

# lin:
# for each point cloud
# cube_v is a point coordinates container
# cube_v.shape: 8*3
# size 1*1*1
# [-0.5, 0.5] * [-0.5, 0.5] * [-0.5, 0.5]
cube_v = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0],
                   [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
cube_v = cube_v - 0.5

# lin:
# cube_f.shape: 12*3
# cube_f is the vertex index for each triangle face of the cube
# each facet of cube is represented as triangle instead of rectangular. so there are 2*6 triangle for use.
cube_f = np.array(
    [[1, 7, 5], [1, 3, 7], [1, 4, 3], [1, 2, 4], [3, 8, 7], [3, 4, 8], [5, 7, 8], [5, 8, 6], [1, 5, 6], [1, 6, 2],
     [2, 6, 8], [2, 8, 4]]).astype(np.int)


# 0.5 is the occupancy threshhold
def voxels_to_mesh(pred_vol, thresh=0.5):
    v_counter = 0
    tot_points = np.greater(pred_vol, thresh).sum()
    # v_all size is 8*3*tot_points
    # each voxel is represented by a cube, and this voxel is at the center of the cube
    # and each cube is represented by 8 vertices and 12 triangle faces
    # for single object volume estimation result, creat the correct vertices and faces settings (v_all, f_all).
    v_all = np.tile(cube_v, [tot_points, 1])
    # f_all size is 12*3*tot_points
    f_all = np.tile(cube_f, [tot_points, 1])
    f_offset = np.tile(np.linspace(0, 12 * tot_points - 1, 12 * tot_points), 3).reshape(3, 12 * tot_points).transpose()
    f_offset = (f_offset // 12 * 8).astype(np.int)
    f_all += f_offset

    for x in range(pred_vol.shape[0]):
        for y in range(pred_vol.shape[1]):
            for z in range(pred_vol.shape[2]):
                if pred_vol[x, y, z] > thresh:
                    radius = pred_vol[x, y, z]
                    v_all[v_counter:v_counter + 8, :] *= radius
                    v_all[v_counter:v_counter + 8, :] += (np.array(
                        [[x, y, z]]) + 0.5)  # [x,y,z] is the coordinate of current point in voxel space
                    v_counter += 8
    return v_all, f_all


def voxels_to_points(pred_vol, thresh=0.5):
    v_counter = 0
    tot_points = np.greater(pred_vol, thresh).sum()
    v_all = np.zeros([tot_points, 3])
    for x in range(pred_vol.shape[0]):
        for y in range(pred_vol.shape[1]):
            for z in range(pred_vol.shape[2]):
                if pred_vol[x, y, z] > thresh:
                    v_all[v_counter, :] = (np.array([[x, y, z]]) + 0.5)
                    v_counter += 1
    return v_all


def append_obj(mf_handle, vertices, faces):
    for vx in range(vertices.shape[0]):
        mf_handle.write('v {:f} {:f} {:f}\n'.format(vertices[vx, 0], vertices[vx, 1], vertices[vx, 2]))
    for fx in range(faces.shape[0]):
        mf_handle.write('f {:d} {:d} {:d}\n'.format(faces[fx, 0], faces[fx, 1], faces[fx, 2]))
    return


def append_mtl_obj(mf_handle, vertices, faces, mtl_ids):
    for vx in range(vertices.shape[0]):
        mf_handle.write('v {:f} {:f} {:f}\n'.format(vertices[vx, 0], vertices[vx, 1], vertices[vx, 2]))
    for fx in range(faces.shape[0]):
        mf_handle.write('usemtl m{}\n'.format(mtl_ids[fx]))
        mf_handle.write('f {:d} {:d} {:d}\n'.format(faces[fx, 0], faces[fx, 1], faces[fx, 2]))
    return


def append_mtl(mtl_handle, mtl_ids, colors):
    for mx in range(len(mtl_ids)):
        mtl_handle.write('newmtl m{}\n'.format(mtl_ids[mx]))
        mtl_handle.write('Kd {:f} {:f} {:f}\n'.format(colors[mx, 0], colors[mx, 1], colors[mx, 2]))
        mtl_handle.write('Ka 0 0 0\n')
    return


def render_mesh(mesh_file, png_dir, scale=1):
    cmd = '/home/li216/anaconda3/envs/bpy35/bin/python {:s}/render_script.py  --obj_file {:s} --out_dir {:s} --r 2 --delta_theta 30 --sz_x {} --sz_y {}'.format(
        blender_dir, mesh_file, png_dir, int(640 * scale), int(480 * scale))
    print('cmd is {}'.format(cmd))
    os.system(cmd)
    return


def render_directory(mesh_dir, png_dir, scale=1, prefix=None):
    cmd = '/home/li216/anaconda3/envs/bpy35/bin/python {:s}/render_dir_script.py --prefix {:s} --obj_dir {:s} --out_dir {:s} --r 2 --delta_theta 30 --sz_x {} --sz_y {}'.format(
        blender_dir, prefix, mesh_dir, png_dir, int(640 * scale), int(480 * scale))
    print('cmd is {}'.format(cmd))
    os.system(cmd)
    return


def render_sil(mesh_file, mat_dir, scale=1):
    cmd = '/home/li216/anaconda3/envs/bpy35/bin/python {:s}/render_sil.py  --obj_file {:s} --out_dir {:s} --r 2 --delta_theta 30 --sz_x {} --sz_y {}'.format(
        blender_dir, mesh_file, mat_dir, int(640 * scale), int(480 * scale))
    print('cmd is {}'.format(cmd))
    os.system(cmd)
    return


def render_dir_sil(mesh_dir, hdf5_dir, scale=0.5, prefix=None):
    cmd = '/home/li216/anaconda3/envs/bpy35/bin/python {:s}/render_dir_sil.py --obj_dir {:s} --out_file {:s}  --r 2 --delta_theta 30 --sz_x {} --sz_y {}'.format(
        blender_dir, mesh_dir, hdf5_dir, int(640 * scale), int(480 * scale))
    print('cmd is {}'.format(cmd))
    os.system(cmd)
    return


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# lin: with flexibility for r and theta
def render_mesh(mesh_file, png_dir, r=2., theta=30., scale=1):
    cmd = '/home/li216/anaconda3/envs/bpy35/bin/python {:s}/render_script.py  --obj_file {:s} --out_dir {:s} --r {} --delta_theta {} --sz_x {} --sz_y {}'.format(
        blender_dir, mesh_file, png_dir, float(r), float(theta), int(640 * scale), int(480 * scale))
    print('cmd is {}'.format(cmd))
    os.system(cmd)
    return


def render_directory(mesh_dir, png_dir, r=2., theta=30., scale=1, prefix=None):
    cmd = '/home/li216/anaconda3/envs/bpy35/bin/python {:s}/render_dir_script.py --prefix {:s} --obj_dir {:s} --out_dir {:s} --r {} --delta_theta {} --sz_x {} --sz_y {}'.format(
        blender_dir, prefix, mesh_dir, png_dir, float(r), float(theta), int(640 * scale), int(480 * scale))
    print('cmd is {}'.format(cmd))
    os.system(cmd)
    return


def render_sil(mesh_file, mat_dir, r=2., theta=30., scale=1):
    cmd = '/home/li216/anaconda3/envs/bpy35/bin/python {:s}/render_sil.py  --obj_file {:s} --out_dir {:s} --r {} --delta_theta {} --sz_x {} --sz_y {}'.format(
        blender_dir, mesh_file, mat_dir, float(r), float(theta), int(640 * scale), int(480 * scale))
    print('cmd is {}'.format(cmd))
    os.system(cmd)
    return


def render_dir_sil(mesh_dir, hdf5_dir, r=2., theta=30., scale=0.5, prefix=None):
    cmd = '/home/li216/anaconda3/envs/bpy35/bin/python {:s}/render_dir_sil.py --obj_dir {:s} --out_file {:s}  --r {} --delta_theta {} --sz_x {} --sz_y {}'.format(
        blender_dir, mesh_dir, hdf5_dir, float(r), float(theta), int(640 * scale), int(480 * scale))
    print('cmd is {}'.format(cmd))
    os.system(cmd)
    return


# ---------------------------------------------------------------------------------------------------------------------------------------------------


# downsample by convolution
class Downsample(torch.nn.Module):
    def __init__(self, s, use_max=False, batch_mode=False):
        super(Downsample, self).__init__()
        self.batch_mode = batch_mode
        if (use_max):
            layer = torch.nn.MaxPool3d(s, stride=s)
        else:
            layer = torch.nn.Conv3d(1, 1, s, stride=s)
            layer.weight.data.fill_(1. / layer.weight.data.nelement())
            layer.bias.data.fill_(0)
        self.layer = layer

    def forward(self, vol):
        if self.batch_mode:
            out_vol = self.layer.forward(vol)
        else:
            out_vol = self.layer.forward(torch.unsqueeze(torch.unsqueeze(vol, 0), 0))[0, 0]
        return out_vol


def downsample(vol, s, use_max=False):
    module = Downsample(s, use_max=use_max).type(torch.FloatTensor)
    vol_var = Variable(torch.from_numpy(vol), requires_grad=False).type(torch.FloatTensor)
    return module.forward(vol_var).data.numpy()


# get predicted volume and transform matrix
def prediction_to_entity(pred):
    if torch.is_tensor(pred[0]):
        pred = [p.numpy() for p in pred]
    volume = pred[0].astype(np.float)
    transform = pred[1].astype(np.float)
    if transform.shape[0] == 4 and transform.shape[1] == 4:
        return volume, transform
    else:
        scale_mat = np.diag(pred[1].astype(np.float))  # scale
        rot_mat = transformations.quaternion_matrix(pred[2].astype(np.float))[0:3, 0:3]  # rotation
        transform = np.eye(4)
        transform[0:3, 0:3] = np.matmul(rot_mat, scale_mat)
        transform[0:3, 3] = pred[3].astype(np.float)
        return volume, transform

# get predicted volume and transform matrix
def prediction_to_entity_nyu(pred):
    if torch.is_tensor(pred['shape']):
        pred = [p.numpy() for p in pred]
    volume = pred['shape'].astype(np.float)
    transform = pred['trans'].astype(np.float)
    if transform.shape[0] == 4 and transform.shape[1] == 4:
        return volume, transform
    else:
        scale_mat = np.diag(pred['scale'].astype(np.float))  # scale
        if pred['quat'].shape != (4,):
            rot_mat = transformations.quaternion_matrix(pred['quat'][0,:].astype(np.float))[0:3, 0:3]  # rotation nyu take symmetry into accound
        else:
            rot_mat = transformations.quaternion_matrix(pred['quat'].astype(np.float))[0:3, 0:3]  # rotation nyu take symmetry into accound
        transform = np.eye(4)
        transform[0:3, 0:3] = np.matmul(rot_mat, scale_mat)
        transform[0:3, 3] = pred['trans'].astype(np.float)
        return volume, transform

#   save mesh and voxel data; .obj with .mat files
def save_parse(mesh_file, codes, thresh=0.5, use_soft_voxels=True, save_objectwise=False):
    [os.remove(i) for i in glob.glob(os.path.join(mesh_file, '*.obj'))]
    mtl_file = mesh_file.replace('.obj', '.mtl')
    fout_mtl = open(mtl_file, 'w')
    mtl_file = mtl_file.split('/')[-1]

    n_parts = len(codes)
    color_inds = np.linspace(0, 255, n_parts).astype(np.int).tolist()
    for p in range(n_parts):
        cmap = colormap[color_inds[p]]
        fout_mtl.write(
            'newmtl m{:d}\nKd {:f} {:f} {:f}\nKa 0 0 0\n'.format(p, cmap[0], cmap[1],
                                                                 cmap[2]))  # kd: diffuse color; ka: ambient color
    fout_mtl.close()
    if not save_objectwise:
        fout = open(mesh_file, 'w')
        fout.write('mtllib {:s}\n'.format(mtl_file))

    f_counter = 0
    # lin:
    vox_dict = {}
    vox_file = mesh_file.replace('.obj', '.mat')

    for p in range(n_parts):
        volume, transform = prediction_to_entity(codes[p])

        # lin note: each object have one pair of volume and transform data
        vox_dict['volume_' + str(p)] = volume
        vox_dict['transform_' + str(p)] = transform

        if save_objectwise:
            fout = open(mesh_file.replace('.obj', '_' + str(p) + '.obj'), 'w')
            fout.write('mtllib {:s}\n'.format(mtl_file))
        volume = downsample(volume, volume.shape[0] // 32)  # lin: if volume size is not 32*32*32, downsample it
        v, f = voxels_to_mesh(volume, thresh=thresh)
        v = v / 32 - 0.5  # lin note: why v is normed, but f is not: cause f only contains vertices indexes

        if v.size > 0:
            n_verts = v.shape[0]
            v_homographic = np.concatenate((v, np.ones((n_verts, 1))), axis=1).transpose()
            v_transformed = np.matmul(transform[0:3, :], v_homographic).transpose()

            # # lin: debug
            # scene_vol = points_to_scene_voxels(v_transformed)  # W X H X D
            # w,h,d = scene_vol.shape
            # scene_vol_vis = np.transpose(scene_vol, (0, 2, 1))
            # # visualize_vol
            # plt.close('all')
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # ax.set(title='point_cloud_cam', xlabel='w', ylabel='d', zlabel='h')
            # ax.set_zlim((h,0))
            # ax.voxels(scene_vol_vis, facecolors='red', edgecolor='k')
            # for angle in range(0, 360, 30):
            #     # ax1.view_init(30, angle)
            #     # ax2.view_init(30, angle)
            #     # ax3.view_init(30, angle)
            #     ax.view_init(30, angle)
            #     plt.draw()
            #     plt.pause(.5)
            # plt.show()
            # # plt.pause(.1)
            # plt.close()

            fout.write('usemtl m{:d}\n'.format(p))
            append_obj(fout, v_transformed, f + f_counter)

            if not save_objectwise:
                f_counter += n_verts

        if save_objectwise or p == (n_parts - 1):
            fout.close()
    #  lin original volume grid prediction and transform matrix
    sio.savemat(vox_file, vox_dict)

#   save mesh and voxel data; .obj with .mat files
def save_parse_nyu(mesh_file, codes, thresh=0.5, use_soft_voxels=True, save_objectwise=False):
    [os.remove(i) for i in glob.glob(os.path.join(mesh_file, '*.obj'))]
    mtl_file = mesh_file.replace('.obj', '.mtl')
    fout_mtl = open(mtl_file, 'w')
    mtl_file = mtl_file.split('/')[-1]

    # codes_keys = codes.keys()
    n_parts = len(codes)
    color_inds = np.linspace(0, 255, n_parts).astype(np.int).tolist()
    for p in range(n_parts):
        cmap = colormap[color_inds[p]]
        fout_mtl.write(
            'newmtl m{:d}\nKd {:f} {:f} {:f}\nKa 0 0 0\n'.format(p, cmap[0], cmap[1],
                                                                 cmap[2]))  # kd: diffuse color; ka: ambient color
    fout_mtl.close()
    if not save_objectwise:
        fout = open(mesh_file, 'w')
        fout.write('mtllib {:s}\n'.format(mtl_file))

    f_counter = 0
    # lin:
    vox_dict = {}
    vox_file = mesh_file.replace('.obj', '.mat')

    for p in range(n_parts):
        volume, transform = prediction_to_entity_nyu(codes[p])

        # lin note: each object have one pair of volume and transform data
        vox_dict['volume_' + str(p)] = volume
        vox_dict['transform_' + str(p)] = transform

        if save_objectwise:
            fout = open(mesh_file.replace('.obj', '_' + str(p) + '.obj'), 'w')
            fout.write('mtllib {:s}\n'.format(mtl_file))
        volume = downsample(volume, volume.shape[0] // 32)  # lin: if volume size is not 32*32*32, downsample it
        v, f = voxels_to_mesh(volume, thresh=thresh)
        v = v / 32 - 0.5  # lin note: why v is normed, but f is not: cause f only contains vertices indexes

        if v.size > 0:
            n_verts = v.shape[0]
            v_homographic = np.concatenate((v, np.ones((n_verts, 1))), axis=1).transpose()
            v_transformed = np.matmul(transform[0:3, :], v_homographic).transpose()

            # # lin: debug
            # scene_vol = points_to_scene_voxels(v_transformed)  # W X H X D
            # w,h,d = scene_vol.shape
            # scene_vol_vis = np.transpose(scene_vol, (0, 2, 1))
            # # visualize_vol
            # plt.close('all')
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # ax.set(title='point_cloud_cam', xlabel='w', ylabel='d', zlabel='h')
            # ax.set_zlim((h,0))
            # ax.voxels(scene_vol_vis, facecolors='red', edgecolor='k')
            # for angle in range(0, 360, 30):
            #     # ax1.view_init(30, angle)
            #     # ax2.view_init(30, angle)
            #     # ax3.view_init(30, angle)
            #     ax.view_init(30, angle)
            #     plt.draw()
            #     plt.pause(.5)
            # plt.show()
            # # plt.pause(.1)
            # plt.close()

            fout.write('usemtl m{:d}\n'.format(p))
            append_obj(fout, v_transformed, f + f_counter)

            if not save_objectwise:
                f_counter += n_verts

        if save_objectwise or p == (n_parts - 1):
            fout.close()
    #  lin original volume grid prediction and transform matrix
    sio.savemat(vox_file, vox_dict)


def codes_to_points(codes, thresh=0.5, objectwise=False):
    scene_verts = []
    n_parts = len(codes)
    for p in range(n_parts):
        volume, transform = prediction_to_entity(codes[p])
        volume = downsample(volume, volume.shape[0] // 32)
        v = voxels_to_points(volume, thresh=thresh)
        v = v / 32 - 0.5

        if v.size > 0:
            n_verts = v.shape[0]
            v_homographic = np.concatenate((v, np.ones((n_verts, 1))), axis=1).transpose()
            v_transformed = np.matmul(transform[0:3, :], v_homographic).transpose()
            scene_verts.append(v_transformed)

    if not objectwise:
        scene_verts = np.concatenate(scene_verts, axis=0)

    return scene_verts


def dispmap_to_mesh(dmap, k_mat, scale_x=1, scale_y=1, min_disp=1e-2):
    '''
    Converts a inverse depth map to a 3D point cloud.

    Args:
        dmap: H X W inverse depth map
        k_mat : 3 X 3 intrinsic matrix
        scale_x: Scale the intrinsic matrix's x row by this factor e.g. scale=0.5 implies downsampling by factor of 2
        scale_y: Scale the intrinsic matrix's y row by this factor e.g. scale=0.5 implies downsampling by factor of 2
        min_disp: Points with disp less than this are not rendered
    Returns:
        vs: n_pts X 3 [x,y,z] coordinates
        fs: mesh faces
    '''
    H = np.shape(dmap)[0]
    W = np.shape(dmap)[1]
    dmap = dmap.reshape((H, W))
    k_mat[0, :] = scale_x * k_mat[0, :]
    k_mat[1, :] = scale_y * k_mat[1, :]
    k_inv = np.linalg.inv(k_mat)
    num_pts = H * W
    pts = np.ones((3, num_pts))
    ctr = 0
    for y in range(H):
        for x in range(W):
            pts[0, ctr] = x + 0.5
            pts[1, ctr] = y + 0.5
            pts[:, ctr] *= (1 / dmap[y, x])
            ctr += 1

    verts = np.transpose(np.matmul(k_inv, pts))
    num_faces_max = H * W * 2
    faces = np.zeros((num_faces_max, 3))
    face_ctr = 0
    for y in range(H - 1):
        for x in range(W - 1):
            if (dmap[y, x] > min_disp) and (dmap[y, x + 1] > min_disp) and (dmap[y + 1, x + 1] > min_disp):
                faces[face_ctr, 0] = y * W + x + 1
                faces[face_ctr, 1] = y * W + (x + 1) + 1
                faces[face_ctr, 2] = (y + 1) * W + (x + 1) + 1
                face_ctr += 1

            if (dmap[y, x] > min_disp) and (dmap[y + 1, x] > min_disp) and (dmap[y + 1, x + 1] > min_disp):
                faces[face_ctr, 0] = y * W + x + 1
                faces[face_ctr, 1] = (y + 1) * W + x + 1
                faces[face_ctr, 2] = (y + 1) * W + (x + 1) + 1
                face_ctr += 1
    faces = faces[0:face_ctr, :]
    return verts, faces.astype(np.int)


def dispmap_to_points(dmap, k_mat, scale_x=1, scale_y=1, min_disp=1e-2):
    '''
    Converts a inverse depth map to a 3D point cloud.

    Args:
        dmap: H X W inverse depth map
        k_mat : 3 X 3 intrinsic matrix
        scale_x: Scale the intrinsic matrix's x row by this factor e.g. scale=0.5 implies downsampling by factor of 2
        scale_y: Scale the intrinsic matrix's y row by this factor e.g. scale=0.5 implies downsampling by factor of 2
        min_disp: Points with disp less than this are not rendered
    Returns:
        n_pts X 3 [x,y,z] coordinates
    '''
    H = np.shape(dmap)[0]
    W = np.shape(dmap)[1]
    dmap = dmap.reshape((H, W))
    k_mat[0, :] = scale_x * k_mat[0, :]
    k_mat[1, :] = scale_y * k_mat[1, :]
    k_inv = np.linalg.inv(k_mat)
    num_pts = np.sum(np.greater(dmap, min_disp))
    pts = np.ones((3, num_pts))
    ctr = 0
    for y in range(H):
        for x in range(W):
            if (dmap[y, x] > min_disp) and (ctr < num_pts):
                pts[0, ctr] = x + 0.5
                pts[1, ctr] = y + 0.5
                pts[:, ctr] *= (1 / dmap[y, x])
                ctr += 1
    return np.transpose(np.matmul(k_inv, pts))


def points_to_cubes(points, edge_size=0.05):
    '''
    Converts an input point cloud to a set of cubes.

    Args:
        points: N X 3 array
        edge_size: cube edge size
    Returns:
        vs: vertices
        fs: faces
    '''
    v_counter = 0
    tot_points = points.shape[0]
    v_all = np.tile(cube_v, [tot_points, 1])
    f_all = np.tile(cube_f, [tot_points, 1])
    f_offset = np.tile(np.linspace(0, 12 * tot_points - 1, 12 * tot_points), 3).reshape(3, 12 * tot_points).transpose()
    f_offset = (f_offset // 12 * 8).astype(np.int)
    f_all += f_offset
    for px in range(points.shape[0]):
        v_all[v_counter:v_counter + 8, :] *= edge_size
        v_all[v_counter:v_counter + 8, :] += points[px, :]
        v_counter += 8

    return v_all, f_all


# ref: scene_comparison.py
def points_to_scene_voxels(points, voxels_width=64, voxels_height=32, voxels_depth=64):
    """Returns scene_voxels given points.

    Args:
        points: n X 3 numpy array
    Returns:
        scene_voxels: W X H X D occupancies
    """
    scene_voxels = np.zeros((voxels_width, voxels_height, voxels_depth))
    vs = np.copy(points)
    vs /= 0.04 * (64 // voxels_height)
    vs[:, 0] += scene_voxels.shape[0] / 2.0 # W
    vs[:, 1] += scene_voxels.shape[1] / 2.0 # H
    vs = np.floor(vs).astype(np.int32)
    valid_inds = (vs[:, 0] >= 0) & (vs[:, 1] >= 0) & (vs[:, 2] >= 0) & (vs[:, 0] < voxels_width) & (
            vs[:, 1] < voxels_height) & (vs[:, 2] < voxels_depth)
    vs = vs[valid_inds, :]
    for v in range(vs.shape[0]):
        x = vs[v, 0]
        y = vs[v, 1]
        z = vs[v, 2]
        if scene_voxels[x, y, z] == 0:
            scene_voxels[x, y, z] = 1
    return scene_voxels
