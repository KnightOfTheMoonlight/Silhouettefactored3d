"""
/home/li216/anaconda3/envs/bpy35/bin/python render_script.py  --obj_file data/objs/gt_codes.obj  --hostname vader --r 2 --delta_theta 30
"""

# get the ground truth camera parameters, and then render the 3d model by this file camera view points

# objects in the example room is 209, 67, 109, 625

import os, sys
file_path = os.path.realpath(__file__)
sys.path.insert(0, os.path.dirname(file_path))
sys.path.insert(0, os.path.join(os.path.dirname(file_path), 'bpy'))
import renderer.blender.bpy
import numpy as np
from imp import reload
import renderer.blender.timer
# import matplotlib.pyplot as plt

import renderer.blender.render_utils as ru
import renderer.blender.render_engine as re
import scipy.misc
import scipy.io as sio
import argparse, pprint


def parse_args(str_arg):
  parser = argparse.ArgumentParser(description='render_script_savio')

  parser.add_argument('--hostname', type=str, default='vader')
  parser.add_argument('--out_dir', type=str, default='/hdd/suncg/renderings_sil_32/sil')
  # parser.add_argument('--out_dir', type=str, default='../../cachedir/visualization/blender/')
  # parser.add_argument('--shapenet_dir', type=str, default='/global/scratch/saurabhg/shapenet/')

  parser.add_argument('--obj_file', type=str, default='/data/code/factored3d/benchmark/suncg/../../cachedir/rendering/dwr_shape_ft/1/c_gt_codes.obj')
  parser.add_argument('--layout_file', type=str)
  parser.add_argument('--sz_x', type=int, default=640)
  parser.add_argument('--sz_y', type=int, default=480)

  parser.add_argument('--delta_theta', type=float, default=30.)
  parser.add_argument('--r', type=float, default=2.) # ?
  parser.add_argument('--format', type=str, default='png')


  if len(str_arg) == 0:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args(str_arg)

  pprint.pprint(vars(args))
  return args

# ? tranform to float data?
def deform_fn(vs):
  out = []
  for vs_ in vs:
    _ = vs_*1.
    # _[:,0] = vs_[:,2]
    # _[:,2] = -vs_[:,0]
    out.append(_)
  return out

if __name__ == '__main__':
  args = parse_args(sys.argv[1:])
  print(args)
  output_path = args.out_dir
  # re._prepare(640, 480, use_gpu=False, engine='BLENDER_RENDER', threads=1)
  tmp_file = os.path.join('./sil_results/' + str(os.getpid()) + '.' + args.format)
  exr_files = None;

  write_png_jpg = True
  vis = False
  write_exr = False

  name = os.path.splitext(os.path.basename(args.obj_file))[0]
  print(name)

  # set camera
  # debug
  camera_xyz = np.array([[0, 0, 0]])
  lookat_xyz = np.array([[0, -2, 0]])

  jpg_dir = os.path.join(args.out_dir)#, 'jpg')
  # mask_dir = os.path.join(args.out_dir, 'mask')

  re._prepare(args.sz_x, args.sz_y, use_gpu=False, engine='BLENDER_RENDER',
    threads=1, render_format=args.format)

  #shape_files = [os.path.join(args.obj_file), os.path.join(args.layout_file)]
  shape_files = [os.path.join(args.obj_file)]

  _, masks, _ = re._render(shape_files, re._get_lighting_param_png(), vps=None,
      camera_xyz=camera_xyz, lookat_xyz=lookat_xyz,
      tmp_file=tmp_file, exr_files=exr_files,
      deform_fns=[deform_fn])

  # write files here
  if write_png_jpg:
      re.mkdir_if_missing(os.path.join(jpg_dir))
      # re.mkdir_if_missing(os.path.join(mask_dir))

      for i in range(len(masks)):
          masks[i][masks[i] != 0] = 255
          scipy.misc.imshow(masks[i])
          a = {}
          a["sil_images"] = masks[i]
          a["obj_model_id"] = np.array([1])
          sio.savemat(output_path, mdict=a) # to do change 1 to the real model id

          print('save sil image in .mat file done')
          # output_path = os.path.join(mask_dir, '{:s}_render_{:03d}.jpg'.format(name, i))
          # scipy.misc.imsave(output_path, masks[i])

  # plt can not work at this moment
  # if vis:
  #     fig, axes = plt.subplot(2,5)
  #     axes = axes.ravel()[::-1].tolist()
  #     for j in range(5):
  #         im = ims[j][:,:,:3]
  #         ax = axes.pop()
  #         ax.set_axis_off()
  #         ax.imshow(im.astype(np.uint8))
  #
  #         im = masks[j]
  #         ax = axes.pop()
  #         ax.set_axis_off()
  #         ax.imshow(im.astype(np.uint8))
