"""
/home/li216/anaconda3/envs/bpy35/bin/python render_script_savio.py  --obj_dir meshes --hostname vader --r 2 --delta_theta 30
"""
import os, sys
file_path = os.path.realpath(__file__)
sys.path.insert(0, os.path.dirname(file_path))
sys.path.insert(0, os.path.join(os.path.dirname(file_path), 'bpy'))
import bpy
import numpy as np
from imp import reload
import timer
# import matplotlib.pyplot as plt

import render_utils as ru
import render_engine as re
import scipy.misc
import argparse, pprint


def parse_args(str_arg):
  parser = argparse.ArgumentParser(description='render_script_savio')

  parser.add_argument('--out_dir', type=str, default='../../cachedir/visualization/blender/')
  # parser.add_argument('--shapenet_dir', type=str, default='/global/scratch/saurabhg/shapenet/')
  parser.add_argument('--prefix', type=str, default=None)

  parser.add_argument('--obj_dir', type=str)
  parser.add_argument('--sz_x', type=int, default=320)
  parser.add_argument('--sz_y', type=int, default=240)

  parser.add_argument('--delta_theta', type=float, default=30.)
  parser.add_argument('--r', type=float, default=2.)
  parser.add_argument('--format', type=str, default='png')


  if len(str_arg) == 0:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args(str_arg)

  pprint.pprint(vars(args))
  return args

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

  # re._prepare(640, 480, use_gpu=False, engine='BLENDER_RENDER', threads=1)
  # tmp_file = os.path.join('/data', 'code/factored3d/cachedir/rendering', 'bpy-' + str(os.getpid()) + '.' + args.format)
  tmp_file = os.path.join(args.out_dir, 'tmpfile-' + str(os.getpid()) + '.' + args.format)
  exr_files = None;

  camera_xyz = np.zeros((5,3))
  lookat_xyz = np.zeros((5,3))
  r = args.r
  # rng = np.random.RandomState(0)
  # n_cams = 5
  # camera_xyz = np.zeros((n_cams,3))
  # lookat_xyz = np.zeros((n_cams,3))
  i = 0
  for l in [0, 2]:
    for t in [-args.delta_theta, args.delta_theta]:
      i = i+1
      t = np.deg2rad(t)
      camera_xyz[i,l] = r*np.sin(t)
      camera_xyz[i,1] = r*np.cos(t) - r
  lookat_xyz[:,1] = -r
      # camera_xyz[i,l] = 2*r*np.sin(t)
      # camera_xyz[i,1] = 2*r*np.cos(t) - 2*r
  # lookat_xyz[:,1] = -1.5*r

  jpg_dir = os.path.join(args.out_dir)#, 'jpg')
  #
  re._prepare(args.sz_x, args.sz_y, use_gpu=False, engine='BLENDER_RENDER',
    threads=1, render_format=args.format) # blender 2.79
  # re._prepare(args.sz_x, args.sz_y, use_gpu=False, engine='xxx',
  #   threads=1, render_format=args.format) # tried for blender  2.80

  if(args.obj_dir.endswith('.obj')):
    shape_files = [os.path.join(args.obj_dir)]
  else:
    shape_files = [os.path.join(args.obj_dir, x) for x in os.listdir(args.obj_dir) if x.endswith('.obj')]

  # generate different camera view results,
  # ims: num_camera_view*img_h*img_w*c
  # masks: num_camera_view*img_h*img_w
  ims, masks, _ = re._render(shape_files, re._get_lighting_param_png(), vps=None,
      camera_xyz=camera_xyz, lookat_xyz=lookat_xyz,
      tmp_file=tmp_file, exr_files=exr_files,
      deform_fns=[deform_fn])


  write_png_jpg = True
  # write files here
  if write_png_jpg:
      re.mkdir_if_missing(os.path.join(jpg_dir))
      for i in range(len(ims)):
          im_ = np.concatenate((ims[i], masks[i][:,:,np.newaxis].astype(np.uint8)), axis=2)
          output_path = os.path.join(jpg_dir, '{}_render_{:03d}.png'.format(args.prefix, i))
          scipy.misc.imsave(output_path, im_)
