"""
/home/li216/anaconda3/envs/bpy35/bin/python render_mask.py  --obj_dir data/objs --img_dir data/imgs  --out_dir output/img_with_mask --hostname vader --r 2 --delta_theta 30
"""

# get the ground truth camera parameters, and then render the 3d model by this file camera view points

# objects in the example room is 209, 67, 109, 625

import os, sys
file_path = os.path.realpath(__file__)
sys.path.insert(0, os.path.dirname(file_path))
sys.path.insert(0, os.path.join(os.path.dirname(file_path), 'bpy'))
import bpy
import numpy as np
from imp import reload
# import renderer.blender.timer
import timer
# import matplotlib.pyplot as plt

# import renderer.blender.render_utils as ru
# import renderer.blender.render_engine as re
import render_utils as ru
import render_engine as re
import scipy.misc
import scipy.io as sio
import h5py
import argparse, pprint
import shutil
import colorsys
import random
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
# if "DISPLAY" not in os.environ:
#     plt.switch_backend('agg')
import matplotlib.patches as patches
import matplotlib.lines as lines
import tkinter



def parse_args(str_arg):
  parser = argparse.ArgumentParser(description='render_script_savio')

  parser.add_argument('--hostname', type=str, default='vader')
  parser.add_argument('--out_dir', type=str, default='output/img_with_mask')
  parser.add_argument('--img_dir', type=str, default='data/imgs')
  parser.add_argument('--obj_dir', type=str, default='data/objs')
  parser.add_argument('--layout_file', type=str)
  parser.add_argument('--sz_x', type=int, default=499)
  parser.add_argument('--sz_y', type=int, default=376)

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

def apply_mask(image, mask, color, alpha=0.5):
  """Apply the given mask to the image.
  """
  for c in range(3):
    image[:, :, c] = np.where(mask == 1,
                              image[:, :, c] *
                              (1 - alpha) + alpha * color[c] * 255,
                              image[:, :, c])
  return image

def random_colors(N, bright=True):
  """
  Generate random colors.
  To get visually distinct colors, generate them in HSV space then
  convert to RGB.
  """
  brightness = 1.0 if bright else 0.7
  hsv = [(i / N, 1, brightness) for i in range(N)]
  colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
  # random.shuffle(colors)
  return colors

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    print(args)
    output_path = args.out_dir
    # re._prepare(640, 480, use_gpu=False, engine='BLENDER_RENDER', threads=1)
    tmp_file = os.path.join('rendering_test/' + str(os.getpid()) + '.' + args.format)
    exr_files = None;

    write_png_jpg = True
    vis = False
    write_exr = False

    # set camera
    # debug
    camera_xyz = np.array([[0, 0, 0]])
    lookat_xyz = np.array([[0, -2, 0]])

    jpg_dir = os.path.join(args.out_dir)#, 'jpg')

    re._prepare(args.sz_x, args.sz_y, use_gpu=False, engine='BLENDER_RENDER',
    threads=1, render_format=args.format)

    img_num = os.listdir(args.obj_dir)
    for img_num_idx in range(len(img_num)):
        shape_files = [[os.path.join(args.obj_dir, img_num[img_num_idx], x)] for x in sorted(os.listdir(os.path.join(args.obj_dir, img_num[img_num_idx]))) if x.endswith('.obj')]
        img_file = os.path.join(args.img_dir, '{}.png'.format(img_num[img_num_idx]))
        img = scipy.misc.imread(img_file)

        # save for all mask
        # masked_image = img

        N = len(shape_files)
        colors = random_colors(N)

        # save for all mask
        # dpi = 100
        # inch_0 = img.shape[0]/dpi
        # inch_1 = img.shape[1]/dpi
        # fig = plt.figure(figsize=(inch_1, inch_0), dpi=dpi, frameon=False)
        # left, width = 0, 1
        # bottom, height = 0, 1
        # rect = [left, bottom, width, height]
        # ax = plt.axes(rect)

        color_i = 0
        for s in shape_files:
            # save for separate
            dpi = 100
            inch_0 = img.shape[0]/dpi
            inch_1 = img.shape[1]/dpi
            fig = plt.figure(figsize=(inch_1, inch_0), dpi=dpi, frameon=False)
            left, width = 0, 1
            bottom, height = 0, 1
            rect = [left, bottom, width, height]
            ax = plt.axes(rect)

            sep_img = img.copy()

            _, masks, _ = re._render(s, re._get_lighting_param_png(), vps=None,
              camera_xyz=camera_xyz, lookat_xyz=lookat_xyz,
              tmp_file=tmp_file, exr_files=exr_files,
              deform_fns=[deform_fn])
            mask = masks[0]

            mask[mask != 0] = 255
            mask[mask != 0] = 1

            color = colors[color_i]

            # for separate mask
            apply_mask(sep_img, mask, color)

            # separate image
            # masked_image = apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)

            # ----------------------------------------------
            # save for seperate mask
            ax.imshow(sep_img)
            ax.axis('off')
            if not os.path.exists(os.path.join(args.out_dir, img_num[img_num_idx])):
                os.mkdir(os.path.join(args.out_dir, img_num[img_num_idx]))
            plt.savefig(os.path.join(args.out_dir, img_num[img_num_idx], 'img_mask_{}.png'.format(color_i)))
            ax.cla()
            plt.clf()
            plt.close()
            ax = plt.axes(rect)


            color_i = color_i + 1

        # ----------------------------------------------
        # # save for all masks
        # ax.imshow(masked_image)
        # ax.axis('off')
        # plt.tight_layout()
        # if not os.path.exists(os.path.join(args.out_dir, img_num[img_num_idx])):
        #     os.mkdir(os.path.join(args.out_dir, img_num[img_num_idx]))
        # plt.savefig(os.path.join(args.out_dir, img_num[img_num_idx], 'img_mask.png'))
        # plt.close()
