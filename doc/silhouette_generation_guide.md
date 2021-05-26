
There are four methods to get rendered 2D images. refer to [render/blender/render_script.py](render/blender/render_script.py), [render/blender/render_dir_script.py](render/blender/render_dir_script.py), [render/blender/render_engine.py](render/blender/render_engine.py), [render/blender/render_utils.py](render/blender/render_utils.py) 
All based on blender engine.

---

1. #### [render/blender/render_script.py](render/blender/render_script.py)

   ##### generate rendered images from one obj mesh model

example command:
```
render_script.py  --obj_file meshes/code_gt_6.obj  --hostname vader --r 2 --delta_theta 30

or 
1. use conda environment activated
python3 /data/code/factored3d/renderer/blender/render_script.py  --obj_file /data/code/factored3d/benchmark/suncg/../../cachedir/rendering/dwr_shape_ft/layout.obj --out_dir /data/code/factored3d/benchmark/suncg/../../cachedir/rendering/dwr_shape_ft/rendering --r 2 --delta_theta 30 --sz_x 320 --sz_y 240


2. use bpy35 builtin conda python
/home/li216/anaconda3/envs/bpy35/bin/python render_script.py  --obj_file ./data/objs/0/gt_codes_0.obj  --out_dir ./rendering_test --r 2 --delta_theta 30 --sz_x 320 --sz_y 240
```
cotent:
the most out wrap for rendering 


for generating  rendered images with the same order as in the `.obj` folder

**procedure**
```
1. setting camera parameters, image size etc.
2. ims, masks, _ = re._render(params)
3. save imgs, camera_xyz for generate ground truth mesh .obj data  
```

**some code details**

ground truth camera_xyz and lookout_xyz
```
| --- camera_xyz --- | ---- | --- lookout_xyz -- |
| 0    | 0    | 0    |      | 0    | -2   | 0    |
```


**my modified version:**

`render/blender/sil_gen_lin.py` 

example:
```
sil_gen_lin.py
--obj_file /data/code/factored3d/benchmark/suncg/../../cachedir/rendering/dwr_shape_ft/1/c_gt_codes.obj --out_dir ../rendering_test --r 2 --delta_theta 30 --sz_x 640 --sz_y 480
```
**results:**
rendered object rgb images, and mask (alpha(alpha!=0)==255)



---
2. #### [render/blender/render_dir_script.py](render/blender/render_dir_script.py)
is the folder version for the above, render all obj data for one scene (under `obj_dir` is all the obj for one scene) with 5 camera view point settings


```
python render_dir_script.py --obj_dir ./data/objs/0 --out_dir rendering_test --r 2 --delta_theta 30 --sz_x 640 --sz_y 480
```

generate rendered images iteratively from mesh models

**notes**

```
shape_files = [[os.path.join(args.obj_dir, x)] for x in sorted(os.listdir(args.obj_dir)) if x.endswith('.obj')]

if not 'sil_images' in f.keys():
    f.create_dataset('sil_images', data=data)
elif f['sil_images'][...].shape != data.shape:
    del f['sil_images']
    f.create_dataset('sil_images', data=data)
else:
    f['sil_images'][...] = data
```
good to render different obj mesh models to one image



---

3. #### [render/blender/render_engine.py](render/blender/render_engine.py)

most important  is the function `_render(params)` 

use blender python package `bpy` to render 

return ims, masks

`tmp_file` is a temporary file to get rendered `rgb-alpha` image

  - [x] camera is choose  by the author, is there any real one for the ground truth. yes! the first one

```
  camera_xyz = np.zeros((5,3))
  lookat_xyz = np.zeros((5,3))
  i = 0
  r = args.r
  for l in [0, 2]:
    for t in [-args.delta_theta, args.delta_theta]:
      i = i+1
      t = np.deg2rad(t)
      camera_xyz[i,l] = r*np.sin(t)
      camera_xyz[i,1] = r*np.cos(t) - r
  lookat_xyz[:,1] = -r
```

```
| camera_xyz | x          | y            | z            | lookat_xyz | x     | y    | z    |      |              |
| :--------- | ---------- | ------------ | ------------ | ---------- | ----- | ---- | ---- | ---- | ------------ |
| 0          | 0          | 0            | 0            |            | 0     | -2   | 0    |      | ground truth |
| 1          | 2*sin(-30) | 2*cos(-30)-2 | 0            |            | 0     | -2   | 0    |      |              |
| 2          | 2*sin(30)  | 2*cos(30)-2  | 0            |            | 0     | -2   | 0    |      |              |
| 3          | 0          | 2*cos(-30)-2 | 2*sin(-30)   |            | 0     | -2   | 0    |      |              |
| 4          | 0          | 2*cos(30)-2  | 2*sin(30)    |            | 0     | -2   | 0    |      |              |
|            |            |              |              |            |       |      |      |      |              |
|            |            |              |              |            |       |      |      |      |              |
|            | [ 0.       | 0            | 0.        ]  |            | [[ 0. | -2   | 0.]  |      |              |
|            | [-1.       | -0.26794919  | 0.        ]  |            | [ 0.  | -2   | 0.]  |      |              |
|            | [ 1.       | -0.26794919  | 0.        ]  |            | [ 0.  | -2   | 0.]  |      |              |
|            | [ 0.       | -0.26794919  | -1.        ] |            | [ 0.  | -2   | 0.]  |      |              |
|            | [ 0.       | -0.26794919  | 1.        ]  |            | [ 0.  | -2   | 0.]] |      |              |
```

---
4. #### [render/blender/render_utils.py](render/blender/render_utils.py)

```
python
quaternionProduct()
camRotQuaternion()
camPosToQuaternion()
quaternionFromYawPitchRoll()
camPosToQuaternion()
get_calibration_matrix_K_from_blender()
obj_centened_camera_pos()
```

```
get_calibration_matrix_K_from_blender ()
pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
s_u = resolution_x_in_px * scale / sensor_width_in_mm
s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
alpha_u = f_in_mm * s_u
alpha_v = f_in_mm * s_v
u_0 = resolution_x_in_px*scale / 2
v_0 = resolution_y_in_px*scale / 2
skew = 0 # only use rectangular pixels

K = np.array(
[[alpha_u, skew,    u_0],
[    0  ,  alpha_v, v_0],
[    0  ,    0,      1 ]])   
```


resolution_x_in_px - 1280

resolution_y_in_px - 960

sensor_width_in_mm - 640.0

sensor_height_in_mm - 480.0

scale - 0.5




