import bpy
import math
import numpy as np

# environment light ------------------------------------
bpy.context.scene.world.light_settings.use_environment_light = True # blender 2.79
bpy.context.scene.world.light_settings.environment_color = 'PLAIN'
#bpy.context.scene.world.light_settings.environment_energy=0
bpy.context.scene.world.light_settings.environment_energy=1

rng = np.random.RandomState(0)


# set lamp light ----------------------------------------
# lin:
# light number 2-8
g_syn_light_num_lowbound = 2
g_syn_light_num_highbound = 8
# light distance 8-20
g_syn_light_dist_lowbound = 6
g_syn_light_dist_highbound = 10
# lin: light location looks too random
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 360
g_syn_light_elevation_degree_lowbound = -90
g_syn_light_elevation_degree_highbound = 90
# lin: enhance lighting
# before --------
# g_syn_light_energy_mean = 2
# g_syn_light_energy_std = 0.01
# g_syn_light_environment_energy_lowbound = 0
# g_syn_light_environment_energy_highbound = 2.01
# after ---------
g_syn_light_energy_mean = 2
g_syn_light_energy_std = 0.01
g_syn_light_environment_energy_lowbound = 0.2
g_syn_light_environment_energy_highbound = 0.5


# delete all objects except the camera and object
# to_delete_objects = set(bpy.data.objects.keys())-set(['Camera']) - set(['pred_codes'])
to_delete_objects = set(bpy.data.objects.keys())-set(['Camera']) - set(['gt_codes.002'])
bpy.ops.object.select_all(action='DESELECT')
for s in list(to_delete_objects):
    bpy.data.objects[s].select = True
    bpy.ops.object.delete()

def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

for i in range(rng.randint(g_syn_light_num_lowbound, g_syn_light_num_highbound+1)):
        light_azimuth_deg   = rng.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
        light_elevation_deg = rng.uniform(g_syn_light_elevation_degree_lowbound, g_syn_light_elevation_degree_highbound)
        light_dist          = rng.uniform(g_syn_light_dist_lowbound, g_syn_light_dist_highbound)

        lx, ly, lz          = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
        obj = set(bpy.data.objects.keys())
        bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(lx, ly, lz))
        new_lamp_name = 'Point' if i == 0 else 'Point.{:03d}'.format(i)
        bpy.data.objects[new_lamp_name].data.energy = rng.normal(g_syn_light_energy_mean, g_syn_light_energy_std)
