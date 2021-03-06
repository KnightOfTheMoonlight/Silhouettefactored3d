# Silhouettefactored3d

## Introduction

This repository contains parts of the implementation of **Silhouette-Assisted 3D Object Instance Reconstruction from a Cluttered Scene.** (Silhouettefactored3d),  
presented in our ICCV workshop 2019 paper ([Link](https://openaccess.thecvf.com/content_ICCVW_2019/papers/3DRW/Li_Silhouette-Assisted_3D_Object_Instance_Reconstruction_from_a_Cluttered_Scene_ICCVW_2019_paper.pdf)). If you find our work useful in your
research, please consider citing:

```
@inproceedings{li2019silhouette,
  title={Silhouette-assisted 3d object instance reconstruction from a cluttered scene},
  author={Li, Lin and Khan, Salman and Barnes, Nick},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
\  year={2019}
}
```

## demo gif compare vs factored3d

* factored3d
<img src="factored3d.gif">
* ours
<img src="ours.gif">

## Experiments

All rendering are based on blender engine 2.79. [notice: 2.80 version is a whole different story]
* [blender_setup](./doc/blender_setup.md): Instructions to setup blender environment.
* [Silhouette Generation](./doc/silhouette_generation_guide.md): Instructions to generate amodal silhouette.
* [Final Result Visualization](./doc/final_result_visualization_guide.md): Instructions to visualize the final result.
