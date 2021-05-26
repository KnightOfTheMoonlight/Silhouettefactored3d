
use `intsall_deps.sh` to install oiio openimageio package, to keep use the compiled bpy.so from previous successfully installed ones.
change oiio to 1.8 based version
`OIIO_VERSION="1.8.16"`

refer to [doc/blender_installaspymodule.md](doc/blender_installaspymodule.md) and [doc/silhouette_generation_guide.md](doc/silhouette_generation_guide.md)

```
./blender/build_files/build_environment/install_deps.sh --with-all
```

```
official guide
https://wiki.blender.org/wiki/Building_Blender/Linux/Ubuntu

cmake cmake-gui git
```
```
----python setting----
--anaconda python3.5--
source activate bpy35
pip install scipy==1.1.0 numpy
source activate bpy35
```
```
----source code------
cd  ~/code
mkdir blender-git
cd blender-git
git clone https://git.blender.org/blender.git
cd blender

# other setting
git submodule update --init --recursive
git submodule foreach
git checkout master
git submodule foreach
git pull --rebase origin master


# make sure it's 2.79 version
git checkout origin/blender-v2.79-release

# install dependencies (generate cmake options)
cd ..
./blender/build_files/build_environment/install_deps.sh --source ./installDeps --threads=4 --skip-osd --skip-ffmpeg


ref:
./blender/build_files/build_environment/install_deps.sh --source ./installDeps --threads=4 --with-all --skip-osd --skip-ffmpeg  -skip-osl --skip-opencollada


cd blender
sudo make bpy -j24 (will fail; only prepare for python api)
cd ..
sudo ccmake build_linux_bpy #()

 OpenColorIO ON                     
                      anaconda 3.5                                                                                                                                       
 PYTHON_EXECUTABLE                                                                      /home/li216/anaconda3/envs/bpy35/bin/python3.5                                                                                            
 PYTHON_INCLUDE_CONFIG_DIR        /home/li216/anaconda3/envs/bpy35/include/python3.5m                                                                                                                                                                                                       
 PYTHON_INCLUDE_DIR                                                                                                                            /home/li216/anaconda3/envs/bpy35/include/python3.5m                                                                                             
 PYTHON_LIBPATH                   /opt/lib/python-3.5/lib                                                                                                                            /home/li216/anaconda3/envs/bpy35/lib                                                                                             
 PYTHON_LIBRARY                   /opt/lib/python-3.5/lib/libpython3.5m.a                                                                                                           /home/li216/anaconda3/envs/bpy35/lib/libpython3.5m.so                                                                                             
 PYTHON_LINKFLAGS                 -Xlinker -export-dynamic                                                                                                                                                                                                                        
 PYTHON_SITE_PACKAGES             /opt/lib/python-3.5/lib/python3.5/site-packages                                                                                                    /home/li216/anaconda3/envs/bpy35/lib/python3.5/site-packages                                                                                             
 PYTHON_VERSION                   3.5      

 WITH_CYCLE						OFF (ON)


 rebuild bpy

cd blender
(sudo make bpy clean, doesnot work very well)
sudo make bpy -j
```


