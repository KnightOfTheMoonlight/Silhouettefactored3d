\#Install blender as a module with python 3.5 and pyenv

Tested on Ubuntu 14.04.

### Setting up a new python environment using pyenv

Follow instructions from [here](https://gist.github.com/alexlee-gk/ba07524dc0d972be9eac#setting-up-a-new-python-environment-using-pyenv).

### Installing boost

Follow instructions from [here](https://gist.github.com/alexlee-gk/ba07524dc0d972be9eac#installing-boost).

### Installing blender as a module

The instructions are mostly the same as the [official installation instructions](http://wiki.blender.org/index.php/User:Ideasman42/BlenderAsPyModule) except for a few modifications specified below.

Install the python dependecies using `pip`:

```
pip install numpy
pip install requests
```

When blender is build as a module, the `blender` binary doesn't get built. So, first build blender as normal following   https://wiki.blender.org/wiki/Building_Blender/Linux/Ubuntu/CMake 



**Run `install_deps.sh` to generate the cmake options.** For example, build all libraries except for opensubdivision, opencollada and ffmpeg:

--source: 源码存放的位置

```
./blender/build_files/build_environment/install_deps.sh --source ./ --threads=4 --with-all --skip-osd --skip-ffmpeg
```

When using cmake, use the following python options (in addition to any other options returned from the command above that you need):

```
cmake -DPYTHON_VERSION=3.5 -DPYTHON_ROOT_DIR=~/.pyenv/versions/3.5.1 ../blender
```

Make sure to build it and install it:

```
make -j4
make install
```

This should have created the blender binary `bin/blender`. Now, build blender as a module as described in the [original post](http://wiki.blender.org/index.php/User:Ideasman42/BlenderAsPyModule) (in addition to any other options):

````
cmake -DWITH_PLAYER=OFF -DWITH_PYTHON_INSTALL=OFF -DWITH_PYTHON_MODULE=ON ../blender
``　ｍａｋｅ的時候會報錯

`cmake-gui` 只留WITH_PYTHON_MODULE=ON 錯

Build it an install it:

​```
make -j4
make install
​```

This should have created the python library `bin/bpy.so`.
````

