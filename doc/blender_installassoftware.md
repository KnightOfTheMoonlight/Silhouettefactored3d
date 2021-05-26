7.2 開始

## Downloading Blender Source

You need git. 7 billion people on this planet know what [git](https://git-scm.com/) is. If you don’t, join the 7 billion today.

We will also need to download `build-essentials` for use.

| 1    | sudo apt install git build-essentials |
| ---- | ------------------------------------- |
|      |                                       |

Then, time to download Blender. This is what I do, and I’ll be using those steps.

I keep the `Build` folder (where the completed compilation files will be including the blender executable), the `Source` (where the actual Blender will be downloaded to) and the `installDeps` folder where the dependencies for installing and building blender are found.

So let’s get going.

| 1    | mkdir ~/Blender                                |
| ---- | ---------------------------------------------- |
| 2    | cd ~/Blender                                   |
| 3    | mkdir Source                                   |
| 4    | cd Source/                                     |
| 5    | git clone http://git.blender.org/blender.git . |
| 6    | git submodule update --init --recursive        |
| 6.5  | git submodule foreach                          |
| 7    | git checkout master                            |
| 8    | git submodule foreach                          |
| 9    | git pull --rebase origin master                |

In English

- Make a directory called Blender in home directory, as in, `/home/khophi/Blender`
- Change directory to the `Blender` folder
- Make another directory in the `Blender` folder called `Source` and change directory to the `Source` folder
- Clone the Blender repo into the same folder, as indicated with the `dot` (`.`) at the end of line 5.
- The remaining commands are just to recursively update other submodules that come with Blender source

To update you simply use the command:

| 1    | git pull --rebasegit submodule foreach |
| ---- | -------------------------------------- |
| 2    | git pull --rebase origin master        |

## Installing Blender Dependencies

約麼估計一小時

change something in the `install_deps.sh` file

```
PYTHON_VERSION="3.5.2"
PYTHON_VERSION_MIN="3.5"
```

We want to install all the Blender Dependencies automatically using the Blender `install_deps.sh` script.

| 1    | cd ~/Research/software/Blender                               |
| ---- | ------------------------------------------------------------ |
| 2    | ./Source/build_files/build_environment/install_deps.sh --source /home/li216/Research/software/Blender/installDeps --threads=4 --with-all --skip-osd --skip-ffmpeg |

ｎｏｔｅ: 用with-all　會安裝不必要的庫譬如opencollada　不必ｆｏｌｌｏｗ

Now that can take some time, so give it a break.

The `install_deps.sh` takes in some options, namely:

| --source <path>  | Where to store downloaded sources for libraries we have to build (defaults to ~/src/blender-deps). |
| ---------------- | ------------------------------------------------------------ |
| --install <path> | Where to install the libraries we have to build (defaults to /opt/lib). |
| --with-all       | Include some extra libraries that are by default not considered mandatory (main effect of this one is building OpenCollada). |

Hopefully, all should install properly and complete without errors. If there’re any errors, make sure you do fix them all following the error descriptions. Troubleshooting the errors aren’t something I can list here and the possible errors can be many.

 

[wp_ad_camp_1]

 

## cuda9 support

change cycle setting with cuda as:

```
https://developer.blender.org/rBf55735e533601b559d53fd1e2c5297092e844345
/intern/cycles/kernel/CMakeLists.txt
if((CUDA_VERSION MATCHES "90" or CUDA_VERSION MATCHES "91") AND ${arch} MATCHES "sm_2.")
```



## Compiling Blender

As much as I love using the terminal when going the GUI way makes sense, I’m all in. So we’ll be using `cmake` but it also has a GUI, called `cmake-gui`

| 1    | sudo apt install cmake cmake-gui |
| ---- | -------------------------------- |
|      |                                  |

After installation, go:

| 1    | mkdir Blender_ROOT/Build |
| ---- | ------------------------ |
| 2    | cd Blender_ROOT/Build    |
| 3    | cmake-gui                |

That should open up a CMake GUI interface like so

[![Blender CMake GUI](https://blog.khophi.co/wp-content/uploads/2016/07/Screenshot-from-2016-07-25-12-39-50-700x394.png)](https://blog.khophi.co/wp-content/uploads/2016/07/Screenshot-from-2016-07-25-12-39-50.png)Blender CMake GUI

Answer the “Where is the source code” and “Where to build the binaries” questions with the right directories. If you’re using folder structures similar to mine, you should check the image to see where my Build and Source folders are respectively.

Press on the `Configure` button. You should see no errors. If you do, fix them. One way to fix them is to use Synaptic Package Manager to download the respective libraries missing.

1. **After first configuration, change cmake flags do as**

```
 ~/Research/software/Blender/BUILD_NOTES.txt 
```

2. python setting

```
-D WITH_PLAYER=OFF 
-D WITH_PYTHON_INSTALL=OFF 
-D WITH_PYTHON_MODULE=OFF
-D PYTHON_VERSION=3.5

```

second `configure`

For instance, if `boost` related libraries are plaguing you, via Synaptic Package Manager, you can install the right libraries.

[![Synaptic Package Manager in use](https://blog.khophi.co/wp-content/uploads/2016/07/Screenshot-from-2016-07-25-12-46-37-700x394.png)](https://blog.khophi.co/wp-content/uploads/2016/07/Screenshot-from-2016-07-25-12-46-37.png)Synaptic Package Manager in use

If you get no more errors, then it means you’re on good track, and then press the `Generate` button.

Generation should begin, and when all goes well, you should see something like in the previous image.

Close `cmake-gui` then change directory to the `Build` folder. In there, enter:

| 1    | make -j 16 |
| ---- | ---------- |
|      |            |

The `-j 16` means it’ll use 16 threads. I’m on an 8-core processor, so twice that is usually fine, and builds okay. If you’re on a lower core, like say, 4, using `-j 8` is fine.

If you’re building for the first time, this can take some time, and even more if you’re on an HDD instead of SSD, like I am.

Keep staring at the compiling process. You’ve earned it.

[wp_ad_camp_1]

Afterwards,

| 1    | make install -j 16 |
| ---- | ------------------ |
|      |                    |

If that also goes well, go back to your `Build` folder, and you should see a `bin` folder. In it is your `blender` executable.



## blender python module

```
Build/bin/bpy.so
```



## Blender Launcher

You likely will want to find your Blender file in the launcher so that you could easily launch it when you search for it. And you’ll also want your Blender icon to show up on the sidebar. Well, you can, and let’s do that now.

[![Access Blender in Launcher](https://blog.khophi.co/wp-content/uploads/2016/07/Screenshot-from-2016-07-25-13-07-26-700x394.png)](https://blog.khophi.co/wp-content/uploads/2016/07/Screenshot-from-2016-07-25-13-07-26.png)Access Blender in Launcher

In the `Build` folder, in the `bin` folder, you’ll find a `blender.desktop`. Copy the file into the `~/.local/share/applications`

| 1    | cp ~/Blender/Build/bin/blender.desktop ~/.local/share/applications |
| ---- | ------------------------------------------------------------ |
|      |                                                              |

Visit the `blender.desktop` file in the `~/.local/share/applications` folder. Right-click and go `Properties`

You should see a box like this. Enter what you see from there, then using the icon button on the top left of the properties dialog, select the Blender.svg file from the `~/Blender/Build/bin` folder

[![blender.desktop properties](https://blog.khophi.co/wp-content/uploads/2016/07/Screenshot-from-2016-07-25-13-14-57-700x394.png)](https://blog.khophi.co/wp-content/uploads/2016/07/Screenshot-from-2016-07-25-13-14-57.png)blender.desktop properties

Our last step is to create the Symbolic link to our actual Blender executable. So go to the `~/Blender/Build/bin` folder the right click on Blender > Make link

Copy the link file created into `~/Blender/` and rename as `Blender`









build blender as python modul by rebuild blender

設置cmake-gui



取消collada

