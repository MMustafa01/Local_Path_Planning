
- [CARLA Codes](#carla-codes)
  - [Using CARLA](#using-carla)
  - [Installation and Building CARLA](#installation-and-building-carla)

# CARLA Codes
![Alt text](image.png)

## Using CARLA 
## Installation and Building CARLA
Note: this will be for window users. 
\
CARLA gives two options to build CARLA on a Windows setup. The first is directly downloading CARLA files from its repository. And, second is using Docker. For the Docker installation, it seems like one would need a CUDA capable GPU. And, even after that, I faced a lot of issues using DOCKER. Thus, I would instead suggest the following steps:
1. Installing Anaconda Navigator from:
2. Installing visual studio c++ runtime.
3. Intsalling DirectX 11.
4. Creating a conda environment with:
   1. Python version 3.7
   2. Numpy latest version
   3. Add the rest of the Python packages you might find relevant. I would suggest:
      1. Open CV
      2. Gym
      3. Tensorflow
      4. Pillow
      5. Matplotlib
      6. Pytorch
      7. Stable-Baselines3
5. Download CARLA packages from its repository. I would suggest installing CARLA version >= 12. Note: Use the documentation of whatever version of CARLA you are using.
6. Open the terminal of the conda environment. Go to the directory ../PythonAPI/CARLA/Dist. There you will find a <.whl> file. Rund the code: 
```
pip install <filename>.whl
```
Now you can directly import CARLA in any Python script. Otherwise, you would've had to use the following method:
```
sys.path.append(glob.glob('C:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
import carla
```
7. Now before running any script. You must start a local instance of CARLA running on your computer, or a remote server that you can connect to. For that:
   1. Activate the conda environment in the Terminal.
   2. Navigate to a folder named WindowsNoEditor. (In the downloaded files).
   3. Run the following ``` .\CarlaUE4 -dx11``` 
8. Now you can run a Python script.
