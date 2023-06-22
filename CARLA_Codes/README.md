# Running a CARLA SCRIPT

- [Running a CARLA SCRIPT](#running-a-carla-script)
  - [To run any CARLA scrip.](#to-run-any-carla-scrip)


## To run any CARLA scrip.
1. First, have a CARLA server running on your PC or connect to a remote CARLA server.
2. Find the python .egg files using the Syntext

```
sys.path.append(glob.glob('C:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
```
Make sure that the correct path for .egg files is listed. They are usually in the pythonAPI/carla/dist folder. 