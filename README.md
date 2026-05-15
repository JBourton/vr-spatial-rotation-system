### Running the spatial rotation renderer

### Project overview
This project renders the Stanford bunny as a simple VR-style orientation visualiser driven by the supplied IMU dataset. In `render.py`, the gyroscope readings are integrated into quaternion orientation estimates, then corrected with complementary filters: the first sequence fuses gyroscope and accelerometer data for roll/pitch stability, while the second sequence adds magnetometer correction to reduce yaw drift. The central bunny uses this fused orientation, while additional falling bunnies are spawned with simple gravity, drag, and collision behaviour to make the rendered scene more dynamic.

The renderer is deliberately software-based: the mesh is loaded from `bunny.obj`, normalised into view, transformed with matrices, rasterised into a framebuffer, and displayed with OpenCV. This keeps the coursework focus on the transformation, filtering, and rendering pipeline rather than relying on a game engine.

### Static render reference
![Static renders showing rest frame, peak roll, peak pitch, and peak yaw](/imgs/static_renders.png)

The static render above is a quick visual reference for the orientation behaviour used by the main sequence. It shows the bunny at its rest frame, then at the peak roll, pitch, and yaw poses dictated by the orientation code in `render.py`. Internally, each fused quaternion is converted back into Euler angles with `convert_quaternion_to_euler`, and the central bunny's model matrix applies those rotations as `matrix_rotater_z(yaw) @ matrix_rotater_y(pitch) @ matrix_rotater_x(roll)`. The static image isolates those maximum-axis poses to show what each rotational component is doing before watching the full time-varying render.

### Installing dependencies
I've only used basic libraries for this, so chances are, these libraries are already on your computer. However, if they're not, you can install them with these instructions.

I reccomend using a virtual environment to install them as other students library versions might be slighly different to mine, but that's not essential.

I ran this in a venv setup with python 3.11.1, and ran a pip freeze > requirements.txt.
To install, please run this command in your terminal in VSCode (or wherever else you run this):
```pip install -r requirements.txt```

If that doesn't work, please just run: 
```pip install numpy opencv-python pandas scipy matplotlib```
in the command line, and that should install the same libraries as present in my requirements.txt.


### Running the sequences
I've set the sequences up to run in two seperate videos. Therefore, once the first video completes, please press any key to close off that window, at which point the 
second sequence will start up automatically.

Once the second sequence has finished, just press any key when on the window to close off the program.


### report_analysis_metrics.py
I've included this file in my submission for auditability's sake, as it contains the calcualtions used to present the data I disucss in my report. You can run this with:
```python report_analysis_metrics.py``` 
to see the results printed to the screen. If there's minor variations in the output, it's just down to the fact I haven't randomseeded it.

I've set this file up in 3 segmenets, one for each section of data I report with, and start the file with copy-paste functions from my ```render.py``` file to keep everything self-contained. 
