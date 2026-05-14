### How to run my coursework attempt

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