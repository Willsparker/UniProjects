## CS3VI18 Module: Kinect-Based Object Recognition

The specification can be read at [Edited_Specification.md]. This contains information about the assignment, without any additional information specific to University of Reading, that would be superfluous (or not safe to Open Source).

The report has also been edited to remove student numbers and only include my own name.

The work is demonstrated in the *.py* files in *VI-2020-21-Python_Package/*. Unfortunately, I can't provide the datasets as they were not owned by me, however the sets came in the form of 2 directories, named `Set1` and `Set2`. They both contained short videos of a person holding up various objects to an XBox 360 Kinect Sensor, in both depth and the RGB representations. `Set1` was the training set, and the sequence of the objects shown can be found in *VI-2020-21-Python_Package/Set1Labels.txt*. 

The Freenect Framework (and argument parser) in *cv-demo.py* was provided as part of the assignment. The rest of the code is my own. 

The assignment was originally expected to run in a virtual environment, hence why there's no information on setting up Freenect, and why the python files are unlikely to work out the box - these are more just for reference, I suppose :-) Examples of the object recognition working can be found in the Report PDf anyway. 
