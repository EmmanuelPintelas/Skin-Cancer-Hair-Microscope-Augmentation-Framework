# Skin-Cancer---Hair-Microscope---Augmentation-Framework
This Algorithm can be used for simulating and injecting realistic microscope and hair artifacts for image Augmentation in Skin Cancer detection tasks


    Instructions to use:
    1st - manually select some sample images (e.g. 10-50) containing the Microscope noise
    2nd - Run this _Craft_Microscope_Effect_ script specifying the root and dist dirs, and number of clusters. The dist dirs is the output of this script.
    Running the script, it will segment the Microscope artifact (with also non-Microscope segmented regions based on the number of input clusters)
    for every sample.
    3rd - From the output, select the Microscope regions images, discarding the non-Microscope segmented regions), and save them on a folder
    e.g. Sample_Microscope_Regions
    4th - Now, you can use the Sample_Microscope_Regions via the _Injection_Algorithm_, which will fuse the segmented Microscope_Regions into
    new case images, in order to perform image augmentation for Skin-Cancer tasks
