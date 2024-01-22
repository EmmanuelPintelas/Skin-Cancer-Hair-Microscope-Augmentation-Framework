# Skin Cancer - Hair-Microscope - Augmentation Framework

The main challenge in Skin Cancer detection problem, is lying on adressing the unstable learning state due to the high noise ratio. 
Common types of noise found in Skin-Cancer images which downgrade the CNNs' learning efficacy involves the Microscope and Hair artifacts.

This Algorithm can be used for simulating and injecting realistic Microscope and Hair artifacts for image Augmentation to enhance Skin Cancer diagnosis tasks.


    Instructions to use:
    1st - Manually select some sample images (e.g. 10-50) containing the Microscope and Hair noise
    2nd - Run the _Craft_Microscope_Effect_ or _Craft_Hair_Effect_ scripts specifying the root and dist dirs, and number of clusters. The dist dirs is the output of this script.
    Running these scripts, they will segment the Microscope or Hair artifact (with also non-noise segmented regions based on the number of input clusters)
    for every sample.
    3rd - From the output, select the Microscope and Hair regions images, discarding the non-noise segmented regions), and save them on a folder
    e.g. Sample_Microscope_Regions, Sample_Hair_Regions
    4th - Now, you can use the Sample_Microscope_Regions and Sample_Hair_Regions via the _Injection_Algorithm_, which will fuse the segmented Noise_Regions into
    new case images, in order to perform image augmentation for Skin-Cancer tasks
