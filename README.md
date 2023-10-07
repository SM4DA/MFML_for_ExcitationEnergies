# Multi-Fidelity Machine Learning

This repository contains scripts to reproduce the various results of the research paper (available at XXXX). 

The scripts are written in python. The files requirements.txt lists the versions of the libraries used. The rawdata is stored in the directories `Data` and `Evaluation`.

The scripts perform the following:
* `MFML_LearningCurves.py` generates the outputs for all the the molecules and trajectories. These are used to generate the plots of Figure 3, Figure S5, and the scatter plots of Figure S7. The same outputs are used to generate Figure 4. 
* The script `MFML_QZVP_LC.py` generates the outputs used to generate Figure 5. 
* `MFML_AntDFTB_analysis.py` generates the output to generate Figure S6. 
* `SingleDifferenceML_LearningCurves.py` generates the two-level ML model as seen in Figure S8.
