# Multi-Fidelity Machine Learning
(For Molecular Excitation Energy)

This repository contains scripts to reproduce the various results of the research paper (available at 10.1021/acs.jctc.3c00882). 

The scripts are written in python. The files requirements.txt lists the versions of the libraries used. The rawdata is stored in the directories `Data` and `Evaluation`.

The scripts perform the following:
* `MFML_LearningCurves.py` generates the outputs for all the the molecules and trajectories. These are used to generate the plots of Figure 3, Figure S5, and the scatter plots of Figure S7. The same outputs are used to generate Figure 4. 
* The script `MFML_QZVP_LC.py` generates the outputs used to generate Figure 5. 
* `MFML_AntDFTB_analysis.py` generates the output to generate Figure S6. 
* `SingleDifferenceML_LearningCurves.py` generates the two-level ML model as seen in Figure S8.

All plots can be generated via the jupyter notebook after suitable directory modifications while loading data.
