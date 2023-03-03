# DREAM_LOA: Limits of Acceptability with DREAM algorithm (check also DREAM-Package)

## Description

This essay illustrates some recent developments to the DiffeRential Evolution Adaptive Metropolis (DREAM) MATLAB toolbox of \cite{vrugt2016} to delineate and sample the behavioural solution space of set-theoretic likelihood functions used within the GLUE (Limits of Acceptability) framework \citep{beven1992,beven2001,beven2006,beven2014}. This work builds on the DREAM$_\text{(ABC)}$ algorithm of \cite{sadegh2014} and enhances significantly the accuracy and CPU-efficiency of Bayesian inference with GLUE. In particular it is shown how lack of adequate sampling in the model space might lead to unjustified model rejection.

## Getting Started

### Installing

* Download and unzip the zip file 'DREAM_LOA_MATLAB_Code_Oct_10_2017.rar'.
* You can then run the script 'run_NC_model.m' available in 'DREAM_LOA_MATLAB_Code_Oct_10_2017.rar'

### Executing program

* After intalling, you can simply execute the script 'run_NC_model.m'. This is an application to a Nash-Cascade of reservoirs. 
* Please refer to DREAM-Package for a full implementation of the DREAM-LOA algorithm. With many more examples and functionalities

## Authors

* Vrugt, Jasper A. (jasper@uci.edu) 

## Version History

* 1.0
    * Initial Release
    * After the initial release, I merged DREAM/DREAM_ZS/DREAM_D/DREAM_DZS/DREAM_BMA and DREAM_LOA into one algorithm called DREAM_Package. Download this instead

## Acknowledgments
