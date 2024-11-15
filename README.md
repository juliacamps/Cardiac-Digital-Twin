# Cardiac-Digital-Twin
This repository contains the code developed for the study:

_Harnessing 12-lead ECG and MRI data to personalise repolarisation profiles in cardiac digital twin models for enhanced virtual drug testing_

https://doi.org/10.1016/j.media.2024.103361

## Example data 
The meshes, ECGs, electrodes, cellular models, digital twins, etc. to run this code, generate new digital twins, and reproduce the results in the paper can be found in the Cardiac_Digital_Twin_Data repository in Zenodo at https://doi.org/10.5281/zenodo.14034739.



##	Code structure of the digital twinning pipeline

The code has been packaged for ease of adoption and future extension. The digital twinning code is mostly in Python, except for the scripts for generating the action potential look-up tables in Matlab.
The code is divided into modules with distinct responsibilities to facilitate integration with other existing workflows. These modules and their responsibilities are as follows:
-	Geometry: classes for the geometrical information of the ventricles.
-	Conduction system: determines how the conduction system should be handled for a given geometry.
-	Cellular: contains the action potential simulations required by the reaction-Eikonal.
-	Propagation: contains implementations of different models for the propagation of the electrical wavefront in the ventricles, such as the Eikonal and the reaction-Eikonal models. This does not contain the monodomain implementation, which was simulated using the MonoAlg3D open-source solver. 
-	Electrophysiology: defines how the cellular and propagation modules interact.
-	Electrocardiogram: is responsible for calculating and processing the ECG recordings.
-	Simulation: determines what data will be simulated using the Electrophysiology and Electrocardiogram modules.
-	Discrepancy: contains different strategies for calculating discrepancy metrics that compare clinical ECG data with simulated ECG data.
-	Evaluation: determines the output from evaluating a set of parameter values (e.g., discrepancy to clinical data, ECG signal, repolarisation map, etc.) and links the classes of the Simulation and Discrepancy modules.
-	Sampling: classes for implementations of the inference and sensitivity analysis methods that interface with the Evaluation module.
-	Adapter: handling class for all the input and sampled parameters. This module controls which parameters are being sampled (theta) and makes sure that they are forwarded to the correct modules. 

These modules are coupled with each other as follows (Module name: main function name):
(Glossary) In the code, theta refers to the parameters being sampled, while the parameter refers to the combination of sampled and prescribed parameters.
-	Sampling: sample_theta(population size)
  -	Evaluation: evaluate_theta(theta)
    -	Discrepancy: evaluate_metric(simulation data)
    -	Adapter: translate_theta_to_parameter(theta) 
    -	Simulation: simulate(parameter)
      -	Electrocardiogram: calculate_ecg(electrophysiology data)
        -	Geometry: None
      -	Electrophysiology: simulate_electrophysiolgy(parameter)
        -	Propagation: simulate_propagation(parameter)
          -	Geometry: None
            -	Conduction system: generate_Purkinje(ab, rt, tm, tv)
            -	Cellular: generate_action_potential(APD)

Generic and shared functions across modules can be found in the utils or input-output (io) modules. 
The code contains several main scripts for different use cases. 
Each of them initialises and links classes from these modules explicitly to serve as templates for future use cases. 
For example, to include a scar and perform inference on its conduction properties, it would be necessary to extend the geometry module to have a class with a scar and to incorporate those parameters in the propagation module and main script.
