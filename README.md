**NOTE:** Description below is from the submisstion of this repository as a final project for ME570: Robot Motion Planning at Boston University. The description may become out-of-date before the end of the Spring 2023 semester (as the project is ongoing). To see the full results, please refer to the tag titled **v1.0** within this repository.

___

### ME 570: Final Project
Title: An Exploration of the Koopman Operator for Modeling Controlled Dynamics and Distance Propagation

For final results execute: ``main_results.m`` in the root folder.

___

This folder contains the files necessary for testing and plotting the results of a paper submitted as a final project for the class ME 570: Robot Motion Planning at Boston University. It contains two methods for computing the Koopman operator for a system with holonomic dynamics and spherical obstacles; analytically and through the data-driven least-squares approach.

The files are separated into the following folders:

0. Root:
	- ``fdm.m``
1. HolonomicPoint: main execution files for testing and report
    - ``main_results.m``
    - ``analytical.m``
    - ``datadriven.m``
    - ``model.m``
2. Data: pre-computed Koopman operators for use in ``main_results.m``
    - ``K_24x24_analytical.mat``
    - ``K_24x24_datadriven.mat``
3. DataFunctions: for processing and organizing training data
    - ``generate_data.m``
    - ``stack_data.m``
4. KoopFunctions: for observation functions and computing Koopman operators
    - ``Koopman.m``
    - ``KoopmanWithControl.m``
    - ``KoopmanAnalytical.m``
    - ``observables.m``
    - ``observables_partial.m``
5. PlotFunctions: for plotting data (primarily used in testing)
    - ``plot_comparisons.m``
    - ``plot_path.m``
6. SphereWorld: environment functions and data
    - ``animate.m``
    - ``distance.m``
    - ``plot_sphere.m``
    - ``plot_sphereworld.m``
    - ``sphereworld.mat``
    - ``sphereworld_minimal.mat``
