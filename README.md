# STL Surrogate

Algorithms and data are available to create ML-based surrogate models for the STL problem.
Four STL models were addressed:

* Analytical model for infinite plates

* Correction factor approach for finite plates

* Modal Summation model with and without 1/3 octave band average

* Finite Element Model (FEM) using Comsol with and without 1/3 octave band average
 
The variables are the plate thickness h, density ρ, Young's Modulus E, Poisson's ratio ν, damping factor η, and, for the case of finite plates, the plate width a and length b. The outputs are the STL values for each frequency in the df_freq files. 

Four ML algorithms are used to create the surrogates:
* Neural Networks (NN)
* Guassian Process Regressor (GPR)
* Random Forest (RF)
* Gradient Boosting Trees (GBT)

In addition, Mean Decrease in Impurity (MDI)-based Sensitivity Analysis is performed with Random Forest constructed with Multi-Output Regressor, and the addiction of physics-guided features is investigated.
