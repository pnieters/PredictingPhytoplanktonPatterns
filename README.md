# PredictingPhytoplanktonPhenomtypes
Public Repository for the Paper "Predicting environmental drivers of phytoplankton bloom phenotypes"

## Authors: 
1. Maximilian Berthold, Department of Biology, Mount Allison University, Canada
1. Pascal Nieters, Institute of Cognitive Science, University Osnabrück, Germany
1. Rahel Vortmeyer-Kley, Institute for Chemistry and Biology of the Marine Environment, University Oldenburg, Germany

*All authors contributed equally. Correspondence addressed to: rahel.vortmeyer-kley@uni-oldenburg.de*

## Code-License
The code is licensed under an MIT-License (c) 2023, Berthold, Nieters, Vortmeyer-Kley

## Data
We supply z-scored data in .h5 format in TidyData/. The data was originally collected by a third party and are under a CC BY-NC-ND license.
```
The raw data was supplied by the State Agency for Environment, Nature Conservation and Geology Mecklenburg-Vorpommern (LUNG-MV). The here presented data follows CC BY-NC-ND and the LUNG-MV needs to be credited as the creator.
```

## Quick Guide:
We used matlab to preprocess data according to the procedure desribed in `PreparingTrainingData.md` (you can just read the .h5)

R was used to fit the GAMM model to the data.

Julia was used to setup the Universal Differential Equation and Artificial Neural Network as well as to use SInDy on the function the network learned.

The julia implementation relies on additional code found in `src/`. The ensemble model on data from all years gets fit in `meanmodel.jl`, all results are analyzed, plotted, and SInDy is applied in `results_and_sindy.jl`