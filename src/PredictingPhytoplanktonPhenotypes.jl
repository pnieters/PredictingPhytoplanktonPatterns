module PredictingPhytoplanktonPhenotypes

using LinearAlgebra
using OrdinaryDiffEq
using Zygote
using Flux
using Dates
using HDF5
using DataFrames

include("data_tools.jl")
export CallbackLog, 
       data_matrix_from_hdf5,
       dataframe_from_hdf5, 
       select_timeseries, 
       DataPoint,
       stack_data_by_year


include("ml_tools.jl")
export polar_loss, mse_loss, train_ude

end # module AlgaeBlooming
