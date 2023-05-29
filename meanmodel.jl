##############################################
# file to run training on all blooming types to produce data for mean model
# MISSING: parallel loop to speed up training e.g. one bloom type per thread???
##############################################
using Flux
using PredictingPhytoplanktonPhenotypes
using DifferentialEquations
using LinearAlgebra
using Plots
using DiffEqFlux
using HDF5

import Logging
Logging.disable_logging(Logging.Warn) # cheap but effective

# iterations for training
max_iters=500 #300

# define file names to load
Statname =  ["OuTrBlstatallMaxdata.h5",
             "SuBlstatallMaxdata.h5",
             "InTrBlstatallMaxdata.h5",
             "InDuBlstatallMaxdata.h5",
             "DeSpBlstatallMaxdata.h5",
             "AdBlEWstatallMaxdata.h5"]

Startname =  ["OuTrBlstartallMaxdata.h5",
              "SuBlstartallallMaxdata.h5",
              "InTrBlstartallMaxdata.h5",
              "InDuBlstartallMaxdata.h5",
              "DeSpBlstartallMaxdata.h5",
              "AdBlEWstartallMaxdata.h5"]

DFname = ["trainOuTrBloT",
          "trainSuBloT",
          "trainInTrBloT",
          "trainInDuBloT",
          "trainDeSpBloT",
          "trainAdBlEWoT"]

function save_to_path(path,
                     losses, 
                     prediction, 
                     time, 
                     final_weights)

    h5write(path, "Loss", losses)
    h5open(path,"w") do file
        write(file, "Sol", prediction)
        write(file, "time", time)
        # write(file, "WeightsIni", init_weights)
        write(file, "WeightsFin", final_weights)
        # write(file, "LossWeights", loss_weights)
    end
end

function run_experiment(station_id, run_id)

    UA = FastChain(
        FastDense( 5 , 16, gelu, initW=Flux.glorot_normal), 
        FastDense(16 , 16, gelu, initW=Flux.glorot_normal),
        FastDense(16 , 16, gelu, initW=Flux.glorot_normal),
        FastDense(16 , 16, gelu, initW=Flux.glorot_normal),
        FastDense(16 ,  5, gelu, initW=Flux.glorot_normal),
    )

    # keep in this file --> we're going for specific Eqs as part of the exploration
    function UDE(u,p,t)
        z = UA(u,p)
        [z[1],
        z[2],
        z[3],
        z[4],
        z[5]]
    end

    # load trainingdata
    # 1 .  2 . 3 .  4 . 5 . 6 .    7 .    8 . 9 . 10
    # year Chl Temp DIN DIP totalN totalP sal DoY Dayfromfirst
    time_all, y = data_matrix_from_hdf5(pwd() * "/TidyData/" * Statname[station_id],
                                        pwd() * "/TidyData/" * Startname[station_id],
                                        [2,4,5,6,7]) # which vars?

    # initialization startvalues
    u₀ = y[:,1]
    tspan = ( time_all[1], time_all[end] )

    # initial weights
    θ₀ = initial_params(UA)

    #### set up the UDE problem and training of the neural network
    prob_nn = ODEProblem(UDE, u₀, tspan, θ₀)

    loss_weights = [0.2 0.8] 
    predict(θ) = Array(concrete_solve(prob_nn, Tsit5(), u₀, θ, saveat=time_all)) 
    loss(θ) = polar_loss(predict, θ, y; weights=loss_weights)

    ### train the UDE with ADAM
    callback = CallbackLog(T=Float64)
    opt = Flux.Optimiser(
        WeightDecay(1e-5),
        ExpDecay(1e-2,0.5,5000,1e-10), 
        ADAM()
    ) 

    res = DiffEqFlux.sciml_train(loss, θ₀, opt, cb=callback, maxiters=max_iters)

    ret_code = (res.retcode == ReturnCode.Default)

    if ret_code
        println("Successfully completed station station $(Statname[station_id]) run $(run_id)")
    end

    return res, callback, time_all, ret_code

end

function rerun_weights(weights, station_id)
    UA = FastChain(
        FastDense( 5 , 16, gelu, initW=Flux.glorot_normal), 
        FastDense(16 , 16, gelu, initW=Flux.glorot_normal),
        FastDense(16 , 16, gelu, initW=Flux.glorot_normal),
        FastDense(16 , 16, gelu, initW=Flux.glorot_normal),
        FastDense(16 ,  5, gelu, initW=Flux.glorot_normal),
    )

    # keep in this file --> we're going for specific Eqs as part of the exploration
    function UDE(u,p,t)
        z = UA(u,p)
        [z[1],
        z[2],
        z[3],
        z[4],
        z[5]]
    end

    time_all, y = data_matrix_from_hdf5(pwd() * "/TidyData/" * Statname[station_id],
                                        pwd() * "/TidyData/" * Startname[station_id],
                                        [2,4,5,6,7]) # which vars?

    # initialization startvalues
    u₀ = y[:,1]
    tspan = ( time_all[1], time_all[end] )

    # initial weights
    θ₀ = initial_params(UA)

    #### set up the UDE problem and training of the neural network
    prob_nn = ODEProblem(UDE, u₀, tspan, θ₀)

    predict(θ) = Array(concrete_solve(prob_nn, Tsit5(), u₀, θ, saveat=time_all)) 

    sol = predict(weights)

    return sol

    ### train the UDE with ADAM
end

println("$(Threads.nthreads()) RUNNING")

all_runs = [ (station, run_id) for (station, run_id) in Iterators.product(1:6, 1:25)][:]

all_results = Dict()

Threads.@threads for id in eachindex(all_runs)

    station_id = all_runs[id][1]
    run_id = all_runs[id][2]

    println("Starting run $(id) for station: $(Statname[station_id]) and repeat run $(run_id)")

    res, callback, time, ret_code = run_experiment(station_id, run_id)


    rpt_counter = 0
    while ~ret_code
        rpt_counter += 1
        println("Rerunning exoeriment for $(Statname[station_id]) on run $(rund_id). Rerun Nr. $(rpt_counter)")
        _, _, ret_code = run_experiment(station_id, run_id)
    end

    all_results[(station_id, run_id)] = (res, callback, time)
end

for (station_id, run_id) in keys(all_results)

    datapath = "DataAllStat/" * DFname[station_id] * string(run_id) * ".h5"

    save_to_path(datapath, 
                 all_results[(station_id,run_id)][2].losses,
                 all_results[(station_id,run_id)][2].predictions[end],
                 all_results[(station_id,run_id)][3],
                 all_results[(station_id,run_id)][1].minimizer)

end
