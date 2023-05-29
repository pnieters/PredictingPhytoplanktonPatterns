using HDF5
using Plots
using LinearAlgebra
using Statistics
using AlgaeBlooming
using Flux
using DifferentialEquations 
using DiffEqFlux

using DataDrivenDiffEq
using ModelingToolkit
using DataDrivenSparse 

# define file names
Statname =  ["OuTrBlstatallMaxdata.h5",
             "SuBlstatallMaxdata.h5",
             "InTrBlstatallMaxdata.h5",
             "InDuBlstatallMaxdata.h5",
             "DeSpBlstatallMaxdata.h5",
             "AdBlEWstatallMaxdata.h5"]

Startname =  ["OuTrBlstartMaxdata.h5",
              "SuBlstartallMaxdata.h5",
              "InTrBlstartallMaxdata.h5",
              "InDuBlstartallMaxdata.h5",
              "DeSpBlstartallMaxdata.h5",
              "AdBlEWstartallMaxdata.h5"]
FFname = ["trainOuTrBloT.png",
          "trainSuBloT.png",
          "trainInTrBloT.png",
          "trainInDuBloT.png",
          "trainDeSpBloT.png",
          "trainAdBlEWoT.png"]
SFname = ["SindyOuTrBloT.png",
          "SindySuBloT.png",
          "SindyInTrBloT.png",
          "SindyInDuBloT.png",
          "SindyDeSpBloT.png",
          "SindyAdBlEWoT.png"]
TXTname = ["ODEOuTrBloT.txt",
           "ODESuBloT.txt",
           "ODEInTrBloT.txt",
           "ODEInDuBloT.txt",
           "ODEDeSpBloT.txt",
           "ODEAdBlEWoT.txt"]

# define blooming type names for plotting and reading
DFname = ["trainOuTrBloT",
          "trainSuBloT",
          "trainInTrBloT",
          "trainInDuBloT",
          "trainDeSpBloT",
          "trainAdBlEWoT"]
Oname = ["OuTrBl",
         "SuBl",
         "InTrBl",
         "InDuBl",
         "DeSpBl",
         "AdBl"]
Dname = ["data OuTrBl",
         "data SuBl",
         "data InTrBl",
         "data InDuBl",
         "data DeSpBl",
         "data AdBlEW"]


for idx = 1:6 # loop blooming tpyes

    # load solutions
    Sol = []
    SolW = []
    for idxrun = 1:20   # loop training runs
        
        datapath = "DataallStat\\" * Oname[idx] * "\\" * DFname[idx] * string(idxrun) * ".h5" 
        # read solution
        H = h5open(datapath, "r") do file 
            read(file, "Sol")
        end
        push!(Sol,H)

        # read weights
        H2 = h5open(datapath, "r") do file
            read(file,  "WeightsFin")
        end
        push!(SolW,H2)
    end

    # read time (same for all training runs)
    idxrun = 1
    datapath =  "DataallStat\\" * Oname[idx] * "\\" * DFname[idx] * string(idxrun) * ".h5"
    T = h5open(datapath, "r") do file
        read(file, "time")
    end

    # training data load
    Chl_all =[]
    DIN_all = []
    DIP_all = []
    totalN_all = []
    totalP_all = []
    time_all = []

    # load trainingdata
    # 1 .  2 . 3 .  4 . 5 . 6 .    7 .    8 . 9 . 10
    # year Chl Temp DIN DIP totalN totalP sal DoY Dayfromfirst

    Bdata = h5open(Statname[idx], "r") do file #h5open("OuTrBlstat14Zscore.h5", "r") do file
        read(file, "Statallsort")
    end #[itertaion,run]    # string anlegen mit "Stat$(k)sort"

    # load startvalue
    Sdata = h5open(Startname[idx], "r") do file #h5open("OuTrBlstartMaxdata.h5", "r") do file
        read(file, "StartV")
    end

    # here used variables: Chl DIN DIP totalN totalP
    push!(Chl_all, [Sdata[1,1]+Sdata[1,2]; Bdata[:,2]; Sdata[1,2]])
    push!(DIN_all, [Sdata[3,1]+Sdata[3,2]; Bdata[:,4]; Sdata[3,2]])
    push!(DIP_all, [Sdata[4,1]+Sdata[4,2]; Bdata[:,5]; Sdata[4,2]])
    push!(totalN_all, [Sdata[5,1]+Sdata[5,2]; Bdata[:,6]; Sdata[5,2]])
    push!(totalP_all, [Sdata[6,1]+Sdata[6,2]; Bdata[:,7]; Sdata[6,2]])
    push!(time_all, [1; Bdata[:,9]; 365])

    #######################
    # calculate Mean and STD
    mChl = []
    mDIN = []
    mDIP = []
    mtotN = []
    mtotP = []
    for k=1:size(Sol,1)#20 
        push!(mChl, Sol[k][1,:])
        push!(mDIN, Sol[k][2,:])
        push!(mDIP, Sol[k][3,:])
        push!(mtotN, Sol[k][4,:])
        push!(mtotP, Sol[k][5,:])
    end

    stdChl = (std(reduce(vcat,transpose.(mChl)),dims=1))'
    mChl = (mean(reduce(vcat,transpose.(mChl)),dims=1))'
    stdDIN = (std(reduce(vcat,transpose.(mDIN)),dims=1))'
    mDIN = (mean(reduce(vcat,transpose.(mDIN)),dims=1))'
    stdDIP = (std(reduce(vcat,transpose.(mDIP)),dims=1))'
    mDIP = (mean(reduce(vcat,transpose.(mDIP)),dims=1))'
    stdtotN = (std(reduce(vcat,transpose.(mtotN)),dims=1))'
    mtotN = (mean(reduce(vcat,transpose.(mtotN)),dims=1))'
    stdtotP = (std(reduce(vcat,transpose.(mtotP)),dims=1))'
    mtotP = (mean(reduce(vcat,transpose.(mtotP)),dims=1))'

    
    #####################
    # MSE meanmodel data
    MSEchl = sum((mChl - Chl_all[1]).^2)/length(time_all[1])

    ### comparison meanmodel and single fit (MSE)
    dfDiff = []
    for k=1:size(Sol,1)#20
        push!(dfDiff,sum((Sol[k][1,:] - mChl).^2)/length(time_all[1]))
        push!(dfDiff,sum((Sol[k][2,:] - mDIN).^2)/length(time_all[1]))
        push!(dfDiff,sum((Sol[k][3,:] - mDIP).^2)/length(time_all[1]))
        push!(dfDiff,sum((Sol[k][4,:] - mtotN).^2)/length(time_all[1]))
        push!(dfDiff,sum((Sol[k][5,:] - mtotP).^2)/length(time_all[1]))
    end
    
    dfDiff = reshape(dfDiff,5,size(Sol,1))

    # find smallest mean MSE and index to use their weights for SInDy
    Sidx = findmin(mean(dfDiff,dims=1))

    #################
    # calculate mean model based on 365 days
    # ANN setup
    UA = FastChain(
        FastDense( 5 , 16, gelu, initW=Flux.glorot_normal), #gelu relu
        FastDense(16 , 16, gelu, initW=Flux.glorot_normal),
        FastDense(16 , 16, gelu, initW=Flux.glorot_normal),
        FastDense(16 , 16, gelu, initW=Flux.glorot_normal),
        FastDense(16 ,  5, gelu, initW=Flux.glorot_normal),
    )

    # keep in this file --> we're going for specific Eqs as part of the exploration
    function UniDiffEq(u,p,t)
        z = UA(u,p)
        [z[1],
        z[2],
        z[3],
        z[4],
        z[5]]
    end

    # use UA to recover solution on 365-day timeframe
    # initialization startvalues (Chl DIN DIP totalN totalP)
    u₀ = Float64[Chl_all[1][1], DIN_all[1][1], DIP_all[1][1], totalN_all[1][1], totalP_all[1][1]]
    tspan = (1, 365)

    # Weights initialization
    θ₀ =  initial_params(UA) 

    prob_nn = ODEProblem(UniDiffEq, u₀, tspan, θ₀)
    predict(θ) = Array(concrete_solve(prob_nn, Tsit5(), u₀, θ, saveat=1))
    SolH365 = []
    for ggg=1:size(Sol,1)#20
        println("run $(ggg)")
        push!(SolH365,predict(SolW[ggg]))
    end

    # calculate Mean and STD based on 365 day data
    mChl365 = []
    for k=1:size(SolH365,1)#20 
        push!(mChl365, SolH365[k][1,:])
    end

    stdChl365 = (std(reduce(vcat,transpose.(mChl365)),dims=1))'
    mChl365 = (mean(reduce(vcat,transpose.(mChl365)),dims=1))'

    ################################ 
    # PLOTS
    ###############################
    # CHLa
    fig2 =  plot(time_all[1],Chl_all[1], seriestype = :scatter, markerstrokewidth=0, markersize=2, ma=0.5, label=Dname[idx])
    for k=1:size(Sol,1)#20
        plot!(fig2,T,Sol[k][1,:], legend=:false, title=Dname[idx])
    end
    plot!(fig2,T, mChl, lw=3, seriescolor=:red)
    plot!(fig2,T, mChl .- stdChl, lw=3, seriescolor=:red,linestyle=:dash)
    plot!(fig2,T, mChl .+ stdChl, lw=3, seriescolor=:red,linestyle=:dash)
    xlims!(1,365)
    ylims!(-3.5,3.5)
    xlabel!(fig2, "day of the year")
    ylabel!(fig2, "Chl-a")

    # DIN
    fig3 =  plot(time_all[1],DIN_all[1], seriestype = :scatter, markerstrokewidth=0, markersize=2, ma=0.5, label=Dname[idx])
    for k=1:size(Sol,1)
        plot!(fig3,T,Sol[k][2,:], legend=:false)
    end
    plot!(fig3,T, mDIN, lw=3, seriescolor=:red)
    plot!(fig3,T, mDIN .- stdDIN, lw=3, seriescolor=:red,linestyle=:dash)
    plot!(fig3,T, mDIN .+ stdDIN, lw=3, seriescolor=:red,linestyle=:dash)
    xlims!(1,365)
    ylims!(-3.5,3.5)
    xlabel!(fig3, "day of the year")
    ylabel!(fig3, "DIN")

    # DIP
    fig4 =  plot(time_all[1],DIP_all[1], seriestype = :scatter, markerstrokewidth=0, markersize=2, ma=0.5, label=Dname[idx])
    for k=1:size(Sol,1)
        plot!(fig4,T,Sol[k][3,:], legend=:false)
    end
    plot!(fig4,T, mDIP, lw=3, seriescolor=:red)
    plot!(fig4,T, mDIP .- stdDIP, lw=3, seriescolor=:red,linestyle=:dash)
    plot!(fig4,T, mDIP .+ stdDIP, lw=3, seriescolor=:red,linestyle=:dash)
    xlims!(1,365)
    ylims!(-3.5,3.5)
    xlabel!(fig4, "day of the year")
    ylabel!(fig4, "DIP")

    # totN
    fig5 =  plot(time_all[1],totalN_all[1], seriestype = :scatter, markerstrokewidth=0, markersize=2, ma=0.5, label=Dname[idx])
    for k=1:size(Sol,1)
        plot!(fig5,T,Sol[k][4,:], legend=:false)
    end
    plot!(fig5,T, mtotN, lw=3, seriescolor=:red)
    plot!(fig5,T, mtotN .- stdtotN, lw=3, seriescolor=:red,linestyle=:dash)
    plot!(fig5,T, mtotN .+ stdtotN, lw=3, seriescolor=:red,linestyle=:dash)
    xlims!(1,365)
    ylims!(-3.5,3.5)
    xlabel!(fig5, "day of the year")
    ylabel!(fig5, "totalN")

    # totP
    fig6 =  plot(time_all[1],totalP_all[1], seriestype = :scatter, markerstrokewidth=0, markersize=2, ma=0.5, label=Dname[idx])
    for k=1:size(Sol,1)
        plot!(fig6,T,Sol[k][5,:], legend=:false)
    end
    plot!(fig6,T, mtotP, lw=3, seriescolor=:red)
    plot!(fig6,T, mtotP .- stdtotP, lw=3, seriescolor=:red,linestyle=:dash)
    plot!(fig6,T, mtotP .+ stdtotP, lw=3, seriescolor=:red,linestyle=:dash)
    xlims!(1,365)
    ylims!(-3.5,3.5)
    xlabel!(fig6, "day of the year")
    ylabel!(fig6, "totalP")

    K=plot(fig2, fig3, fig4, fig5, fig6, layout=(2,3), size=(1000,1000))
    savefig(K, FFname[idx])

    ###############################
    ## save mean model ############
    datapath = "DataallStat\\MeanM" * DFname[idx] * "_new.h5"
            h5open(datapath,"w") do file
                write(file, "mChl", mChl)
                write(file, "time", T)
                write(file, "stdChl", stdChl)
                write(file, "mChl365", mChl365)
                write(file, "stdChl365", stdChl365)
                write(file, "MSEchl", MSEchl)
                write(file, "SindyIdx", Sidx[2][2])
                write(file, "SindyIdxVal", Sidx[1][1])
            end
    #######################################

########################################################################################
    # SINDY
    predict2(θ) = Array(concrete_solve(prob_nn, Tsit5(), u₀, θ, saveat= 1))  # solution based on 365-days 
    SindySol = predict2(SolW[Sidx[2][2]])          # values
    dSindySol = UA(SindySol, SolW[Sidx[2][2]])     # derivatives

    # DEFINE basis
    @variables u[1:5] t # u = [Chl; DIN; DIP; totN; totP]
    # Generate the basis functions, multivariate polynomials up to deg 2
    # and sine and cosine
    bf = [polynomial_basis(u, 2); sin.(u); cos.(u)]
    basis = Basis(bf, u, iv = t)

    # DEFINE datadriven problem
    @time ddprob = ContinuousDataDrivenProblem(SindySol, 1:365, dSindySol)

    # solve datadriven problem (SINDY)
    λ = collect(1e-3:1e-5:1)
    opt = STLSQ(λ)
    @time ddsol = solve(ddprob, basis, opt, options = DataDrivenCommonOptions(digits = 4, maxiters=1000, denoise=:true))
    
    Sindyeqs = get_basis(ddsol)                     # eqs-basis
    Sindyparams = get_parameter_map(Sindyeqs)       # parameters
    dSindyGuess = ddsol.basis(get_problem(ddsol))   # derivaties based on eqs-basis
    SindyCoeff = ddsol.out[1].coefficients          # coefficient matrix

    #print solution to REPL
    println(ddsol)
    println(Sindyeqs)
    println(Sindyparams)
    plot(plot(ddprob),plot(ddsol), layout=(1,2))

    ######################################################################
    #print solution to txt file
    open(TXTname[idx], "w") do f
        println(f, ddsol)                           # info on solution
        println(f,Sindyeqs)                         # eqs-basis
        println(f,Sindyparams)                      # parameters
        println(f, "SindyIdx $(Sidx[2][2])")        # index of chosen run for Sindy
        println(f, "SindyIdxVal $(Sidx[1][1])")     # mean MSE of chosen run for Sindy
    end

    # write solution and coefficient matrix to .h5 file 
    datapath = "DataallStat\\Sindy" * DFname[idx] * "_HQ.h5"
            h5open(datapath,"w") do file
                write(file, "dSindygGuess", dSindyGuess)    # Sindy derivativ guess
                write(file, "dSindySol", dSindySol)         # ANN derivative guess
                write(file, "SindyCoeff", SindyCoeff)       # bais function coefficient matrix
                write(file, "SindyIdx", Sidx[2][2])         # index of chosen run for Sindy
                write(file, "SindyIdxVal", Sidx[1][1])      # mean MSE of chosen run for Sindy
            end
    ###############################################################################################

    ################################ 
    # PLOTS
    ###############################
    # comparison mena model and chosen run for Sindy

    k=Sidx[2][2]
    fig2 = plot(T, mChl, lw=3, seriescolor=:red, label="365-MeanModel")
    plot!(fig2,T,Sol[k][1,:], seriescolor=:blue, label="bestANNguess", title=Dname[idx])
    #plot!(fig2, ddprob.t, ddprob.X[1,:], seriescolor=:green, label="DataDrivenProb", linestyle=:dash)
    xlims!(1,365)
    ylims!(-3.5,3.5)
    xlabel!(fig2, "day of the year")
    ylabel!(fig2, "Chl-a")

    fig3 = plot(T, mDIN, lw=3, seriescolor=:red,label="365-MeanModel")
    plot!(fig3,T,Sol[k][2,:], seriescolor=:blue, label="bestANNguess")
    #plot!(fig3, ddprob.t, ddprob.X[2,:], seriescolor=:green, label="DataDrivenProb", linestyle=:dash)
    xlims!(1,365)
    ylims!(-3.5,3.5)
    xlabel!(fig3, "day of the year")
    ylabel!(fig3, "DIN")


    fig4 = plot(T, mDIP, lw=3, seriescolor=:red, label="365-MeanModel")
    plot!(fig4,T,Sol[k][3,:], seriescolor=:blue, label="bestANNguess")
    #plot!(fig4, ddprob.t, ddprob.X[3,:], seriescolor=:green, label="DataDrivenProb", linestyle=:dash)
    xlims!(1,365)
    ylims!(-3.5,3.5)
    xlabel!(fig4, "day of the year")
    ylabel!(fig4, "DIP")


    fig5 = plot(T, mtotN, lw=3, seriescolor=:red, label="365-MeanModel")
    plot!(fig5,T,Sol[k][4,:], seriescolor=:blue, label="bestANNguess")
    #plot!(fig5, ddprob.t, ddprob.X[4,:], seriescolor=:green, label="DataDrivenProb", linestyle=:dash)
    xlims!(1,365)
    ylims!(-3.5,3.5)
    xlabel!(fig5, "day of the year")
    ylabel!(fig5, "totalN")

    fig6 = plot(T, mtotP, lw=3, seriescolor=:red, label="365-MeanModel")
    plot!(fig6,T,Sol[k][5,:], seriescolor=:blue, label="bestANNguess")
    #plot!(fig6, ddprob.t, ddprob.X[5,:], seriescolor=:green, label="DataDrivenProb", linestyle=:dash)
    xlims!(1,365)
    ylims!(-3.5,3.5)
    xlabel!(fig6, "day of the year")
    ylabel!(fig6, "totalP")

    K=plot(fig2, fig3, fig4, fig5, fig6, layout=(2,3), size=(1000,1000))
    savefig(K, SFname[idx])

end