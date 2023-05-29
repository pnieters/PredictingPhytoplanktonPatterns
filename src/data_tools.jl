""" CallbackLog{T}
Variably log {parameters, losses, predictions} of a UDE during training
"""
struct CallbackLog{T}
    log_params::Bool
    log_loss::Bool
    log_preds::Bool

    parameters::Vector{Vector{T}}
    losses::Vector{T}
    predictions::Vector{Array{T}}
end

function CallbackLog(;log_params=true, log_loss=true, log_preds=true, T=Any)
    CallbackLog(log_params,
                log_loss,
                log_preds,
                Vector{Vector{T}}(),
                Vector{T}(),
                Vector{Array{T}}())
end

function (cb::CallbackLog)(parameters, loss, prediciton)

    if cb.log_params
        push!(cb.parameters, copy(parameters))
    end

    if cb.log_loss
        push!(cb.losses, loss)
    end 

    if cb.log_preds
        push!(cb.predictions, prediciton)
    end 

    if any(isnan.(loss))
        throw(DomainError(NaN, "NaN in losses"))
    elseif any(isinf.(loss))
        throw(DomainError(Inf, "Inf in losses"))
    end 

    return false
end

""" dataframe_from_hdf5(hdf5path, gropups; dtype)
Create a data frame from the dictionary resulting from directly loading 
Max's data and the calculated IVs
--> Expected Format: HDF5 File for each bloomtype
    [Year, Chl, T, DIN, DIP, totalN, totalP, sal, DoY, DayFromFirst]
--> also reads calculated initial values
--> Does no longer use date types, all data was pre-merged and only needs to be read!
"""
function data_matrix_from_hdf5(hdf5_path_data,
                               hdf5_path_IV,
                               used_variable_idx; 
                               data_group_name="Statallsort",
                               IV_group_name="StartV",
                               dtype=Float64)


    _data = h5open(hdf5_path_data, "r") do file
        read(file, data_group_name)
    end

    iv_data =h5open(hdf5_path_IV, "r") do file
        read(file, IV_group_name)
    end

    iv_values = [iv_data[id, 1] + iv_data[id, 2] for id in used_variable_idx .- 1]

    final_values = [iv_data[id, 2] for id in used_variable_idx .- 1]

    data = hcat(
        iv_values,
        _data[:, used_variable_idx]', 
        final_values
    )
    time = [dtype(1), _data[:,9]..., dtype(365)]

    return time, data
end

""" dataframe_from_hdf5(hdf5path, gropups; dtype)
Create a data frame from the dictionary resulting from directly loading 
"OuTrBlstat14Zscore.h5
--> Expected Format: HDF5 File with groups for each station
hdf_file/station := SamplePoints x 10 variables 
                                 |
                                 |
    [Year, Chl, T, DIN, DIP, totalN, totalP, sal, DoY, DayFromFirst]
"""
function dataframe_from_hdf5(hdf5_path, groups; dtype=Float64)

    station = String[]
    date = Date[]

    chlorophyl = dtype[]
    temperature = dtype[]
    din = dtype[]
    dip = dtype[]
    totalN = dtype[]
    totalP = dtype[]
    salination = dtype[]

    for group in groups

        data = h5open(hdf5_path, "r") do file
            read(file, group)
        end

        push!(station, repeat([group], size(data,1))...)
        push!(date, [Date(data[idx, 1]) + Dates.Day(floor(Int,day))
                        for (idx, day) in enumerate(data[:,9])]...)
        push!(chlorophyl, data[:,2]...)
        push!(temperature, data[:,3]...)
        push!(din, data[:,4]...)
        push!(dip, data[:,5]...)
        push!(totalN, data[:,6]...)
        push!(totalP, data[:,7]...)
        push!(salination, data[:,8]...)
    end

    return sort!(DataFrame("Station" => station,
                           "Date" => date,
                           "Chlorophyl" => chlorophyl,
                           "Temp" => temperature,
                           "DIN" => din,
                           "DIP" => dip,
                           "TotalN" => totalN,
                           "TotalP" => totalP,
                           "Sal" => salination),
                [:Station, :Date]
                )

end

""" select_timeseries(df, first_day, last_day, station)
select a timeseries from a data frame, from first to last day, at a particular station
returns Matrix of samples x (Day of Sample, Temp, DIN, DIP, TotalN, TotalP, Sal)
"""
function select_timeseries(df, first_day, last_day, station)   

    rows = df[(df.Station .== station) .& (first_day .< df.Date .< last_day), 
              Not(:Station)]

    samples = Matrix(rows[!, Not(:Date)])
    t = Float64.([(date - rows[1,:Date]).value for date in rows[!,:Date]])

    return t, samples

end

""" DataPoint{VT, TT}
Each datapoint for UDE training is one initival value, and a number of sampled datapoints.
"""
struct DataPoint{VT, TT}
    initial_value::Vector{VT}

    sample_times::Vector{TT}
    measured_samples::Matrix{VT}
end

#TODO: Constructor can verify that dimensions of vectors and matrix fit!

""" create_stacked_data(df)
Create an iterator over DataPoints that stacks data from all years for each
station, shows station data sequentially as specificed by order (full station name)
for a specified number of iterations.
"""
function stack_data_by_year(df, order, repititions)
    data = []

    for station in order
        _df = df[df.Station .== station,:]
        _df.DoY = [_day_of_year(date) for date in _df[!,:Date]]
        sort!(_df, :DoY)
        # samples = Matrix(_df[!, [:Chlorophyl, :Temp, :DIN, :DIP, :TotalN, :TotalP, :Sal]])
        samples = permutedims(Matrix(
                    _df[!, [:Chlorophyl, :Temp, :DIN, :DIP, :TotalN, :TotalP]]
                  ), [2,1])

        dp = DataPoint(
            samples[:,1],
            _day_of_year_offset!(Float64.(_df[!,:DoY])),
            samples
        )

        push!(data, Base.Iterators.repeat([dp], repititions))
    end

    return vcat(data...)
end

function stack_data_by_year_and_station(df, repititions)
    _df = copy(df)

    _df.DoY = [_day_of_year(date) for date in _df[!,:Date]]
    sort!(_df, :DoY)
    samples = permutedims(Matrix(
                _df[!, [:Chlorophyl, :Temp, :DIN, :DIP, :TotalN, :TotalP]]
                ), [2,1])
    dp = DataPoint(
        samples[:,1],
        _day_of_year_offset!(Float64.(_df[!,:DoY])),
        samples
    )
    return Base.Iterators.repeat([dp], repititions)
end


@inline _day_of_year(date) = (date - Date(year(date))).value
function _day_of_year_offset!(v)

    @assert issorted(v)

    n_unique_elements = length(unique(v))
    while n_unique_elements !== length(v)
        for idx in 2:length(v)
            if v[idx] == v[idx-1]
                v[idx] += 0.1
            end
        end
        sort!(v)
        n_unique_elements = length(unique(v))
    end

    return v
end