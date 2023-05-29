""" polar_loss(predict_fn, parameters, measured_dynamics; weights=[0.5, 0.5])
calculate the polar loss weights[1] * normed length distance +
                         weights[2] * cosine distance
between the predicted dynamics of predict_fn with weights parameters and
the originally measured dynamics
"""
function polar_loss(predict_fn, parameters, measured_dynamics; weights=[0.5, 0.5])
    predicted_dynamics = predict_fn(parameters) 
    n_samples = size(predicted_dynamics, 2)
    loss = sum(weights[1] * [_normed_ld(measured_dynamics[:,i], 
                                       predicted_dynamics[:,i]) for i in 1:n_samples] .+
               weights[2] * [_cos_distance(measured_dynamics[:,i], 
                                           predicted_dynamics[:,i]) for i in 1:n_samples]
              )
    return loss, predicted_dynamics
end

function mse_loss(θ, y, predict)
    ŷ = predict(θ)
    loss = sum(abs2, y .- ŷ), ŷ
end

@inline _cos_similarity(a, b) = dot(a,b) / (norm(a) * norm(b))
@inline _cos_distance(a,b) = (1 - _cos_similarity(a,b)) / 2
@inline _normed_ld(a,b) = abs(norm(a) - norm(b)) / (norm(a) + norm(b))

""" train_ude(loss_fn, initial_parameters, options; callback, maxiters)

Optimises a UDE model as measured by loss from intial_parameters.
"""
function train_ude(loss, 
                   model, 
                   solver, 
                   initial_parameters, 
                   training_data, 
                   optimiser; 
                   cb=(args...)->(), 
                   maxiters=1)

    params = copy(initial_parameters)
    f_params = Flux.params(params)

    for (step, d) in enumerate(training_data)

        (;initial_value, sample_times, measured_samples) = d
        tspan = (first(sample_times), last(sample_times))

        # problem = ODEProblem(model, iv, tspan, initial_parameters)
        problem = ODEProblem(model, initial_value, tspan, params)

        predict_fn(parameters) = Array(concrete_solve(problem, 
                                                      solver, 
                                                      initial_value, 
                                                      parameters, 
                                                      saveat=sample_times))

        loss_fn(parameters) = loss(predict_fn, parameters, measured_samples)

        try
            local x
            gs = gradient(f_params) do
                x = loss_fn(params) 
                first(x) # first is loss, second is prediction
            end
            Flux.Optimise.update!(optimiser, f_params, gs)
            cb(params, x...)
        catch ex
            if ex isa DomainError
                println(ex)
                break
            else
                rethrow(ex)
            end
        end

        if step >= maxiters
            break
        end

        println("Finished step \t $step.")

    end

    return params

end