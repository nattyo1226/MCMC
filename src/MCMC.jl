module MCMC

using Random
include("union_find.jl")

export Params, montecarlo, calc_autocorrelation, calc_integrated_correlation_times

struct Params
    size::Tuple{Int64, Int64}
    J::Float64
    h::Float64
    T::Float64
end

function init_config(
    rng::AbstractRNG,
    size::Tuple{Int64, Int64},
)::Array{Int64, 2}
    return rand(rng, [-1, 1], size)
end

function calc_Si(
    params::Params,
    config::Array{Int64, 2},
    index::Tuple{Int64, Int64},
)::Int64
    size = params.size;
    indices_nearest = [
        (
            mod1(index[1] - 1, size[1]),
            index[2],
        ),
        (
            mod1(index[1] + 1, size[1]),
            index[2],
        ),
        (
            index[1],
            mod1(index[2] - 1, size[2]),
        ),
        (
            index[1],
            mod1(index[2] + 1, size[2]),
        ),
    ]

    return sum(config[ind...] for ind in indices_nearest)
end

function calc_energy(
    params::Params,
    config::Array{Int64, 2},
)::Float64
    size = params.size
    J = params.J
    h = params.h

    return sum(
        (-J * calc_Si(params, config, (ind_x, ind_y)) / 2 - h) * config[ind_x, ind_y] for
        ind_x in 1:size[1], ind_y in 1:size[2]
    )
end

function calc_energy_delta(
    params::Params,
    config::Array{Int64, 2},
    index::Tuple{Int64, Int64},
)::Float64
    J = params.J
    h = params.h

    Si = calc_Si(params, config, index)
    return 2 * (J * Si + h) * config[index...]
end

function calc_magnetization(
    config::Array{Int64, 2},
)::Float64
    return sum(config) / length(config)
end

function update_metropolis!(
    rng::AbstractRNG,
    params::Params,
    config::Array{Int64, 2},
    index::Tuple{Int64, Int64},
)
    T = params.T
    dE = calc_energy_delta(params, config, index)
    if rand(rng) <= exp(-dE / T)
        config[index...] *= -1
    end
end

function update_heatbath!(
    rng::AbstractRNG,
    params::Params,
    config::Array{Int64, 2},
    index::Tuple{Int64, Int64},
)
    T = params.T
    dE = calc_energy_delta(params, config, index)
    if rand(rng) <= exp(-dE / T) / (1 + exp(-dE / T))
        config[index...] *= -1
    end
end

function connect_bonds!(
    rng::AbstractRNG,
    params::Params,
    config::Array{Int64, 2},
    cluster::UnionFind,
    index_1::Tuple{Int64, Int64},
    index_2::Tuple{Int64, Int64},
)
    J = params.J
    T = params.T
    if (config[index_1...] * config[index_2...] > 0) && (rand(rng) <= 1 - exp(-2 * J / T))
        size = params.size
        ind_1 = index_1[1] + (index_1[2] - 1) * size[2]
        ind_2 = index_2[1] + (index_2[2] - 1) * size[2]
        unite!(cluster, ind_1, ind_2)
    end
end

function update_swendsen_wang!(
    rng::AbstractRNG,
    params::Params,
    config::Array{Int64, 2},
)
    if params.h != 0
        throw(
            ArgumentError(
                "Swendsen-Wang algorithm is valid where external magnetic field h = 0",
            ),
        )
    end

    size = params.size

    cluster = UnionFind(*(size...))
    flip = rand(rng, [-1, 1], size)

    for i in 1:size[1]
        for j in 1:size[2]
            connect_bonds!(
                rng,
                params,
                config,
                cluster,
                (i, j),
                (mod1(i + 1, size[1]), j),
            )
            connect_bonds!(
                rng,
                params,
                config,
                cluster,
                (i, j),
                (i, mod1(j + 1, size[2])),
            )
        end
    end

    for i in 1:(*(size...))
        root = get_root(cluster, i)
        config[i] = flip[root]
    end
end

function update_config!(
    rng::AbstractRNG,
    params::Params,
    config::Array{Int64, 2},
    method::String,
)
    if method == "metropolis"
        size = params.size
        for i in 1:size[1]
            for j in 1:size[2]
                update_metropolis!(rng, params, config, (i, j))
            end
        end
    elseif method == "heatbath"
        size = params.size
        for i in 1:size[1]
            for j in 1:size[2]
                update_heatbath!(rng, params, config, (i, j))
            end
        end
    elseif method == "swendsen_wang"
        update_swendsen_wang!(rng, params, config)
    else
        throw(ArgumentError("There is no method: $method"))
    end
end

function montecarlo(
    rng::AbstractRNG,
    params::Params,
    num_step_thermalization::Int64,
    num_step_measurement::Int64,
    interval_measurement::Int64,
    method::String,
)::Tuple{Array{Float64, 1}, Array{Float64, 1}}
    size = params.size
    config = init_config(rng, size)
    Es = Float64[]
    Ms = Float64[]

    for step in 1:(num_step_thermalization + num_step_measurement)
        update_config!(rng, params, config, method)

        if (step > num_step_thermalization) && (step % interval_measurement == 0)
            E = calc_energy(params, config) / *(size...)
            M = calc_magnetization(config)
            push!(Es, E)
            push!(Ms, M)
        end
    end

    return Es, Ms
end

function calc_correlation(
    obs_1::Array{Float64, 1},
    obs_2::Array{Float64, 1},
)
    dev_obs_1 = obs_1 .- sum(obs_1) / length(obs_1)
    dev_obs_2 = obs_2 .- sum(obs_2) / length(obs_2)

    return [
        sum(
            dev_obs_1[n] * dev_obs_2[n + k]
            for n in 1:length(dev_obs_1) - k
        ) / (length(dev_obs_1) - k)
        for k in 0:length(dev_obs_1) - 1
    ]
end

function calc_autocorrelation(
    obs::Array{Float64, 1},
)
    return calc_correlation(obs, obs)
end

function calc_integrated_correlation_times(
    obs::Array{Float64, 1},
)
    cor_obs = calc_autocorrelation(obs)
    return [
        sum(cor_obs[i] / cor_obs[1] for i in 1:n)
        for n in 1:length(cor_obs)
    ]
end

end
