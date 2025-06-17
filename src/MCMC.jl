module MCMC

using Match
using Random
include("UF.jl")

struct Settings
    T::Float64
end

struct Ising2DParams
    size::Tuple{Int64, Int64}
    J::Float64
    h::Float64
end

@enum MethodUpdate::UInt8 begin
    metropolis
    heatbath
    swendsen_wang
end

function init_spins_2D(
    rng::AbstractRNG,
    size::Tuple{Int64, Int64},
)::Array{Int64, 2}
    return rand(rng, [-1, 1], size)
end

function calc_Si(
    params::Ising2DParams,
    config::Array{Int64, 2},
    index::Tuple{Int64, Int64},
)::Int64
    size = params.size;
    indices_nearest = [
        (
            mod(index[1] - 1, size[1]),
            index[2],
        ),
        (
            mod(index[1], size[1]) + 1,
            index[2],
        ),
        (
            index[1],
            mod(index[2] - 1, size[2]),
        ),
        (
            index[1],
            mod(index[2], size[2]) + 1,
        ),
    ]

    return sum(config(ind...) for ind in indices_nearest)
end

function calc_energy(
    params::Ising2DParams,
    config::Array{Int64, 2},
)::Float64
    size = params.size
    J = params.J
    h = params.h

    return sum((-J * calc_Si(config, (ind_x, ind_y), params) / 2 - h) * config[ind_x, ind_y] for ind_x in 1:size[1], ind_y in 1:size[2])
end

function calc_energy_delta(
    params::Ising2DParams,
    config::Array{Int64, 2},
    index::Tuple{Int64, Int64},
)::Float64
    J = params.J
    h = params.h

    Si = calc_Si(config, index, params)
    return 2 * (J * Si + h) * config[index...]
end

function update_metropolis!(
    rng::AbstractRNG,
    params::Ising2DParams,
    settings::Settings,
    config::Array{Int64, 2},
    index::Tuple{Int64, Int64},
)
    T = settings.T
    dE = calc_energy_delta(config, index, params)
    if rand(rng) <= exp(-dE / T)
        config[index...] *= -1
    end
end

function update_heatbath!(
    rng::AbstractRNG,
    params::Ising2DParams,
    settings::Settings,
    config::Array{Int64, 2},
    index::Tuple{Int64, Int64},
)
    T = settings.T
    dE = calc_energy_delta(config, index, params)
    if rand(rng) <= exp(-dE / T) / (1 + exp(-dE / T))
        config[index...] *= -1
    end
end

function connect_bonds!(
    rng::AbstractRNG,
    params::Ising2DParams,
    settings::Settings,
    config::Array{Int64, 2},
    cluster::UF.UnionFind,
    index_1::Tuple{Int64, Int64},
    index_2::Tuple{Int64, Int64},
)
    J = params.J
    T = settings.T
    if (config[index_1...] * config[index_2...] > 0) && (rand(rng) <= e^(-2 * J / T))
        size = params.size
        ind_1 = index_1[1] + (index_1[2] - 1) * size[2]
        ind_2 = index_2[1] + (index_2[2] - 1) * size[2]
        UF.unite!(cluster, ind_1, ind_2)
    end
end

function update_swendsen_wang!(
    rng::AbstractRNG,
    params::Ising2DParams,
    settings::Settings,
    config::Array{Int64, 2},
)
    if params.h != 0
        throw(ArgumentError("Swendsen-Wang algorithm is valid where external magnetic filed h = 0"))
    end

    size = params.size
    
    cluster = UF.UnionFind(*(size...))
    flip = rand(rng, [-1, 1], size)
    
    for i in 1:size[1]
        for j in 1:size[2]
            connect_bonds!(rng, params, settings, config, cluster, (i, j), (mod(i, size[1]) + 1, j))
            connect_bonds!(rng, params, settings, config, cluster, (i, j), (i, mod(j, size[2]) + 1))
        end
    end

    for i in 1:*(size...)
        root = UF.get_root(cluster, i)
        config[i] = flip[root]
    end
end

function update_config!(
    rng::AbstractRNG,
    params::Ising2DParams,
    settings::Settings,
    config::Array{Int64, 2},
    method::MethodUpdate,
)
    @match method begin
        MethodUpdate::metropolis => begin
            size = params.size
            for i in 1:size[1]
                for j in 1:size[2]
                    update_metropolis!(rng, config, (i, j), params, settings)
                end
            end
        end
        MethodUpdate::heatbath => begin
            size = params.size
            for i in 1:size[1]
                for j in 1:size[2]
                    update_heatbath!(rng, config, (i, j), params, settings)
                end
            end
        end
        MethodUpdate::swendsen_wang => begin
            update_swendsen_wang!(rng, config, params, settings)
        end
    end
end

end
