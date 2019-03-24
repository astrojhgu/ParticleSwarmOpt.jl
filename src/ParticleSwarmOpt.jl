module ParticleSwarmOpt

import Random

mutable struct Particle{V}
    position::V
    velocity::V
    fitness::eltype(V)
    pbest::Union{Missing, Particle{V}}
end

function Particle(position::V, velocity::V, fitness::eltype(V))::Particle{V} where {V}
    Particle{V}(position, velocity, fitness, missing)
end

zeros_like(v::Array) = zeros(eltype(v), size(v)...)

function Particle(f::Function, position::V)::Particle{V} where {V}
    Particle(position, zeros_like(position), f(position), missing)
end


mutable struct ParticleSwarmMaximizer{V}
    particle_count::Int
    ndim::Int
    swarm::Array{Particle{V}, 1}
    gbest::Union{Missing, Particle{V}}
    func::Function
end

function swarm_from_ensemble(func::Function, ensemble::AbstractArray{V,1})::Array{Particle{V}, 1} where {V}
    p(v) = Particle(func, v)
    map(p, ensemble)
end


function ParticleSwarmMaximizer(func::Function, ensemble::AbstractArray{V,1}, guess::Union{Missing, V}) where {V}
    particle_count = length(ensemble)
    swarm = swarm_from_ensemble(func, ensemble)
    ndim = length(first(ensemble))
    gbest = if ismissing(guess) missing else Particle(func, guess) end
    ParticleSwarmMaximizer(particle_count, ndim, swarm, gbest, func)

end

function move(p::Particle{V}, velocity, func::Function) where {V}
    newp = p.position + velocity
    p.position = newp
    p.velocity = velocity
    p.fitness = func(newp)
end

function sample(psm::ParticleSwarmMaximizer{V}, w, c1, c2, rng::Random.AbstractRNG=Random.GLOBAL_RNG) where {V}
    T = eltype(V)
    for p in psm.swarm
        if ismissing(psm.gbest)
            psm.gbest = Particle(copy(p.position), copy(p.velocity), copy(p.fitness), missing)
        elseif psm.gbest.fitness < p.fitness
            psm.gbest.position = copy(p.position)
            psm.gbest.velocity = copy(p.velocity)
            psm.gbest.fitness = copy(p.fitness)
        end

        if ismissing(p.pbest)
            p.pbest = Particle(copy(p.position), copy(p.velocity), copy(p.fitness), missing)
        elseif p.pbest.fitness < p.fitness
            p.pbest = Particle(copy(p.position), copy(p.velocity), copy(p.fitness), missing)
        end
    end

    for p in psm.swarm
        if !ismissing(psm.gbest) && !ismissing(p.pbest)
            part_vel = w*p.velocity
            cog_vel = c1*Random.rand(rng, T, psm.ndim).*(p.pbest.position - p.position)
            soc_vel = c2*Random.rand(rng, T, psm.ndim).*(psm.gbest.position - p.position)
            move(p, part_vel + cog_vel + soc_vel, psm.func)
        end
    end
end

function converged_dfit(psm::ParticleSwarmMaximizer{V}, p, m)::Bool where {V}
    if ismissing(psm.gbest)
        false
    else
        best_sort = sort(map(p->p.fitness, psm.swarm); lt = !isless)
        i1 = Int(floor(psm.particle_count*p))
        best_mean = sum(best_sort[2:i1]) / (i1 - 1)
        abs(psm.gbest.fitness - best_mean) < m
    end
end

function converged_dspace(psm::ParticleSwarmMaximizer{V}, p, m)::Bool where {V}
    if ismissing(psm.gbest)
        false
    else
        sorted_swarm = sort(psm.swarm; lt = (p1, p2)->p1.fitness > p2.fitness)
        i1 = Int(floor(psm.particle_count*p))
        max_norm = maximum(map(p->sum((psm.gbest.position - p.position).^2), sorted_swarm[1:i1]))
        max_norm < m
    end
end

function converged(psm::ParticleSwarmMaximizer{V}, p, m1, m2)::Bool where {V}
    converged_dfit(psm, p, m1) && converged_dspace(psm, p, m2)
end

end # module
