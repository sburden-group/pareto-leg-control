module Handshake 
    using ForwardDiff
    using LinearAlgebra
    using ..Model
    using ..Designs

    # This is a rough approximation, a more accurate model would 
    # compute the kinetic energy in all links, and then M would be given from
    # the hessian of the kinetic energy metric (being configuration dependent)
    const load_m = .1;
    const M = diagm([Model.m_body,Model.Jm,Model.Jm,load_m,load_m])

    """ Behavior specific lagrangian quantities """
    # TODO: MUST ROTATE THE BODY 90 DEGREES!!!
    function potential_energy(q::Vector{T},p::Designs.Params) where T<:Real
        Model.potential_energy(q,p) + load_m*Model.g*q[Model.foot_idx[1]]
    end

    function potential_gradient(q::Vector{T},p::Designs.Params) where T<:Real
        f = q->potential_energy(q,p)
        cfg = ForwardDiff.GradientConfig(f,q,ForwardDiff.Chunk{Model.n_coords}())
        ForwardDiff.gradient(f,q,cfg)
    end

    """ Computes all constraints on behavior """
    function constraints(q::Vector{T},p::Designs.Params) where T<:Real
        [   Model.kin_constraints(q,p)
            q[Model.body_idx]
        ]
    end

    """ Computes jacobian of constraints """
    function constraints_jac(q::Vector{T},p::Designs.Params) where T<:Real
        Array{T}(vcat(Model.kin_constraints_jac(q,p),
            [1.0 0. 0. 0. 0.]
        ))
    end

    """ Computes hessian tensor of constraints """
    function constraints_hess(q::Vector{T},p::Designs.Params) where T<:Real
        Array{T}(vcat(Model.kin_constraints_hess(q,p),
            zeros(Model.n_coords,Model.n_coords)))
    end

    """ Computes the stance dynamics of system given a triple (q,qdot,u) """
    function dynamics(q::Vector{T},qdot::Vector{T},u::Vector{T},p::Designs.Params) where T<:Real
        ∇V = potential_gradient(q,p)
        DA = constraints_jac(q,p)
        m = size(DA,1) # number of constraints
        ddtDA = reshape(constraints_hess(q,p)*qdot,size(DA))
        # solve system of linear equations Ax=b where x = (qddot, λ)
        A = vcat(   hcat(M,DA'),
                    hcat(DA,zeros(m,m))
        )
        b = vcat(
            -∇V+Model.G*u,
            -ddtDA*qdot
        )
        x = A\b 
        qddot = x[1:Model.n_coords]
        λ = x[Model.n_coords+1:end]
        return qddot,λ
    end

    """ Computes projection from Model coordinates to Template coordinates """
    # does this even match the optimization code anymore??
    function anchor_projection(q::Vector{T},p::Designs.Params) where T<:Real
        r = p.l1+p.l2
        θ = -pi/8
        eq = [r*sin(θ),-r*cos(θ)]
        return q[Model.foot_idx] - eq
    end

    """ Computes projection from Model coordinates to Template coordinates """
    function anchor_pushforward()
        return [0. 0. 0. 1. 0.
                0. 0. 0. 0. 1.]
    end
    
    """
    Computes the template dynamics at the projection of (q,qdot)
    """
    function template_dynamics(q::Vector{T},qdot::Vector{T}) where T<:Real
        ω = 2pi
        ζ = 0.5
        return -2ζ*ω*qdot - ω^2 * q
    end

    """ Computes the Handshake controller """
    function control(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
        ∇V = potential_gradient(q,p)
        DA = constraints_jac(q,p)
        m = size(DA,1) # number of constraints
        ddtDA = reshape(constraints_hess(q,p)*qdot,size(DA))
        dπ = anchor_pushforward()
        f = template_dynamics(anchor_projection(q,p),dπ*qdot)
        A = vcat(   hcat(M,DA',-Model.G),
                    hcat(DA,zeros(m,m),zeros(m,size(Model.G,2))),
                    hcat(dπ, zeros(size(dπ,1),m), zeros(size(dπ,1),size(Model.G,2)))
        )
        b = vcat(
            -∇V,
            -ddtDA*qdot,
            f
        )
        (A\b)[end-size(Model.G,2)+1:end]
    end

    """ Computes the Handshake controller """
    function computed_torque(q::Vector{T},qdot::Vector{T},a::Vector{T},p::Designs.Params) where T<:Real
        ∇V = potential_gradient(q,p)
        DA = constraints_jac(q,p)
        m = size(DA,1) # number of constraints
        ddtDA = reshape(constraints_hess(q,p)*qdot,size(DA))
        dπ = anchor_pushforward()
        A = vcat(   hcat(M,DA',-Model.G),
                    hcat(DA,zeros(m,m),zeros(m,size(Model.G,2))),
                    hcat(dπ, zeros(size(dπ,1),m), zeros(size(dπ,1),size(Model.G,2)))
        )
        b = vcat(
            -∇V,
            -ddtDA*qdot,
            a
        )
        (A\b)[end-size(Model.G,2)+1:end]
    end

end