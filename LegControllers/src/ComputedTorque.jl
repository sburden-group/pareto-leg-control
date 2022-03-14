module ComputedTorque 
    using ForwardDiff
    using LinearAlgebra
    using ..Model
    using ..Designs

    # This is a rough approximation, a more accurate model would 
    # compute the kinetic energy in all links, and then M would be given from
    # the hessian of the kinetic energy metric (being configuration dependent)
    #const load_m = 1.0; # [Kg]. a lumped parameter that roughly captures leg inertia and gravitational potential
    const load_m = 1.1; # [Kg]. a lumped parameter that roughly captures leg inertia and gravitational potential
    const M = diagm([Model.m_body,Model.Jm,Model.Jm,load_m,load_m])

    """ Behavior specific lagrangian quantities """
    function potential_energy(q::Vector{T},p::Designs.Params) where T<:Real
        Model.potential_energy(q,p) + load_m*Model.g*q[Model.foot_idx[2]]
    end

    function potential_gradient(q::Vector{T},p::Designs.Params) where T<:Real
        f = q->potential_energy(q,p)
        cfg = ForwardDiff.GradientConfig(f,q,ForwardDiff.Chunk{Model.n_coords}())
        ForwardDiff.gradient(f,q,cfg)
    end

    """ Computes all constraints on behavior """
    function constraints(q::Vector{T},p::Designs.Params) where T<:Real
        Array{T}(vcat(Model.kin_constraints(q,p),
            q[Model.body_idx]
        ))
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

    """ Computes the Handshake controller """
    function control(q::Vector{T},qdot::Vector{T},p::Designs.Params,a::Vector{T}) where T<:Real
        ∇V = potential_gradient(q,p)
        DA = constraints_jac(q,p)
        m = size(DA,1) # number of constraints
        ddtDA = reshape(constraints_hess(q,p)*qdot,size(DA))
        dπ = [0. 0. 0. 1. 0.
                0. 0. 0. 0. 1.]
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

#dummy variables for triggering precompilation of functions
_q = [0.,pi/2,-pi/2,0.,0.]
_q[[4,5]] = -constraints(_q,Designs.default_params)[[1,2]]
_qdot = zeros(5)
_DA = constraints_jac(_q,Designs.default_params)
_DDA = constraints_hess(_q,Designs.default_params)
_u = control(_q,_qdot,Designs.default_params, [0.,0.])
_qddot = dynamics(_q,_qdot,_u,Designs.default_params)
end
