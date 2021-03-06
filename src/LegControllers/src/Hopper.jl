module Hopper
    using ForwardDiff
    using LinearAlgebra
    using ..Model
    using ..Designs

    """ Behavior specific lagrangian quantities """
    function potential_energy(q::Vector{T},p::Designs.Params) where T<:Real
        Model.potential_energy(q,p)
    end

    function potential_gradient(q::Vector{T},p::Designs.Params) where T<:Real
        f = q->potential_energy(q,p)
        cfg = ForwardDiff.GradientConfig(f,q,ForwardDiff.Chunk{Model.n_coords}())
        ForwardDiff.gradient(f,q,cfg)
    end

    # This is a rough approximation, a more accurate model would 
    # compute the kinetic energy in all links, and then M would be given from
    # the hessian of the kinetic energy metric (being configuration dependent)
    const M = diagm([Model.m_body,Model.Jm,Model.Jm,0.,0.])

    """ STANCE DYNAMICS CODE """

    """ Computes constraint that foot cannot move """
    function contact_constraints(q::Vector{T},foot_loc::Vector{Float64}) where T<:Real
        q[Model.foot_idx]-foot_loc
    end

    """ Computes all constraints on behavior """
    function stance_constraints(q::Vector{T},p::Designs.Params,foot_loc::Vector{Float64}) where T<:Real
        vcat(Model.kin_constraints(q,p),contact_constraints(q,foot_loc))
    end

    """ Computes jacobian of constraints """
    function stance_constraints_jac(q::Vector{T},p::Designs.Params) where T<:Real
        Array{T}(vcat(Model.kin_constraints_jac(q,p),
            [0 0 0 1.0 0
             0 0 0 0 1.0]
        ))
    end

    """ Computes fundamental tensor of constraints """
    function stance_constraints_hess(q::Vector{T},p::Designs.Params) where T<:Real
        Array{T}(vcat(
            Model.kin_constraints_hess(q,p),
            zeros(2*Model.n_coords,Model.n_coords)
        ))
    end
    

    """ Computes the stance dynamics of system given a triple (q,qdot,u) """
    function stance_dynamics(q::Vector{T},qdot::Vector{T},u::Vector{T},p::Designs.Params) where T<:Real
        ∇V = potential_gradient(q,p)
        DA = stance_constraints_jac(q,p)
        m = size(DA,1) # number of constraints
        ddtDA = reshape(stance_constraints_hess(q,p)*qdot,size(DA))
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


    """ Computes projection from Model coordinates to Template coordinates in stance mode """
    function stance_anchor_projection(q::Vector{T},p::Designs.Params) where T<:Real
        return [q[1]-p.l1-p.l2-Model.foot_offset]
    end

    """ Computes projection from Model coordinates to Template coordinates in stance mode """
    function stance_anchor_pushforward()
        return [1.0 0 0 0 0]
    end
    
    """
    Computes the template dynamics at the projection of (q,qdot) in stance mode
    """
    function stance_template_dynamics(q::Vector{T},qdot::Vector{T}) where T<:Real
        ω = 5pi
        ζ = 0.01
        return -2ζ*ω*qdot - ω^2 * q - [Model.g]
    end

    """ Computes the Hopper stance controller """
    function stance_control(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
        ∇V = potential_gradient(q,p)
        DA = stance_constraints_jac(q,p)
        m = size(DA,1) # number of constraints
        ddtDA = reshape(stance_constraints_hess(q,p)*qdot,size(DA))
        dπ = stance_anchor_pushforward()
        f = stance_template_dynamics(stance_anchor_projection(q,p),dπ*q)
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

    function stance_control(q::Vector{T},qdot::Vector{T},a::Vector{T},p::Designs.Params) where T<:Real
        ∇V = potential_gradient(q,p)
        DA = stance_constraints_jac(q,p)
        m = size(DA,1) # number of constraints
        ddtDA = reshape(stance_constraints_hess(q,p)*qdot,size(DA))
        dπ = stance_anchor_pushforward()
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

    """ FLIGHT DYNAMICS CODE """

    """ Computes constraints in flight (non-contact) mode """
    function flight_constraints(q::Vector{T},p::Designs.Params) where T<:Real
        Model.kin_constraints(q,p)
    end

    """ Computes jacobian of constraints in flight (non-contact) mode """
    function flight_constraints_jac(q::Vector{T},p::Designs.Params) where T<:Real
        Model.kin_constraints_jac(q,p)
    end

    """ Computes fundamental tensor of constraints in flight (non-contact) mode """
    function flight_constraints_hess(q::Vector{T},p::Designs.Params) where T<:Real
        Model.kin_constraints_hess(q,p)
    end


    function flight_dynamics(q::Vector{T},qdot::Vector{T},u::Vector{T},p::Designs.Params) where T<:Real
        ∇V = potential_gradient(q,p)
        DA = flight_constraints_jac(q,p)
        m = size(DA,1) # number of constraints
        ddtDA = reshape(flight_constraints_hess(q,p)*qdot,size(DA))
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

    const KNEE_MIN_DIST = 0.04
    """ Computes projection from Model coordinates to Template coordinates in flight mode """
    function flight_anchor_projection(q::Vector{T},p::Designs.Params) where T<:Real
        ϕ = (q[Model.θ1_idx]-q[Model.θ2_dx])/2
        return [q[4],p.l1*sin(ϕ)-KNEE_MIN_DIST]
    end

    """ Computes projection from Model coordinates to Template coordinates in flight mode """
    function flight_anchor_pushforward(q::Vector{T},p::Designs.Params) where T<:Real
        ϕ = (q[Model.θ1_idx]+q[Model.θ2_dx])/2
        return [0. 0. 0. 1.0 0.
                0. .5*cos(ϕ) .5*cos(ϕ) 0. 0.]
    end
    
    """
    Computes the template dynamics at the projection of (q,qdot) in flight mode
    """
    function flight_template_dynamics(q::Vector{T},qdot::Vector{T}) where T<:Real
        ω = 1.5*pi 
        ζ = 5.
        return -2ζ*ω*qdot - ω^2 * q 
    end

    """ Computes the Hopper flight controller """
    function flight_control(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
        ∇V = potential_gradient(q,p)
        DA = flight_constraints_jac(q,p)
        m = size(DA,1) # number of constraints
        ddtDA = reshape(flight_constraints_hess(q,p)*qdot,size(DA))
        dπ = flight_anchor_pushforward(q,p)
        f = flight_template_dynamics(flight_anchor_projection(q,p),dπ*q)
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

    function flight_computed_torque(q::Vector{T},qdot::Vector{T},a::Vector{T},p::Designs.Params) where T<:Real
        ∇V = potential_gradient(q,p)
        DA = flight_constraints_jac(q,p)
        m = size(DA,1) # number of constraints
        ddtDA = reshape(flight_constraints_hess(q,p)*qdot,size(DA))
        dπ = flight_anchor_pushforward(q,p)
        f = flight_template_dynamics(flight_anchor_projection(q,p),dπ*q)
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

    """ MODE GUARD FUNCTIONS """
    const STANCE_GUARD_KNEE_DIST = KNEE_MIN_DIST+.005
    const FLIGHT_GUARD_KNEE_DIST = KNEE_MIN_DIST+.005
    
    function stance_guard(q::Vector{T},p::Designs.Params) where T<:Real
        ϕ = Model.interior_leg_angle(q) 
        STANCE_GUARD_KNEE_DIST-2*p.l1*sin(ϕ)
    end

    function flight_guard(q::Vector{T},p::Designs.Params) where T<:Real
        ϕ = Model.interior_leg_angle(q)
        -STANCE_GUARD_KNEE_DIST+2*p.l1*sin(ϕ)

    end

""" precompilation triggers """

_q = [0.,pi/2,pi/2,0.,0.]
_q[[4,5]] = -stance_constraints(_q,Designs.default_params,[0.,0.])[[1,2]]
_qdot = zeros(5)
_DA = stance_constraints_jac(_q,Designs.default_params)
_DDA = stance_constraints_hess(_q,Designs.default_params)
_u = stance_control(_q,_qdot,Designs.default_params)
_qddot = stance_dynamics(_q,_qdot,_u,Designs.default_params)

_q = [0.,pi/2,pi/2,0.,0.]
_q[[4,5]] = -flight_constraints(_q,Designs.default_params)[[1,2]]
_DA = flight_constraints_jac(_q,Designs.default_params)
_DDA = flight_constraints_hess(_q,Designs.default_params)
_u = flight_control(_q,_qdot,Designs.default_params)
_qddot = flight_dynamics(_q,_qdot,_u,Designs.default_params)

stance_guard(_q,Designs.default_params)
flight_guard(_q,Designs.default_params)

end
