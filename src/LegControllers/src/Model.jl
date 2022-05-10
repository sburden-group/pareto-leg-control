module Model
using LinearAlgebra
using ForwardDiff
using ..Designs

"""
Definition of system model constants
"""
const g = 9.81
const m_body = 2.0
const m_weight = 1.0
# const Jm = .5/2*(.087^2+.08^2) # rough approximation motor inertia from thick-walled cylinder model.
const Jm = 2e-3

const foot_offset = 0.03 # distance from kinematic chain closure to foot location in meters
const spring_knee_offset = 0.015 # radial distance from "knee" joints to free end of extension springs
const R = 0.186           # winding resistance
const Kv = (2pi*100.0/60) # (Rad/s) / Volt
const Ke = 1/Kv           # Volt / (rad/s)


# indices of state variables
const body_idx = 1  # body location in world frame
const θ1_idx = 2    # motor 1 axis
const θ2_idx = 3    # motor 2 axis
const foot_idx = [4,5] # foot (x,y) location in world frame
const n_coords = 5  # number of coordinate variables

# input - state map
const G = [
    0. 0.
    1.0 0.0
    0.0 1.0
    0.0 0.0
    0.0 0.0
]

""" Begin kinematics """
function hip_foot_angle(q::Vector{T}) where {T <: Real}
    (q[θ1_idx]-q[θ2_idx])/2.0
end

function interior_leg_angle(q::Vector{T}) where {T <: Real}
    (q[θ1_idx]+q[θ2_idx])/2.0
end

function leg_length(q::Vector{T}, p::Designs.Params) where {T <: Real}
    ϕ = interior_leg_angle(q)
    p.l1*cos(ϕ)+sqrt(p.l2^2-(p.l1*sin(ϕ))^2)+foot_offset
end

"""
Computes holonomic constraints associated with five-bar mechanism
"""
function kin_constraints(q::Vector{T}, p::Designs.Params) where T<:Real
    θ = hip_foot_angle(q)
    l = leg_length(q,p)
    q[foot_idx]-[l*sin(θ),q[body_idx]-l*cos(θ)]
end

function kin_constraints_jac(q::Vector{T},p::Designs.Params) where T<:Real
    f = q->kin_constraints(q,p)
    cfg = ForwardDiff.JacobianConfig(f,q,ForwardDiff.Chunk{n_coords}())
    ForwardDiff.jacobian(f,q,cfg)
end

function kin_constraints_hess(q::Vector{T},p::Designs.Params) where T<:Real
    f = q->kin_constraints_jac(q,p)
    cfg = ForwardDiff.JacobianConfig(f,q,ForwardDiff.Chunk{n_coords}())
    ForwardDiff.jacobian(f,q,cfg)
end

""" Computes potential energy of springs. """
function spring_energy(q::Vector{T},p::Designs.Params) where {T <: Real}
    l = leg_length(q,p)
    θ1 = q[θ1_idx]

    # compute energy in spring 1
    joint_location = Array{T}([p.l1*sin(θ1),-p.l1*cos(θ1)])
    r = (p.l1+p.s1_L+spring_knee_offset)
    fixed_end = Array{T}([r*sin(p.s1_r),-r*cos(p.s1_r)]) 
    free_end = joint_location + spring_knee_offset*(fixed_end-joint_location)/norm(fixed_end-joint_location)
    δx = norm(fixed_end-free_end)
    s1_energy = .5*p.s1_k*δx^2+p.s1_Fi*δx

    # compute energy in spring 3
    δx = p.l1+p.l2+foot_offset-l
    s3_energy = .5*p.s3_k*δx^2

    return T(s1_energy+s3_energy)
end

function potential_energy(q::Vector{T},p::Designs.Params) where T<:Real
        m_body*g*q[body_idx] + spring_energy(q,p)
end

function kinetic_energy(q::Vector{T},qdot::Vector{T},p::Designs.Params) where T<:Real
    .5*m_body*qdot[body_idx]^2 + .5*Jm*sum(qdot[[θ1_idx,θ2_idx]].^2)
end


_q = [0.,pi/2,pi/2,0.,0.]
_q[1] = kin_constraints(_q,Designs.default_params)[1]
_qdot = zeros(5)
_DA = kin_constraints_jac(_q,Designs.default_params)
_DDA = kin_constraints_hess(_q,Designs.default_params)
end
