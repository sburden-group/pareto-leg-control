module Designs
n_params = 12
struct Params{T<:Real}
    s1_L::T        # Extension Spring 1 free length
    s1_k::T        # Extension Spring 1 spring rate
    s1_Fi::T       # Extension Spring 1 initial tension
    s1_r::T        # Extension Spring 1 rest angle
    s2_L::T        # Extension Spring 2 free length
    s2_k::T        # Extension Spring 2 spring rate
    s2_Fi::T       # Extension Spring 2 initial tension
    s2_r::T        # Extension Spring 2 rest angle
    s3_L::T        # Compression Spring free length
    s3_k::T        # Compression Spring spring rate 
    l1::T          # Femur length
    l2::T          # Tibia length
end

"""
Packs a design parameter structure into a vector.
"""
function pack(p::Designs.Params) where T<:Real
    Vector{T}([
        p.s1_L,p.s1_k,p.s1_Fi,p.s1_r,
        p.s2_L,p.s2_k,p.s2_Fi,p.s2_r,
        p.s3_L,p.s3_K,
        p.l1,p.l2
    ])
end

"""
Unpacks a deisgn parameter vector into a design parameter structure
"""
function unpack(x::Vector{T}) where T<:Real
    return Params(x...)
end

end