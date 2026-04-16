# clock_model.jl — Clock models and noise parameters for the Kalman Filter

using LinearAlgebra: I, norm
using StaticArrays: SMatrix, SVector, @SMatrix, @SVector

"""
    ClockNoiseParams

The 3-to-4 noise diffusion coefficients of the canonical clock SDE
(Zucca-Tavella 2005, Eq 1). WPM enters as measurement noise R;
WFM/RWFM/IRWFM enter as state-noise diffusions.
"""
Base.@kwdef struct ClockNoiseParams
    q_wpm::Float64                  # R = σ₀² (WPM, measurement)
    q_wfm::Float64                  # σ₁² (WFM, state)
    q_rwfm::Float64      = 0.0      # σ₂² (RWFM, state); 0 to disable
    q_irwfm::Float64     = 0.0      # σ₃² (IRWFM / drift RW, state)
end

"""
    ClockModel3

3-state linear-Gaussian clock model: (phase, frequency, drift).
Canonical case; matches Zucca-Tavella 2005 Eq 1 with σ₃² enabled.
"""
Base.@kwdef struct ClockModel3
    noise::ClockNoiseParams
    tau::Float64
end

"""
    ClockModel2

2-state: (phase, frequency). Drift is deterministic or absent.
"""
Base.@kwdef struct ClockModel2
    noise::ClockNoiseParams
    tau::Float64
end

"""
    ClockModelDiurnal

5-state: (phase, frequency, drift, diurnal-sin, diurnal-cos).
"""
Base.@kwdef struct ClockModelDiurnal
    noise::ClockNoiseParams
    tau::Float64
    period::Float64       = 86400.0
    q_diurnal::Float64    = 0.0
end

nstates(::ClockModel2) = 2
nstates(::ClockModel3) = 3
nstates(::ClockModelDiurnal) = 5

function build_phi(m::ClockModel2)
    [1.0 m.tau; 0.0 1.0]
end

function build_phi(m::ClockModel3)
    [1.0 m.tau m.tau^2 / 2.0; 0.0 1.0 m.tau; 0.0 0.0 1.0]
end

function build_phi(m::ClockModelDiurnal)
    Phi = Matrix{Float64}(I, 5, 5)
    Phi[1:3, 1:3] = build_phi(ClockModel3(m.noise, m.tau))
    return Phi
end

function build_Q(m::ClockModel2)
    τ = m.tau
    q_wfm = m.noise.q_wfm
    q_rwfm = m.noise.q_rwfm

    Q11 = q_wfm*τ + q_rwfm*τ^3/3.0
    Q12 = q_rwfm*τ^2/2.0
    Q22 = q_rwfm*τ
    return [Q11 Q12; Q12 Q22]
end

function build_Q(m::ClockModel3)
    τ = m.tau
    q_wfm = m.noise.q_wfm
    q_rwfm = m.noise.q_rwfm
    q_irwfm = m.noise.q_irwfm

    τ2 = τ^2; τ3 = τ^3; τ4 = τ^4; τ5 = τ^5
    Q11 = q_wfm*τ + q_rwfm*τ3/3.0 + q_irwfm*τ5/20.0
    Q12 = q_rwfm*τ2/2.0 + q_irwfm*τ4/8.0
    Q13 = q_irwfm*τ3/6.0
    Q22 = q_rwfm*τ + q_irwfm*τ3/3.0
    Q23 = q_irwfm*τ2/2.0
    Q33 = q_irwfm*τ
    
    return [Q11 Q12 Q13; Q12 Q22 Q23; Q13 Q23 Q33]
end

function build_Q(m::ClockModelDiurnal)
    Q = zeros(Float64, 5, 5)
    Q[1:3, 1:3] = build_Q(ClockModel3(m.noise, m.tau))
    Q[4, 4] = m.q_diurnal
    Q[5, 5] = m.q_diurnal
    return Q
end

build_H(::ClockModel2) = [1.0 0.0]
build_H(::ClockModel3) = [1.0 0.0 0.0]
function build_H(m::ClockModelDiurnal, k::Int)
    H = zeros(Float64, 1, 5)
    H[1, 1] = 1.0
    twopi = 2π
    H[1, 4] = sin(twopi * k / m.period)
    H[1, 5] = cos(twopi * k / m.period)
    return H
end

"""
    sigma_y_theory(model, tau) -> Float64

Theoretical Allan deviation σ_y(τ) corresponding to the clock SDE model.
"""
function sigma_y_theory(m::Union{ClockModel2, ClockModel3, ClockModelDiurnal}, tau::Float64)
    noise = m.noise
    R = noise.q_wpm
    q_wfm = noise.q_wfm
    q_rwfm = noise.q_rwfm
    q_irwfm = noise.q_irwfm
    var = 3.0 * R / tau^2 + q_wfm / tau + q_rwfm * tau / 3.0 + q_irwfm * tau^3 / 20.0
    return sqrt(var)
end

"""
    h_to_q(h::NamedTuple, tau0::Float64) -> ClockNoiseParams

Map SP1065 power-law coefficients h_α to clock-SDE diffusion coefficients.
"""
function h_to_q(h, tau0::Float64)
    f_h = 1.0 / (2.0 * tau0)
    q_wpm  = haskey(h,  2.0) ? h[ 2.0] * f_h / (4.0 * π^2) : 0.0
    q_wfm  = haskey(h,  0.0) ? h[ 0.0] / 2.0 : 0.0
    q_rwfm = haskey(h, -2.0) ? (2.0 * π^2) * h[-2.0] : 0.0
    return ClockNoiseParams(q_wpm=q_wpm, q_wfm=q_wfm, q_rwfm=q_rwfm, q_irwfm=0.0)
end

"""
    q_to_h(q::ClockNoiseParams, tau0) -> NamedTuple

Inverse mapping. Returns (h2=..., h0=..., h_2=...).
"""
function q_to_h(q::ClockNoiseParams, tau0::Float64)
    f_h = 1.0 / (2.0 * tau0)
    h2  = q.q_wpm * (4.0 * π^2) / f_h
    h0  = q.q_wfm * 2.0
    h_2 = q.q_rwfm / (2.0 * π^2)
    return (h2=h2, h0=h0, h_2=h_2)
end

function _dare_scalar_H(Φ::AbstractMatrix, Q::AbstractMatrix, R::Real;
                        tol::Float64 = 1e-12, max_iter::Int = 200)
    n = size(Φ, 1)
    P = Matrix{Float64}(Q)
    for _ in 1:max_iter
        Pp = Φ * P * Φ' + Q
        S  = Pp[1, 1] + R
        S <= 0.0 && return P
        K  = Pp[:, 1] ./ S
        P_new = Pp .- K * Pp[1, :]'
        P_new = 0.5 .* (P_new .+ P_new')
        if norm(P_new .- P) / max(norm(P), 1e-30) < tol
            return P_new
        end
        P = P_new
    end
    return P
end

function _dare_scalar_H_static(Φ::SMatrix{3,3,Float64}, Q::SMatrix{3,3,Float64}, R::Real;
                               tol::Float64 = 1e-12, max_iter::Int = 200)
    P = Q
    @inbounds for _ in 1:max_iter
        Pp = Φ * P * Φ' + Q
        S  = Pp[1,1] + R
        S <= 0.0 && return P
        K  = SVector{3,Float64}(Pp[1,1]/S, Pp[2,1]/S, Pp[3,1]/S)
        pr1 = SVector{3,Float64}(Pp[1,1], Pp[1,2], Pp[1,3])
        P_new = Pp - K * pr1'
        P_new = 0.5 * (P_new + P_new')
        diff = P_new - P
        d2 = sum(diff .* diff)
        n2 = max(sum(P .* P), 1e-60)
        if sqrt(d2 / n2) < tol
            return P_new
        end
        P = P_new
    end
    return P
end

function steady_state_covariance(m::ClockModel3)
    Phi = build_phi(m)
    Q = build_Q(m)
    R = m.noise.q_wpm
    return _dare_scalar_H(Phi, Q, R)
end

function steady_state_gain(m::ClockModel3)
    P = steady_state_covariance(m)
    H = build_H(m)
    R = m.noise.q_wpm
    S = (H * P * H')[1,1] + R
    return (P * H')[:] ./ S
end
