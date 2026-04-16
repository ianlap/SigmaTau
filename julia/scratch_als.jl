using Pkg, LinearAlgebra
Pkg.activate("julia")
using SigmaTau
tau0 = 1.0
noise_current = ClockNoiseParams(q_wpm=1e-9, q_wfm=1e-11, q_rwfm=1e-12)
m = ClockModel3(noise=noise_current, tau=tau0)
P_inf = steady_state_covariance(m)
Phi = build_phi(m)
H = build_H(m)
R_current = noise_current.q_wpm
S = (H * P_inf * H')[1,1] + R_current
K = (P_inf * H') ./ S
Abar = Phi - K * H
println("Abar = \n", Abar)

n_s = nstates(m)
pinv_A = pinv(I(n_s^2) - kron(Abar, Abar))

for k in [:q_wpm, :q_wfm, :q_rwfm]
    p_dict = Dict(:q_wpm => 0.0, :q_wfm => 0.0, :q_rwfm => 0.0, :q_irwfm => 0.0)
    p_dict[k] = 1.0
    Q_b = build_Q(ClockModel3(noise=ClockNoiseParams(p_dict[:q_wpm], p_dict[:q_wfm], p_dict[:q_rwfm], p_dict[:q_irwfm]), tau=tau0))
    R_b = p_dict[:q_wpm]
    vec_P = pinv_A * vec(Q_b .+ K * R_b * K')
    P_b = reshape(vec_P, n_s, n_s)
    c = zeros(2)
    c[1] = (H * P_b * H')[1,1] + R_b
    c[2] = (H * Abar * P_b * H')[1,1] - (H * K * R_b)[1]
    println("C_basis C_0 for \$k: ", c[1])
    println("C_basis C_1 for \$k: ", c[2])
end

println("P_inf: ", P_inf)
println("K: ", K)
