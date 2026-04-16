using Pkg
Pkg.activate("julia")
using SigmaTau
using DelimitedFiles
using Plots
using LinearAlgebra

# 1. Load the data
data_raw = readdlm("reference/raw/6k27febunsteered.txt")
time_days = data_raw[:, 1]
phase_ns = data_raw[:, 2]

# The user noted units are in ns. Convert to seconds.
phase_shifted = (phase_ns .- phase_ns[1]) .* 1e-9

tau0 = round((time_days[2] - time_days[1]) * 86400.0)

# 2. Partition data: 20000 points for training, next 10000 for test.
N_train = 20000
train_phase = phase_shifted[1:N_train]

predict_horizon = min(10000, length(phase_shifted) - N_train)
test_phase = phase_shifted[N_train+1:N_train+predict_horizon]

println("Loaded $(length(phase_shifted)) points. Train: $(N_train), Test/Horizon: $(predict_horizon)")

# 3. Fit NLL (from defaults — no mhdev warm-start needed)
println("Fitting NLL...")
noise_nll = optimize_nll(train_phase, tau0;
                         optimize_qwpm=true, optimize_irwfm=false, verbose=true)
println("NLL Noise: ", noise_nll)

# 4. Fit ALS (warm-started from NLL)
println("Fitting ALS...")
noise_als = als_fit(train_phase, tau0;
                    noise_init=noise_nll,
                    lags=50, max_iter=10, burn_in=100, optimize_qwpm=true, optimize_irwfm=false, verbose=true)
println("ALS Noise: ", noise_als)

# 5. Predict holdover using the new API
function run_holdover(noise_params)
    m = ClockModel3(noise=noise_params, tau=tau0)
    kf = kalman_filter(train_phase, m; g_p=0.0, g_i=0.0, g_d=0.0)
    return predict_holdover(kf, predict_horizon)
end

println("Generating predictions...")
hr_nll = run_holdover(noise_nll)
hr_als = run_holdover(noise_als)

# Convert to nanoseconds for plotting
test_phase_ns = test_phase .* 1e9
pred_nll_ns   = hr_nll.phase_pred .* 1e9
pred_als_ns   = hr_als.phase_pred .* 1e9
std_als_ns    = sqrt.(hr_als.P_pred[1, 1, :]) .* 1e9

# Shift all to start at 0
bias_true = test_phase_ns[1]
test_phase_ns .-= bias_true
pred_nll_ns   .-= bias_true
pred_als_ns   .-= bias_true

# 6. Plot
test_t = (1:predict_horizon) .* tau0

p = plot(test_t, test_phase_ns, label="True Data", linewidth=1.0, color=:black,
         title="Holdover Prediction (6k27feb Unsteered)",
         xlabel="Time (s)", ylabel="Phase (ns)")

plot!(p, test_t, pred_nll_ns, label="NLL Pred", linewidth=2, linestyle=:dash)
plot!(p, test_t, pred_als_ns, label="ALS Pred", linewidth=2, linestyle=:dash)

plot!(p, test_t, pred_als_ns .+ 3 .* std_als_ns, color=:red, alpha=0.3, label="ALS 3σ bounds", style=:dot)
plot!(p, test_t, pred_als_ns .- 3 .* std_als_ns, color=:red, alpha=0.3, label="", style=:dot)

savefig(p, "reference/6k27feb_holdover.png")
println("Saved plot to reference/6k27feb_holdover.png")
