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

# 2. Partition data: Use first 10,000 points for training to keep optimization fast.
# Wait, user wants a long prediction. Let's use 20000 points for training.
N_train = 20000
train_phase = phase_shifted[1:N_train]

# test on the next 10000 points
predict_horizon = min(10000, length(phase_shifted) - N_train)
test_phase = phase_shifted[N_train+1:N_train+predict_horizon]

println("Loaded \$(length(phase_shifted)) points. Train: \$(N_train), Test/Horizon: \$(predict_horizon)")

# 3. Fit MHDEV
println("Fitting MHDEV...")
taus, mhdevs, _ = mhtotdev(train_phase, tau0)
noise_mhdev = fit_noise(taus, mhdevs, [:q_wpm, :q_wfm, :q_rwfm])
println("MHDEV Noise: ", noise_mhdev)

# 4. Fit NLL
println("Fitting NLL...")
noise_nll = optimize_nll(train_phase, tau0; 
                         noise_init=noise_mhdev, 
                         optimize_qwpm=true, optimize_irwfm=false, verbose=true)
println("NLL Noise: ", noise_nll)

# 5. Fit ALS
println("Fitting ALS...")
noise_als = als_fit(train_phase, tau0; 
                    noise_init=noise_nll, 
                    lags=50, max_iter=10, burn_in=100, optimize_qwpm=true, optimize_irwfm=false, verbose=true)
println("ALS Noise: ", noise_als)

# 6. Predict Holdover for each
function get_prediction(noise_params)
    m = ClockModel3(noise=noise_params, tau=tau0)
    # filter through train
    fil = kalman_filter(train_phase, m)
    # predict forward
    x_hat = fil.x_hat[end]
    P_hat = fil.P_hat[end]
    
    pred_phase = zeros(predict_horizon)
    Phi = build_phi(m)
    x_curr = x_hat
    P_curr = P_hat
    Q = build_Q(m)
    
    std_phase = zeros(predict_horizon)
    for i in 1:predict_horizon
        x_curr = Phi * x_curr
        # Use discrete Lyap predict
        P_curr = Phi * P_curr * Phi' + Q
        pred_phase[i] = x_curr[1]
        std_phase[i] = sqrt(P_curr[1,1])
    end
    return pred_phase, std_phase
end

println("Generating predictions...")
pred_mhdev, std_mhdev = get_prediction(noise_mhdev)
pred_nll, std_nll = get_prediction(noise_nll)
pred_als, std_als = get_prediction(noise_als)

# Convert all phase plots back to nanoseconds for easier readability.
test_phase_ns = test_phase .* 1e9
pred_mhdev_ns = pred_mhdev .* 1e9
pred_nll_ns = pred_nll .* 1e9
pred_als_ns = pred_als .* 1e9
std_als_ns = std_als .* 1e9

# Shift all to start at 0 so we see the prediction deviation clearly
bias_true = test_phase_ns[1]
test_phase_ns .-= bias_true
pred_mhdev_ns .-= bias_true
pred_nll_ns .-= bias_true
pred_als_ns .-= bias_true

# 7. Plotting
test_t = (1:predict_horizon) .* tau0

p = plot(test_t, test_phase_ns, label="True Data", linewidth=1.0, color=:black, title="Holdover Prediction (6k27feb Unsteered)", xlabel="Time (s)", ylabel="Phase (ns)")

plot!(p, test_t, pred_mhdev_ns, label="MHDEV Pred", linewidth=2, linestyle=:dash)
plot!(p, test_t, pred_nll_ns, label="NLL Pred", linewidth=2, linestyle=:dash)
plot!(p, test_t, pred_als_ns, label="ALS Pred", linewidth=2, linestyle=:dash)

plot!(p, test_t, pred_als_ns .+ 3 .* std_als_ns, color=:red, alpha=0.3, label="ALS 3σ bounds", style=:dot)
plot!(p, test_t, pred_als_ns .- 3 .* std_als_ns, color=:red, alpha=0.3, label="", style=:dot)

savefig(p, "reference/6k27feb_holdover.png")
println("Saved plot to reference/6k27feb_holdover.png")
