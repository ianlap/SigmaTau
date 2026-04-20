# warmstart_compare.jl — Compare ALS convergence & holdover error under two
# different warm starts (naive train-mean vs ML prediction).
#
# Inputs:
#   q_inits.csv — per-window warm-start table produced by notebook_extras.py
#     columns: window,naive_qwpm,naive_qwfm,naive_qrwfm,ml_qwpm,ml_qwfm,ml_qrwfm
#
# Output:
#   warmstart_compare.csv — per-window ALS iteration count & holdover RMS for each
#   warm start.
#
# For each window we take an 80/20 split of the 524288-sample window:
#   - first 80% feeds ALS + KF state estimate
#   - last  20% is the holdover horizon (true phase compared to KF prediction)

using Pkg; Pkg.activate(@__DIR__)
using SigmaTau
using Printf, Statistics

const REAL_PATH = joinpath(@__DIR__, "..", "..", "reference", "raw", "6k27febunsteered.txt")
const OUT_DIR   = joinpath(@__DIR__, "..", "data")
const WINDOW_N  = 524_288
const TAU0      = 1.0
const TRAIN_FRAC = 0.8
const ALS_MAX_ITER = 30
const ALS_TOL      = 1e-6   # tight — default 1e-3 converges in 1 iter from any reasonable seed
const ALS_LAGS     = 30
const ALS_BURN_IN  = 50

function load_phase_ns_to_seconds(path)
    ph = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        tok = split(s)
        length(tok) < 2 && continue
        try; push!(ph, parse(Float64, tok[2])); catch; end
    end
    test_x = ph[1:min(65_536, length(ph))]
    ad1    = adev(test_x, TAU0).deviation[1]
    factor = ad1 > 1e-3 ? 1e-9 : (ad1 > 1e-6 ? 1e-6 : 1.0)
    return ph .* factor
end

function linear_detrend(y::AbstractVector{<:Real})
    n = length(y); t = collect(1.0:n)
    mt = mean(t); my = mean(y)
    b = sum((t .- mt) .* (y .- my)) / sum((t .- mt).^2)
    a = my - b * mt
    return y .- (a .+ b .* t), (a, b)
end

# Replicate als_fit's outer loop so we can count the number of outer iterations
# required to meet the relative-q_wfm convergence criterion (|Δq/q| < tol).
function als_fit_iter_counted(data::Vector{Float64}, tau0::Real, q_init::ClockNoiseParams;
                              max_iter::Int = ALS_MAX_ITER,
                              tol::Float64 = ALS_TOL,
                              lags::Int = ALS_LAGS,
                              burn_in::Int = ALS_BURN_IN)
    current = q_init
    n_iter  = 0
    converged = false
    for iter in 1:max_iter
        n_iter = iter
        n_out = SigmaTau._als_iteration(data, tau0, current, lags, burn_in, false, false)
        diff = abs(n_out.q_wfm - current.q_wfm) / max(current.q_wfm, 1e-40)
        current = n_out
        if diff < tol
            converged = true
            break
        end
    end
    return (noise = current, n_iter = n_iter, converged = converged)
end

function run_one(x_train::Vector{Float64}, x_holdover::Vector{Float64},
                 q_init::ClockNoiseParams)
    # 1. ALS from this warm start — count outer iterations
    res = als_fit_iter_counted(x_train, TAU0, q_init)
    q_opt = res.noise

    # 2. run KF on training window (PID disabled — pure estimation)
    model = ClockModel3(noise = q_opt, tau = TAU0)
    kf = kalman_filter(x_train, model; g_p = 0.0, g_i = 0.0, g_d = 0.0)

    # 3. holdover prediction + RMS
    horizon = length(x_holdover)
    hold    = predict_holdover(kf, horizon; model = model)
    residual = x_holdover .- hold.phase_pred
    rms = sqrt(mean(filter(isfinite, residual.^2)))
    return (n_iter = res.n_iter, converged = res.converged,
            q_opt = q_opt, rms = rms)
end

function main()
    inits_path = joinpath(OUT_DIR, "q_inits.csv")
    isfile(inits_path) || error("missing $(inits_path) — run notebook_extras.py first")
    lines = readlines(inits_path)
    header = split(lines[1], ",")
    rows = [Dict(header[i] => tok[i] for i in eachindex(header)) for tok in
            [split(strip(line), ",") for line in lines[2:end]]]

    ph = load_phase_ns_to_seconds(REAL_PATH)
    @info "phase loaded" n=length(ph)

    results = []
    for r in rows
        w = parse(Int, r["window"])
        rng = (1 + w * WINDOW_N) : ((w+1) * WINDOW_N)
        xw_raw = Vector{Float64}(ph[rng])
        xw, trend = linear_detrend(xw_raw)
        @info "window $w detrend" intercept=trend[1] slope=trend[2]

        n_tr = Int(round(length(xw) * TRAIN_FRAC))
        x_tr = xw[1:n_tr]
        x_ho = xw[n_tr+1:end]

        q_naive = ClockNoiseParams(
            q_wpm  = parse(Float64, r["naive_qwpm"]),
            q_wfm  = parse(Float64, r["naive_qwfm"]),
            q_rwfm = parse(Float64, r["naive_qrwfm"]))
        q_ml = ClockNoiseParams(
            q_wpm  = parse(Float64, r["ml_qwpm"]),
            q_wfm  = parse(Float64, r["ml_qwfm"]),
            q_rwfm = parse(Float64, r["ml_qrwfm"]))

        @info "window $w" n_tr=n_tr n_ho=length(x_ho)
        rn = run_one(x_tr, x_ho, q_naive)
        rm = run_one(x_tr, x_ho, q_ml)
        @info "  ALS naive" n_iter=rn.n_iter rms=rn.rms converged=rn.converged
        @info "  ALS ml"    n_iter=rm.n_iter rms=rm.rms converged=rm.converged

        push!(results, (window=w,
                        n_iter_naive=rn.n_iter, converged_naive=rn.converged, rms_naive=rn.rms,
                        n_iter_ml=rm.n_iter,    converged_ml=rm.converged,    rms_ml=rm.rms))
    end

    out = joinpath(OUT_DIR, "warmstart_compare.csv")
    open(out, "w") do io
        println(io, "window,n_iter_naive,converged_naive,rms_naive,",
                    "n_iter_ml,converged_ml,rms_ml,",
                    "speedup_iter,rms_ratio")
        for r in results
            speedup = r.n_iter_ml > 0 ? r.n_iter_naive / r.n_iter_ml : NaN
            rms_r   = r.rms_ml   > 0 ? r.rms_naive / r.rms_ml : NaN
            @printf(io, "%d,%d,%d,%.6e,%d,%d,%.6e,%.4f,%.4f\n",
                    r.window, r.n_iter_naive, r.converged_naive ? 1 : 0, r.rms_naive,
                    r.n_iter_ml, r.converged_ml ? 1 : 0, r.rms_ml, speedup, rms_r)
        end
    end
    @info "wrote" out
end

main()
