# holdover_trajectories.jl — Trajectory-averaged 1-day holdover RMS per method.
#
# For each GMR6000 window (524288 samples, detrended) and each q-estimator
# (naive / RF / XGB / MHDEV-fit / ALS), we:
#   1. Build a ClockModel3 from the method's q triple.
#   2. Run one Kalman filter over the full window (PID disabled — pure estimation).
#   3. Slide a 1-day holdover start across the KF trace at a fixed stride; for
#      each start i, propagate (x_i, P_i) for HORIZON_S steps and compute the
#      phase-residual RMS against the true data over the horizon.
#   4. Report per-(window, method) mean ± std RMS across all trajectories.
#
# Inputs:
#   q_methods.csv — columns: window,method,q_wpm,q_wfm,q_rwfm
# Output:
#   holdover_trajectories.csv — columns:
#     window,method,n_trajectories,rms_mean,rms_median,rms_p10,rms_p90,rms_std

using Pkg; Pkg.activate(@__DIR__)
using SigmaTau
using Printf, Statistics, LinearAlgebra

const REAL_PATH = joinpath(@__DIR__, "..", "..", "reference", "raw", "6k27febunsteered.txt")
const OUT_DIR   = joinpath(@__DIR__, "..", "data")
const WINDOW_N  = 524_288
const TAU0      = 1.0
const HORIZON_S = 86_400              # 1 day
const STRIDE    = 8_640                # 10 starts per day → ~50 trajectories / window
const BURN_IN   = 10_000              # skip earliest KF transient starts

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
    return y .- (a .+ b .* t)
end

# Propagate a KF state (x,P) forward HORIZON steps under `model` and compute
# the phase-residual RMS vs `truth[1:HORIZON]`. Fast: reuses Φ, Q from model.
function holdover_rms(x0::Vector{Float64}, P0::Matrix{Float64},
                     model, truth::AbstractVector{<:Real})
    Phi = build_phi(model)
    Q   = build_Q(model)
    horizon = length(truth)
    x = copy(x0)
    P = copy(P0)
    ss = 0.0
    n  = 0
    for i in 1:horizon
        x = Phi * x
        P = Phi * P * Phi' + Q
        P = 0.5 .* (P .+ P')
        r = truth[i] - x[1]
        if isfinite(r)
            ss += r * r
            n  += 1
        end
    end
    return n > 0 ? sqrt(ss / n) : NaN
end

function trajectory_rms(xw_detrended::Vector{Float64},
                         q::ClockNoiseParams; stride=STRIDE, horizon=HORIZON_S, burn_in=BURN_IN)
    model = ClockModel3(noise = q, tau = TAU0)
    kf = kalman_filter(xw_detrended, model; g_p = 0.0, g_i = 0.0, g_d = 0.0)
    N = length(xw_detrended)
    last_start = N - horizon
    starts = burn_in:stride:last_start
    rms = Float64[]
    for i in starts
        x0 = Float64[kf.phase_est[i], kf.freq_est[i], kf.drift_est[i]]
        P0 = Matrix{Float64}(kf.P_history[:, :, i])
        truth = @view xw_detrended[i+1:i+horizon]
        r = holdover_rms(x0, P0, model, truth)
        isfinite(r) && push!(rms, r)
    end
    return rms
end

function main()
    q_methods_path = joinpath(OUT_DIR, "q_methods.csv")
    isfile(q_methods_path) || error("missing $(q_methods_path) — run notebook_extras.py first")

    # Parse q_methods: window,method,q_wpm,q_wfm,q_rwfm
    lines = readlines(q_methods_path)
    header = split(lines[1], ",")
    rows = []
    for line in lines[2:end]
        t = split(strip(line), ",")
        push!(rows, (window=parse(Int, t[1]),
                     method=String(t[2]),
                     q_wpm=parse(Float64, t[3]),
                     q_wfm=parse(Float64, t[4]),
                     q_rwfm=parse(Float64, t[5])))
    end

    ph = load_phase_ns_to_seconds(REAL_PATH)
    @info "phase loaded" n=length(ph)

    # Cache per-window detrended windows
    unique_windows = sort(unique([r.window for r in rows]))
    xw_cache = Dict{Int, Vector{Float64}}()
    for w in unique_windows
        rng = (1 + w*WINDOW_N) : ((w+1)*WINDOW_N)
        xw_cache[w] = linear_detrend(Vector{Float64}(ph[rng]))
        @info "detrended window $w" n=length(xw_cache[w])
    end

    results = []
    for r in rows
        xw = xw_cache[r.window]
        q  = ClockNoiseParams(q_wpm=r.q_wpm, q_wfm=r.q_wfm, q_rwfm=r.q_rwfm)
        t0 = time()
        rms = trajectory_rms(xw, q)
        dt = time() - t0
        push!(results, (window=r.window, method=r.method, rms=rms))
        @info "window $(r.window) method=$(r.method) n_traj=$(length(rms)) mean_rms=$(round(mean(rms), sigdigits=3)) dt=$(round(dt, digits=1))s"
    end

    out = joinpath(OUT_DIR, "holdover_trajectories.csv")
    open(out, "w") do io
        println(io, "window,method,n_trajectories,rms_mean,rms_median,rms_p10,rms_p90,rms_std")
        for r in results
            rms = r.rms
            if isempty(rms)
                @printf(io, "%d,%s,0,NaN,NaN,NaN,NaN,NaN\n", r.window, r.method); continue
            end
            @printf(io, "%d,%s,%d,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    r.window, r.method, length(rms),
                    mean(rms), median(rms),
                    quantile(rms, 0.10), quantile(rms, 0.90),
                    std(rms))
        end
    end
    @info "wrote" out n_rows=length(results)
end

main()
