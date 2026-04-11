# Deviation calculation functions for StabLab.jl
# Each function: validate → default mlist → noise_id → kernel → _make_result

"""
    adev(phase_data, tau0; mlist=nothing, confidence=0.683)

Overlapping Allan deviation (OADEV). Second differences on phase.
"""
function adev(phase_data::AbstractVector{T}, tau0::Real;
              mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
              confidence::Real=0.683) where T<:Real
    x = validate_phase_data(phase_data)
    tau0 = validate_tau0(tau0)
    N = length(x)
    mlist = something(mlist, _default_mlist(N, 2))
    alpha = noise_id(x, mlist, "phase")

    tau  = mlist .* tau0
    dev  = Vector{Float64}(undef, length(mlist))
    neff = Vector{Int}(undef, length(mlist))

    for (k, m) in enumerate(mlist)
        L = N - 2m
        if L <= 0
            dev[k] = NaN; neff[k] = 0; continue
        end
        neff[k] = L
        d2 = @view(x[1+2m:N]) .- 2 .* @view(x[1+m:N-m]) .+ @view(x[1:L])
        dev[k] = sqrt(_meansq(d2) / (2 * m^2 * tau0^2))
    end

    _make_result(tau, dev, alpha, neff, tau0, N, "adev", confidence)
end

"""
    mdev(phase_data, tau0; mlist=nothing, confidence=0.683)

Modified Allan deviation. Cumsum-based triple-difference formulation.
"""
function mdev(phase_data::AbstractVector{T}, tau0::Real;
              mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
              confidence::Real=0.683) where T<:Real
    x = validate_phase_data(phase_data)
    tau0 = validate_tau0(tau0)
    N = length(x)
    mlist = something(mlist, _default_mlist(N, 3))
    alpha = noise_id(x, mlist, "phase")

    x_cs = cumsum([zero(T); x])
    tau  = mlist .* tau0
    dev  = Vector{Float64}(undef, length(mlist))
    neff = Vector{Int}(undef, length(mlist))

    for (k, m) in enumerate(mlist)
        Ne = N - 3m + 1
        if Ne <= 0
            dev[k] = NaN; neff[k] = 0; continue
        end
        neff[k] = Ne
        s1 = @view(x_cs[1+m:Ne+m])     .- @view(x_cs[1:Ne])
        s2 = @view(x_cs[1+2m:Ne+2m])   .- @view(x_cs[1+m:Ne+m])
        s3 = @view(x_cs[1+3m:Ne+3m])   .- @view(x_cs[1+2m:Ne+2m])
        d  = (s3 .- 2 .* s2 .+ s1) ./ m
        dev[k] = sqrt(_meansq(d) / (2 * m^2 * tau0^2))
    end

    _make_result(tau, dev, alpha, neff, tau0, N, "mdev", confidence)
end

"""
    mhdev(phase_data, tau0; mlist=nothing, confidence=0.683)

Modified Hadamard deviation. Third differences + moving average.
"""
function mhdev(phase_data::AbstractVector{T}, tau0::Real;
               mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
               confidence::Real=0.683) where T<:Real
    x = validate_phase_data(phase_data)
    tau0 = validate_tau0(tau0)
    N = length(x)
    mlist = something(mlist, _default_mlist(N, 4))
    alpha = noise_id(x, mlist, "phase")

    tau  = mlist .* tau0
    dev  = Vector{Float64}(undef, length(mlist))
    neff = Vector{Int}(undef, length(mlist))

    for (k, m) in enumerate(mlist)
        Ne = N - 4m + 1
        if Ne <= 0
            dev[k] = NaN; neff[k] = 0; continue
        end
        neff[k] = Ne
        d4 = @view(x[1:Ne]) .- 3 .* @view(x[1+m:Ne+m]) .+ 3 .* @view(x[1+2m:Ne+2m]) .- @view(x[1+3m:Ne+3m])
        S  = cumsum([zero(T); d4])
        avg = @view(S[m+1:end]) .- @view(S[1:end-m])
        dev[k] = sqrt(_meansq(avg) / (6 * m^2)) / tau[k]
    end

    _make_result(tau, dev, alpha, neff, tau0, N, "mhdev", confidence)
end

"""
    tdev(phase_data, tau0; mlist=nothing, confidence=0.683)

Time deviation: `TDEV = τ · MDEV / √3`. Returns seconds.
"""
function tdev(phase_data::AbstractVector{T}, tau0::Real;
              mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
              confidence::Real=0.683) where T<:Real
    mr = mdev(phase_data, tau0; mlist=mlist, confidence=confidence)
    tdev_vals = mr.tau .* mr.deviation ./ sqrt(3)
    ci_scaled = mr.ci .* (mr.tau ./ sqrt(3))
    DeviationResult(mr.tau, tdev_vals, mr.edf, ci_scaled, mr.alpha, mr.neff,
                    mr.tau0, mr.N, "tdev", mr.confidence)
end

"""
    ldev(phase_data, tau0; mlist=nothing, confidence=0.683)

Lapinski deviation: `LDEV = τ · MHDEV / √(10/3)`. Returns seconds.
"""
function ldev(phase_data::AbstractVector{T}, tau0::Real;
              mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
              confidence::Real=0.683) where T<:Real
    mr = mhdev(phase_data, tau0; mlist=mlist, confidence=confidence)
    scale = mr.tau ./ sqrt(10/3)
    ldev_vals = scale .* mr.deviation
    ci_scaled = mr.ci .* scale
    DeviationResult(mr.tau, ldev_vals, mr.edf, ci_scaled, mr.alpha, mr.neff,
                    mr.tau0, mr.N, "ldev", mr.confidence)
end

"""
    totdev(phase_data, tau0; mlist=nothing, confidence=0.683)

Total deviation with detrending and symmetric reflection. SP1065.
"""
function totdev(phase_data::AbstractVector{T}, tau0::Real;
                mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                confidence::Real=0.683) where T<:Real
    x = validate_phase_data(phase_data)
    tau0 = validate_tau0(tau0)
    N = length(x)
    mlist = something(mlist, _default_mlist(N, 2))
    alpha = noise_id(x, mlist, "phase")

    xd = detrend_linear(x)
    x_left  = 2xd[1]   .- @view(xd[2:N-1])
    x_right = 2xd[end] .- @view(xd[N-1:-1:2])
    x_star  = [x_left; xd; x_right]
    off = length(x_left)

    tau_out  = Float64[]
    dev_out  = Float64[]
    neff_out = Int[]
    alpha_out = Float64[]

    for (k, m) in enumerate(mlist)
        i_all  = 1:(3N - 2m - 4)
        center = i_all .+ m
        mask   = (center .>= 1) .& (center .<= N)
        any(mask) || continue
        ii = i_all[mask]

        # second differences on extended data — loop to avoid temp index arrays
        D = 0.0
        count = 0
        for j in ii
            d2 = x_star[off + j + 2m] - 2x_star[off + j + m] + x_star[off + j]
            D += d2^2
            count += 1
        end
        den = 2 * (N - 2) * (m * tau0)^2
        push!(tau_out,  m * tau0)
        push!(dev_out,  sqrt(D / den))
        push!(neff_out, count)
        push!(alpha_out, alpha[k])
    end

    isempty(tau_out) && return DeviationResult(
        T[], T[], T[], Matrix{T}(undef,0,2), Int[], Int[], tau0, N, "totdev", confidence)

    _make_result(tau_out, dev_out, alpha_out, neff_out, tau0, N, "totdev", confidence)
end

"""
    hdev(phase_data, tau0; mlist=nothing, confidence=0.683)

Overlapping Hadamard deviation. Third differences on phase.
"""
function hdev(phase_data::AbstractVector{T}, tau0::Real;
              mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
              confidence::Real=0.683) where T<:Real
    x = validate_phase_data(phase_data)
    tau0 = validate_tau0(tau0)
    N = length(x)
    mlist = something(mlist, _default_mlist(N, 4))
    alpha = noise_id(x, mlist, "phase")

    tau  = mlist .* tau0
    dev  = Vector{Float64}(undef, length(mlist))
    neff = Vector{Int}(undef, length(mlist))

    for (k, m) in enumerate(mlist)
        L = N - 3m
        if L <= 0
            dev[k] = NaN; neff[k] = 0; continue
        end
        neff[k] = L
        d3 = @view(x[1+3m:N]) .- 3 .* @view(x[1+2m:N-m]) .+ 3 .* @view(x[1+m:N-2m]) .- @view(x[1:L])
        dev[k] = sqrt(_meansq(d3) / (6 * tau[k]^2))
    end

    _make_result(tau, dev, alpha, neff, tau0, N, "hdev", confidence)
end

"""
    mtotdev(phase_data, tau0; mlist=nothing, confidence=0.683)

Modified total deviation. Half-average detrend, uninverted even reflection.
"""
function mtotdev(phase_data::AbstractVector{T}, tau0::Real;
                 mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                 confidence::Real=0.683) where T<:Real
    x = validate_phase_data(phase_data)
    tau0 = validate_tau0(tau0)
    N = length(x)
    mlist = something(mlist, _default_mlist(N, 3))
    alpha = noise_id(x, mlist, "phase")

    tau_out  = Float64[]
    dev_out  = Float64[]
    neff_out = Int[]
    alpha_out = Float64[]

    for (k, m) in enumerate(mlist)
        nsubs = N - 3m + 1
        nsubs < 1 && continue

        # pre-allocate per-m buffers
        seg_len = 3m
        seq        = Vector{T}(undef, seg_len)
        seq_det    = Vector{T}(undef, seg_len)
        ext        = Vector{T}(undef, 3 * seg_len)
        cs_len     = 3 * seg_len + 1
        cs         = Vector{T}(undef, cs_len)

        outer_sum = 0.0
        for n in 1:nsubs
            copyto!(seq, 1, x, n, seg_len)

            # half-average detrend
            half_n = seg_len / 2
            if m == 1
                slope = (seq[3] - seq[1]) / (2tau0)
            else
                hi = floor(Int, half_n)
                s1 = sum(@view(seq[1:hi])) / hi
                s2 = sum(@view(seq[hi+1:seg_len])) / (seg_len - hi)
                slope = (s2 - s1) / (half_n * tau0)
            end
            for j in 1:seg_len
                seq_det[j] = seq[j] - slope * tau0 * (j - 1)
            end

            # symmetric reflection: [rev; seq_det; rev]
            for j in 1:seg_len
                ext[j]             = seq_det[seg_len - j + 1]
                ext[seg_len + j]   = seq_det[j]
                ext[2seg_len + j]  = seq_det[seg_len - j + 1]
            end

            # cumsum
            cs[1] = zero(T)
            for j in 1:3seg_len
                cs[j+1] = cs[j] + ext[j]
            end

            # second differences via cumsum windows
            n_d2 = 6m - 3m + 1  # length of d2 array = 3m+1
            block_sum = 0.0
            for j in 0:(6m - 3m)
                a1 = (cs[j+m+1] - cs[j+1]) / m
                a2 = (cs[j+2m+1] - cs[j+m+1]) / m
                a3 = (cs[j+3m+1] - cs[j+2m+1]) / m
                d2 = a3 - 2a2 + a1
                block_sum += d2^2
            end
            outer_sum += block_sum / (6m)
        end

        push!(tau_out,  m * tau0)
        push!(dev_out,  sqrt(outer_sum / (2 * (m * tau0)^2 * nsubs)))
        push!(neff_out, nsubs)
        push!(alpha_out, alpha[k])
    end

    isempty(tau_out) && return DeviationResult(
        T[], T[], T[], Matrix{T}(undef,0,2), Int[], Int[], tau0, N, "mtotdev", confidence)

    _make_result(tau_out, dev_out, alpha_out, neff_out, tau0, N, "mtotdev", confidence)
end

"""
    htotdev(phase_data, tau0; mlist=nothing, confidence=0.683)

Hadamard total deviation. Frequency-domain, SP1065 detrending. m=1 delegates to hdev.
"""
function htotdev(phase_data::AbstractVector{T}, tau0::Real;
                 mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                 confidence::Real=0.683) where T<:Real
    x = validate_phase_data(phase_data)
    tau0 = validate_tau0(tau0)
    N = length(x)
    y  = diff(x) ./ tau0
    Ny = length(y)
    mlist = something(mlist, _default_mlist(Ny, 3))
    alpha = noise_id(x, mlist, "phase")

    tau_out  = Float64[]
    dev_out  = Float64[]
    neff_out = Int[]
    alpha_out = Float64[]

    for (idx, m) in enumerate(mlist)
        if m == 1
            hr = hdev(x, tau0, mlist=[1])
            if !isempty(hr.deviation)
                push!(tau_out, tau0); push!(dev_out, hr.deviation[1])
                push!(neff_out, 0); push!(alpha_out, alpha[idx])
            end
            continue
        end

        n_iter = Ny - 3m + 1
        n_iter < 1 && continue

        # pre-allocate per-m buffers
        seg_len = 3m
        xs     = Vector{T}(undef, seg_len)
        x0     = Vector{T}(undef, seg_len)
        xstar  = Vector{T}(undef, 3seg_len)
        cs     = Vector{T}(undef, 3seg_len + 1)

        dev_sum = 0.0
        for i in 0:(n_iter-1)
            copyto!(xs, 1, y, i+1, seg_len)

            # half-average detrend
            hi = floor(Int, seg_len / 2)
            lo_start = ceil(Int, seg_len / 2) + 1
            m1 = sum(@view(xs[1:hi])) / hi
            m2 = sum(@view(xs[lo_start:seg_len])) / (seg_len - lo_start + 1)
            slope = if isodd(seg_len)
                (m2 - m1) / (0.5(seg_len - 1) + 1)
            else
                (m2 - m1) / (0.5seg_len)
            end
            mid = floor(seg_len / 2)
            for j in 1:seg_len
                x0[j] = xs[j] - slope * (j - 1 - mid)
            end

            # symmetric reflection
            for j in 1:seg_len
                xstar[j]            = x0[seg_len - j + 1]
                xstar[seg_len + j]  = x0[j]
                xstar[2seg_len + j] = x0[seg_len - j + 1]
            end

            # cumsum
            cs[1] = zero(T)
            for j in 1:3seg_len
                cs[j+1] = cs[j] + xstar[j]
            end

            # Hadamard differences via cumsum windows
            sq = 0.0
            for j in 0:(6m - 1)
                h1 = (cs[j+m+1]  - cs[j+1])   / m
                h2 = (cs[j+2m+1] - cs[j+m+1])  / m
                h3 = (cs[j+3m+1] - cs[j+2m+1]) / m
                sq += (h3 - 2h2 + h1)^2
            end
            dev_sum += sq / (6m)
        end

        push!(tau_out,  m * tau0)
        push!(dev_out,  sqrt(dev_sum / (6 * n_iter)))
        push!(neff_out, n_iter)
        push!(alpha_out, alpha[idx])
    end

    isempty(tau_out) && return DeviationResult(
        T[], T[], T[], Matrix{T}(undef,0,2), Int[], Int[], tau0, N, "htotdev", confidence)

    _make_result(tau_out, dev_out, alpha_out, neff_out, tau0, N, "htotdev", confidence)
end

"""
    mhtotdev(phase_data, tau0; mlist=nothing, confidence=0.683)

Modified Hadamard total deviation. Linear detrend, symmetric reflection on phase.
"""
function mhtotdev(phase_data::AbstractVector{T}, tau0::Real;
                  mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                  confidence::Real=0.683) where T<:Real
    x = validate_phase_data(phase_data)
    tau0 = validate_tau0(tau0)
    N = length(x)
    mlist = something(mlist, _default_mlist(N, 4))
    alpha = noise_id(x, mlist, "phase")

    tau_out  = Float64[]
    dev_out  = Float64[]
    neff_out = Int[]
    alpha_out = Float64[]

    for (k, m) in enumerate(mlist)
        nsubs = N - 4m + 1
        nsubs < 1 && continue

        seg_len = 3m + 1  # phase segment length

        total_sum = 0.0
        for n in 1:nsubs
            phase_seg = @view(x[n:n+3m])
            pd = detrend_linear(phase_seg)

            # symmetric reflection
            Lp = length(pd)
            ext_len = 3Lp
            # build ext: [rev(pd); pd; rev(pd)]
            # third differences on ext
            L3 = ext_len - 3m
            d3_sum = 0.0
            d3_count = 0
            for j in 1:L3
                # indices into ext: j, j+m, j+2m, j+3m
                # ext[i] = pd[Lp-i+1] for i<=Lp, pd[i-Lp] for Lp<i<=2Lp, pd[Lp-(i-2Lp)+1] for i>2Lp
                v1 = _ext_val(pd, Lp, j)
                v2 = _ext_val(pd, Lp, j+m)
                v3 = _ext_val(pd, Lp, j+2m)
                v4 = _ext_val(pd, Lp, j+3m)
                d3 = v1 - 3v2 + 3v3 - v4
                d3_sum += d3
                d3_count += 1
            end

            # Actually, we need moving-average of third differences, not just sum.
            # Let me do it properly with a cumsum approach but using the ext virtual indexing.

            # Build ext explicitly (small per-segment, pre-allocated approach)
            ext = Vector{T}(undef, ext_len)
            for j in 1:Lp
                ext[j]        = pd[Lp - j + 1]
                ext[Lp + j]   = pd[j]
                ext[2Lp + j]  = pd[Lp - j + 1]
            end

            L3 = ext_len - 3m
            if L3 <= 0
                continue
            end

            # third differences
            d3_vec = Vector{T}(undef, L3)
            for j in 1:L3
                d3_vec[j] = ext[j] - 3ext[j+m] + 3ext[j+2m] - ext[j+3m]
            end

            # moving average via cumsum
            if length(d3_vec) >= m
                S = cumsum([zero(T); d3_vec])
                n_avg = length(S) - m
                block_var = 0.0
                for j in 1:n_avg
                    a = S[j+m] - S[j]
                    block_var += a^2
                end
                block_var /= (n_avg * 6 * m^2)
            else
                block_var = 0.0
            end

            total_sum += block_var
        end

        push!(tau_out,  m * tau0)
        push!(dev_out,  sqrt(total_sum / nsubs) / (m * tau0))
        push!(neff_out, nsubs)
        push!(alpha_out, alpha[k])
    end

    isempty(tau_out) && return DeviationResult(
        T[], T[], T[], Matrix{T}(undef,0,2), Int[], Int[], tau0, N, "mhtotdev", confidence)

    _make_result(tau_out, dev_out, alpha_out, neff_out, tau0, N, "mhtotdev", confidence)
end

# helper for virtual symmetric-reflection indexing (unused after explicit ext, kept for reference)
@inline function _ext_val(pd, Lp, i)
    if i <= Lp
        return pd[Lp - i + 1]
    elseif i <= 2Lp
        return pd[i - Lp]
    else
        return pd[Lp - (i - 2Lp) + 1]
    end
end

# ── Unified Val(N) tuple dispatch for ALL deviation + time-error functions ───

for fn in (:adev, :mdev, :mhdev, :tdev, :ldev, :totdev, :hdev,
           :mtotdev, :htotdev, :mhtotdev,
           :tie, :mtie, :pdev, :theo1)
    @eval function $fn(phase_data::AbstractVector, tau0::Real, ::Val{N}; kwargs...) where {N}
        _tuple_return_check(N, $(string(fn)))
        return unpack_result($fn(phase_data, tau0; kwargs...), Val{N}())
    end
end
