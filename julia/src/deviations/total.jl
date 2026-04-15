# deviations/total.jl — Total deviations (TOTDEV, MTOTDEV, HTOTDEV, MHTOTDEV)

# ── TOTDEV ────────────────────────────────────────────────────────────────────

"""
    totdev(x, tau0; m_list=nothing, data_type=:phase) → DeviationResult

Total deviation. Extends data by symmetric reflection to reduce endpoint effects,
then computes overlapping second differences. SP1065 §5.2.11 Eq. 25.

Algorithm: linear detrend, build 3N-4 extended array by symmetric reflection
about each endpoint, compute second differences at all N center positions.

# Arguments
- `x`:         phase data vector (seconds), length N ≥ 5
- `tau0`:      sampling interval (seconds)
- `m_list`:    averaging factors; auto-generated (octave-spaced) when `nothing`
- `data_type`: `:phase` (default) or `:freq` (fractional-frequency samples)

# Returns
`DeviationResult` with `method == "totdev"`. Bias-corrected via engine.
"""
function totdev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    params = DevParams(
        "totdev",   # name
        2,          # min_factor: N/m ≥ 2
        2,          # d: second-difference order
        m -> m,     # F_fn: unmodified
        0,          # dmin for noise_id
        2,          # dmax for noise_id
    )
    return engine(x, tau0, m_list, _totdev_kernel, params; data_type)
end

# Kernel: linear detrend + symmetric reflection, then overlapping second differences.
# SP1065 §5.2.11 Eq. 25: phase-form denominator is 2τ²(N-2), not the number of overlap samples.
function _totdev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real, x_cs::AbstractVector{<:Real})
    N = length(x)
    T = eltype(x)
    xd = copy(x)
    detrend_linear!(xd)

    # Symmetric reflection: [rev(xd[2:N-1]) mirrored about xd[1]; xd; rev mirrored about xd[N]]
    # Extended array length: (N-2) + N + (N-2) = 3N-4
    x_star = Vector{T}(undef, 3N - 4)
    # Left part: 2xd[1] - xd[2:N-1]
    for i in 1:N-2
        x_star[i] = 2xd[1] - xd[i+1]
    end
    # Center part: xd
    for i in 1:N
        x_star[N-2+i] = xd[i]
    end
    # Right part: 2xd[end] - xd[N-1:-1:2]
    for i in 1:N-2
        x_star[2N-2+i] = 2xd[N] - xd[N-i]
    end

    off = N - 2   # x_star[off+i] == xd[i] for i in 1:N

    D = 0.0; count = 0
    for i in 1:N
        lo = off + i; hi = off + i + 2m
        hi > length(x_star) && continue
        d2 = x_star[hi] - 2x_star[off + i + m] + x_star[lo]
        D += d2^2; count += 1
    end
    count == 0 && return (NaN, 0)
    # SP1065 §5.2.11 Eq. 25: phase form uses 2τ²(N-2)
    var = D / (2 * (N - 2) * (m * tau0)^2)
    return (var, count)
end

# ── MTOTDEV ───────────────────────────────────────────────────────────────────

"""
    mtotdev(x, tau0; m_list=nothing, data_type=:phase) → DeviationResult

Modified total deviation. For each N-3m+1 subsegment of length 3m,
applies half-average detrend, symmetric reflection, then computes modified
ADEV (cumsum second differences). SP1065 §5.12.

# Arguments
- `x`:         phase data vector (seconds), length N ≥ 4
- `tau0`:      sampling interval (seconds)
- `m_list`:    averaging factors; auto-generated (octave-spaced) when `nothing`
- `data_type`: `:phase` (default) or `:freq` (fractional-frequency samples)

# Returns
`DeviationResult` with `method == "mtotdev"`.
"""
function mtotdev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    params = DevParams(
        "mtotdev",  # name
        3,          # min_factor: N/m ≥ 3
        2,          # d: second-difference order
        m -> 1,     # F_fn: modified (F=1)
        0,          # dmin for noise_id
        2,          # dmax for noise_id
    )
    return engine(x, tau0, m_list, _mtotdev_kernel, params; data_type)
end

# Kernel: accumulates variance over all N-3m+1 subsegments.
# Each segment: half-average detrend → symmetric reflection → cumsum second-diff.
function _mtotdev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real, x_cs::AbstractVector{<:Real})
    N = length(x)
    nsubs = N - 3m + 1
    nsubs < 1 && return (NaN, 0)

    T = eltype(x)
    seg_len = 3m
    seq     = Vector{T}(undef, seg_len)
    seq_det = Vector{T}(undef, seg_len)
    ext     = Vector{T}(undef, 3seg_len)
    cs      = Vector{T}(undef, 3seg_len + 1)

    outer_sum = 0.0
    for n in 1:nsubs
        copyto!(seq, 1, x, n, seg_len)

        # Half-average detrend (ported from legacy mtotdev)
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

        # Symmetric reflection: [rev(seq_det); seq_det; rev(seq_det)]
        for j in 1:seg_len
            ext[j]            = seq_det[seg_len - j + 1]
            ext[seg_len + j]  = seq_det[j]
            ext[2seg_len + j] = seq_det[seg_len - j + 1]
        end

        # Cumsum of extended sequence
        cs[1] = zero(T)
        for j in 1:3seg_len
            cs[j+1] = cs[j] + ext[j]
        end

        # Second differences via cumsum windows: sum over 6m positions
        block_sum = 0.0
        for j in 0:(6m - 1)
            a1 = (cs[j+m+1]  - cs[j+1])   / m
            a2 = (cs[j+2m+1] - cs[j+m+1]) / m
            a3 = (cs[j+3m+1] - cs[j+2m+1]) / m
            d2 = a3 - 2a2 + a1
            block_sum += d2^2
        end
        outer_sum += block_sum / (6m)
    end

    var = outer_sum / (2 * (m * tau0)^2 * nsubs)
    return (var, nsubs)
end

# ── HTOTDEV ───────────────────────────────────────────────────────────────────

"""
    htotdev(x, tau0; m_list=nothing, data_type=:phase) → DeviationResult

Hadamard total deviation. Uses frequency data segments with half-average
detrend + symmetric reflection + Hadamard cumsum differences. SP1065 §5.13.

**Critical**: when m == 1, the hdev algorithm (third differences on phase) is
used instead of the total deviation algorithm. (Documented in CLAUDE.md.)

# Arguments
- `x`:         phase data vector (seconds), length N ≥ 5
- `tau0`:      sampling interval (seconds)
- `m_list`:    averaging factors; auto-generated (octave-spaced) when `nothing`
- `data_type`: `:phase` (default) or `:freq` (fractional-frequency samples)

# Returns
`DeviationResult` with `method == "htotdev"`. Bias-corrected via engine.
"""
function htotdev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    params = DevParams(
        "htotdev",  # name
        3,          # min_factor: N/m ≥ 3 (uses frequency data length N-1)
        3,          # d: third-difference order (Hadamard)
        m -> m,     # F_fn: unmodified
        0,          # dmin for noise_id
        2,          # dmax for noise_id
    )
    return engine(x, tau0, m_list, _htotdev_kernel, params; data_type)
end

# Kernel: m==1 uses hdev third-differences; m>1 uses htotdev frequency-segment algorithm.
# Legacy htotdev reference: uses y=diff(x)/tau0, segments of length 3m, cumsum Hadamard diffs.
function _htotdev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real, x_cs::AbstractVector{<:Real})
    N = length(x)
    if m == 1
        # Use hdev formula directly (third differences on phase) — CLAUDE.md critical rule
        L = N - 3
        L <= 0 && return (NaN, 0)
        d3 = @view(x[4:end]) .- 3 .* @view(x[3:end-1]) .+
             3 .* @view(x[2:end-2]) .- @view(x[1:L])
        v = sum(abs2, d3) / (L * 6 * tau0^2)
        return (v, L)
    end

    y  = diff(x) ./ tau0   # fractional frequency data
    Ny = length(y)
    n_iter = Ny - 3m + 1
    n_iter < 1 && return (NaN, 0)

    T = eltype(x)
    seg_len = 3m
    xs    = Vector{T}(undef, seg_len)
    x0    = Vector{T}(undef, seg_len)
    xstar = Vector{T}(undef, 3seg_len)
    cs    = Vector{T}(undef, 3seg_len + 1)

    dev_sum = 0.0
    for i in 0:(n_iter - 1)
        copyto!(xs, 1, y, i + 1, seg_len)

        # Half-average detrend on frequency segment (ported from legacy htotdev)
        hi       = floor(Int, seg_len / 2)
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

        # Symmetric reflection: [rev(x0); x0; rev(x0)]
        for j in 1:seg_len
            xstar[j]            = x0[seg_len - j + 1]
            xstar[seg_len + j]  = x0[j]
            xstar[2seg_len + j] = x0[seg_len - j + 1]
        end

        # Cumsum of extended frequency sequence
        cs[1] = zero(T)
        for j in 1:3seg_len
            cs[j+1] = cs[j] + xstar[j]
        end

        # Hadamard differences via cumsum windows (6m positions)
        sq = 0.0
        for j in 0:(6m - 1)
            h1 = (cs[j+m+1]  - cs[j+1])    / m
            h2 = (cs[j+2m+1] - cs[j+m+1])  / m
            h3 = (cs[j+3m+1] - cs[j+2m+1]) / m
            sq += (h3 - 2h2 + h1)^2
        end
        dev_sum += sq / (6m)
    end

    # Return variance: (dev_sum / (6 * n_iter)) where dev_sum already contains
    # per-segment averages (sq/6m). The result is then sqrt'd by the engine.
    # Legacy line 390: dev[k] = sqrt(dev_sum / (6 * n_iter))
    var = dev_sum / (6 * n_iter)
    return (var, n_iter)
end

# ── MHTOTDEV ──────────────────────────────────────────────────────────────────

"""
    mhtotdev(x, tau0; m_list=nothing, data_type=:phase) → DeviationResult

Modified Hadamard total deviation. For each N-4m+1 subsegment of phase length
3m+1, applies linear detrend, symmetric reflection, third differences, and
moving average. SP1065 / FCS 2001.

# Arguments
- `x`:         phase data vector (seconds), length N ≥ 5
- `tau0`:      sampling interval (seconds)
- `m_list`:    averaging factors; auto-generated (octave-spaced) when `nothing`
- `data_type`: `:phase` (default) or `:freq` (fractional-frequency samples)

# Returns
`DeviationResult` with `method == "mhtotdev"`.
"""
function mhtotdev(
    x         :: AbstractVector{<:Real},
    tau0      :: Real;
    m_list    :: Union{Nothing, AbstractVector{<:Integer}} = nothing,
    data_type :: Symbol = :phase,
)
    params = DevParams(
        "mhtotdev", # name
        4,          # min_factor: N/m ≥ 4
        3,          # d: third-difference order (Hadamard)
        m -> 1,     # F_fn: modified (F=1)
        0,          # dmin for noise_id
        2,          # dmax for noise_id
    )
    return engine(x, tau0, m_list, _mhtotdev_kernel, params; data_type)
end

# Kernel: linear detrend per phase segment, symmetric reflection, third diffs + moving avg.
# Ported from legacy mhtotdev. Variance = total_sum / (nsubs * (m*tau0)^2).
function _mhtotdev_kernel(x::AbstractVector{<:Real}, m::Int, tau0::Real, x_cs::AbstractVector{<:Real})
    m >= 1 || throw(ArgumentError("averaging factor m must be >= 1"))
    N = length(x)
    nsubs = N - 4m + 1
    nsubs < 1 && return (NaN, 0)

    T = eltype(x)
    Lp = 3m + 1   # phase segment length
    ext_len = 3Lp
    L3 = ext_len - 3m

    # Preallocate buffers outside the subsegment loop
    pd     = Vector{T}(undef, Lp)
    ext    = Vector{T}(undef, ext_len)
    d3_vec = Vector{T}(undef, L3)
    S      = Vector{T}(undef, L3 + 1)

    total_sum = 0.0
    for n in 1:nsubs
        copyto!(pd, 1, x, n, Lp)
        detrend_linear!(pd)

        # Symmetric reflection: [rev(pd); pd; rev(pd)]
        for j in 1:Lp
            ext[j]        = pd[Lp - j + 1]
            ext[Lp + j]   = pd[j]
            ext[2Lp + j]  = pd[Lp - j + 1]
        end

        # Third differences on extended array
        for j in 1:L3
            d3_vec[j] = ext[j] - 3ext[j+m] + 3ext[j+2m] - ext[j+3m]
        end

        # Moving average via cumsum (length-m windows over third differences)
        S[1] = zero(T)
        for j in 1:L3
            S[j+1] = S[j] + d3_vec[j]
        end
        
        n_avg = L3 + 1 - m
        if n_avg > 0
            block_var = 0.0
            for j in 1:n_avg
                a = S[j+m] - S[j]
                block_var += a^2
            end
            block_var /= (n_avg * 6.0 * Float64(m)^2)
        else
            block_var = 0.0
        end

        total_sum += block_var
    end

    var = total_sum / (nsubs * (m * tau0)^2)
    return (var, nsubs)
end
