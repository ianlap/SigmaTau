# Time interval error functions for StabLab.jl

"""
    tie(data, tau0; mlist=nothing, confidence=0.683)

Time Interval Error RMS.  RMS of `|x[i+m] - x[i]|` over sliding windows.
"""
function tie(data::AbstractVector{T}, tau0::Real=1.0;
             mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
             confidence::Real=0.683) where T<:Real
    x = validate_phase_data(data)
    N = length(x)
    mlist = something(mlist, _default_mlist(N, 3))

    tau  = Vector{Float64}(undef, length(mlist))
    dev  = Vector{Float64}(undef, length(mlist))
    neff = Vector{Int}(undef, length(mlist))

    for (k, m) in enumerate(mlist)
        tau[k] = m * tau0
        n_pairs = N - m
        if n_pairs < 1
            dev[k] = NaN; neff[k] = 0; continue
        end
        neff[k] = n_pairs
        d = @view(x[1+m:N]) .- @view(x[1:N-m])
        dev[k] = sqrt(_meansq(d))
    end

    alpha = fill(-2, length(mlist))
    _make_result(tau, dev, alpha, neff, tau0, N, "tie", confidence)
end

"""
    mtie(data, tau0; mlist=nothing, confidence=0.683)

Maximum Time Interval Error.  Max peak-to-peak phase over all windows of length `m+1`.
"""
function mtie(data::AbstractVector{T}, tau0::Real=1.0;
              mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
              confidence::Real=0.683) where T<:Real
    x = validate_phase_data(data)
    N = length(x)
    mlist = something(mlist, _default_mlist(N, 3))

    tau  = Vector{Float64}(undef, length(mlist))
    dev  = Vector{Float64}(undef, length(mlist))
    neff = Vector{Int}(undef, length(mlist))

    for (k, m) in enumerate(mlist)
        tau[k] = m * tau0
        n_win = N - m
        if n_win < 1 || m + 1 > N
            dev[k] = NaN; neff[k] = 0; continue
        end
        neff[k] = n_win

        if m < 100
            mx = zero(T)
            for i in 1:n_win
                w = @view x[i:i+m]
                mx = max(mx, maximum(w) - minimum(w))
            end
            dev[k] = mx
        else
            cur_max = maximum(@view x[1:m+1])
            cur_min = minimum(@view x[1:m+1])
            mx = cur_max - cur_min
            for i in 2:n_win
                leaving  = x[i-1]
                entering = x[i+m]
                if leaving == cur_max
                    cur_max = maximum(@view x[i:i+m])
                elseif entering > cur_max
                    cur_max = entering
                end
                if leaving == cur_min
                    cur_min = minimum(@view x[i:i+m])
                elseif entering < cur_min
                    cur_min = entering
                end
                mx = max(mx, cur_max - cur_min)
            end
            dev[k] = mx
        end
    end

    alpha = fill(-2, length(mlist))
    _make_result(tau, dev, alpha, neff, tau0, N, "mtie", confidence)
end

"""
    pdev(data, tau0; mlist=nothing, confidence=0.683)

Parabolic deviation. At `m=1` equals ADEV; for `m>1` uses parabolic weighting
to remove linear drift.
"""
function pdev(data::AbstractVector{T}, tau0::Real=1.0;
              mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
              confidence::Real=0.683) where T<:Real
    x = validate_phase_data(data)
    N = length(x)
    mlist = something(mlist, _default_mlist(N, 3))

    tau  = Vector{Float64}(undef, length(mlist))
    dev  = Vector{Float64}(undef, length(mlist))
    neff = Vector{Int}(undef, length(mlist))

    for (k, m) in enumerate(mlist)
        tau[k] = m * tau0
        if m == 1
            n = N - 2
            if n < 1
                dev[k] = NaN; neff[k] = 0; continue
            end
            neff[k] = n
            ss = 0.0
            for i in 1:n
                d = x[i+2] - 2x[i+1] + x[i]
                ss += d^2
            end
            dev[k] = sqrt(ss / (2n)) / tau[k]
        else
            M = N - 2m
            if M < 1
                dev[k] = NaN; neff[k] = 0; continue
            end
            neff[k] = M
            ss = 0.0
            half_m1 = (m - 1) / 2.0
            for i in 0:M-1
                inner = 0.0
                for j in 0:m-1
                    inner += (half_m1 - j) * (x[i+j+1] - x[i+j+m+1])
                end
                ss += inner^2
            end
            var = 72ss / (M * m^4 * tau[k]^2)
            dev[k] = var < 0 ? NaN : sqrt(var)
        end
    end

    alpha = fill(-2, length(mlist))
    _make_result(tau, dev, alpha, neff, tau0, N, "pdev", confidence)
end

"""
    theo1(data, tau0; mlist=nothing, confidence=0.683)

THEO1 deviation. Requires even `m ≥ 10`. Extended averaging with improved confidence.
"""
function theo1(data::AbstractVector{T}, tau0::Real=1.0;
               mlist::Union{Nothing,AbstractVector{<:Integer}}=nothing,
               confidence::Real=0.683) where T<:Real
    x = validate_phase_data(data)
    N = length(x)

    if mlist === nothing
        mlist = filter(m -> m >= 10 && iseven(m), _default_mlist(N, 3))
        isempty(mlist) && (mlist = [10])
    else
        all(iseven, mlist) || error("THEO1 requires all m values to be even")
    end

    tau  = Vector{Float64}(undef, length(mlist))
    dev  = Vector{Float64}(undef, length(mlist))
    neff = Vector{Int}(undef, length(mlist))

    for (k, m) in enumerate(mlist)
        tau[k] = m * tau0
        if m > N - 1
            dev[k] = NaN; neff[k] = 0; continue
        end
        m_half = m ÷ 2
        total = 0.0
        terms = 0
        for i in 1:N-m
            inner = 0.0
            for d in 0:m_half-1
                w = 1.0 / (m_half - d)
                t1 = x[i] - x[i - d + m_half]
                t2 = x[i + m] - x[i + d + m_half]
                inner += w * (t1 + t2)^2
                terms += 1
            end
            total += inner
        end
        dev[k] = sqrt(total / (0.75 * (N - m) * tau[k]^2))
        neff[k] = terms
    end

    alpha = fill(-2, length(mlist))
    _make_result(tau, dev, alpha, neff, tau0, N, "theo1", confidence)
end

"""
    generate_itu_mask(mask_type, tau_range)

ITU-T mask limits for TIE/MTIE.  Supports `"G.811"` and `"G.812"`.
"""
function generate_itu_mask(mask_type::String, tau_range::AbstractVector{T}) where T<:Real
    mask = zeros(T, length(tau_range))
    if mask_type == "G.811"
        for (i, t) in enumerate(tau_range)
            mask[i] = t < 1 ? 3e-9 : (t < 100 ? 3e-9sqrt(t) : 3e-8)
        end
    elseif mask_type == "G.812"
        for (i, t) in enumerate(tau_range)
            mask[i] = t < 1 ? 1e-8 : (t < 100 ? 1e-8sqrt(t) : 1e-7)
        end
    else
        error("Mask type $mask_type not implemented")
    end
    mask
end
