# Kalman Filter Reference

Reference: MB23 Chapter 13, §13.5–§13.6 (pp. 265–276).

## State vector

3-state model (nstates = 3) — default. States: phase offset (s), fractional
frequency offset (f), frequency drift (d).

```
x = [s, f, d]ᵀ
```

2-state (s, f) and 5-state (s, f, d, sin-diurnal, cos-diurnal) variants supported.

## State transition matrix Φ

(MB23 §13.5.6, kinematic clock model):

```
Φ = [1  τ  τ²/2]
    [0  1   τ  ]
    [0  0   1  ]
```

Encodes constant-velocity/constant-acceleration kinematics over interval τ.

**Implementation** (`julia/src/filter.jl:build_phi!`):

```julia
Φ[1,2] = τ;  Φ[1,3] = τ^2/2;  Φ[2,3] = τ
```

**Status**: ✓ Verified against MB23 §13.5.6.

---

## Process noise matrix Q

Continuous-time power-law noise model integrated over [0, τ] (Van Loan integration;
MB23 §13.5.4; SP1065 noise model):

| Element | Formula | Noise source |
|---------|---------|-------------|
| Q[1,1] | `q_wfm·τ + q_rwfm·τ³/3 + q_irwfm·τ⁵/20` | WFM + RWFM + IRWFM on phase |
| Q[1,2] = Q[2,1] | `q_rwfm·τ²/2 + q_irwfm·τ⁴/8` | RWFM + IRWFM cross |
| Q[2,2] | `q_rwfm·τ + q_irwfm·τ³/3` | RWFM + IRWFM on freq |
| Q[1,3] = Q[3,1] | `q_irwfm·τ³/6` | IRWFM cross (phase-drift) |
| Q[2,3] = Q[3,2] | `q_irwfm·τ²/2` | IRWFM cross (freq-drift) |
| Q[3,3] | `q_irwfm·τ` | IRWFM on drift |
| Q[4,4] = Q[5,5] | `q_diurnal` | Diurnal (nstates=5 only) |

White PM noise (`q_wpm`) does not enter Q — it is the measurement noise R.

**Implementation** (`julia/src/filter.jl:build_Q!`): exact formulas as above.

**Status**: ✓ Verified. τ-power coefficients (τ, τ³/3, τ⁵/20, τ²/2, τ⁴/8, τ³/6)
are exact results of continuous-time integration — do not approximate.

---

## Measurement model H

Phase-only measurement:

```
H = [1, 0, 0, ...]    (1 × nstates row vector)
```

For nstates = 5, H[4] = sin(2πk/T), H[5] = cos(2πk/T) at time step k.

**Status**: ✓ Verified.

---

## Kalman update equations

Standard linear Kalman filter (MB23 §13.5; no Joseph form — simplified form matches
legacy):

```
Innovation:   ν = z_k - H·x̂⁻
Gain:         K = P⁻·Hᵀ / (H·P⁻·Hᵀ + R)
State update: x̂ = x̂⁻ + K·ν
Cov update:   P = (I - K·H)·P⁻
```

**Status**: ✓ Verified. Simplified (non-Joseph) covariance update matches legacy.

---

## PID steering

(MB23 §13.6; Masterclock internal convention):

```
sumx  += x[1]                          # integrate phase error
steer  = -g_p·x[1] - g_i·sumx - g_d·x[2]
```

Phase error (not frequency) is integrated. This is the Masterclock convention and
must not be changed.

Steering fed back into prediction step:

```
x_pred[1] += steer · τ    # phase correction
x_pred[2] += steer        # frequency correction
```

**Status**: ✓ Verified against legacy `filter.jl` lines 138–146 and MB23 §13.6.
