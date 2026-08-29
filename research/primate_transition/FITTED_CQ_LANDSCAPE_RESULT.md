# Fitted two-coordinate effective landscape

## Status

Frozen verdict:

- `SINGLE_WELL_FIELD_DEFORMATION_SUPPORTED`
- `PHASE_WIDE_SHARED_DEFORMATION_NOT_PREDICTIVE`
- `CROSS_COORDINATE_ONE_STEP_COUPLING_NOT_SUPPORTED`
- `BIFURCATION_NOT_IDENTIFIED`

The fitted object is a two-stage kinetic maximum-entropy model over

- structured access: `s_c = +1` for CE/Cross and `-1` otherwise;
- recursive orientation: `s_q = +1` for CE and `-1` for Cross, conditional on structured access.

For either coordinate,

```math
P(s_t=+1\mid s_{t-1})=\sigma(2(h+J s_{t-1})).
```

The corresponding mean-field effective free energy is

```math
V(m)= -\frac{J}{2}m^2-hm
+\frac{1+m}{2}\log\frac{1+m}{2}
+\frac{1-m}{2}\log\frac{1-m}{2},
```

with stationary condition

```math
m=\tanh(Jm+h).
```

For the diagonal two-coordinate fit,

```math
V(m_c,q)=V_c(m_c)+V_q(q),
```

where `m_c = 2 c_struct - 1` and `q = q_rec`.

## Static fields from the published simple-analysis counts

Jeffreys-smoothed fields:

| cohort | h_c access | h_q orientation |
|---|---:|---:|
| Macaque E1 | +0.0775 | -0.0306 |
| Macaque E2 | -0.3299 | +0.4658 |
| Macaque E3 | +0.0845 | +0.1065 |
| U.S. children | +0.2065 | +0.4727 |
| Tsimane' adults | +0.8565 | +0.5126 |
| U.S. adults | +3.0879 | +1.3053 |

The central matched-orientation result is

```math
h_q^{child}-h_q^{macaque,E2}\approx 0.0069,
```

while the access field differs by about

```math
h_c^{child}-h_c^{macaque,E2}\approx 0.5364.
```

Thus extra exposure can make the macaque CE-vs-Cross energy gap essentially child-like without making access to the structured subspace child-like.

## Kinetic parameters from the public raw trial sequences

| cohort | h_c dyn | J_c | h_q dyn | J_q | max abs J |
|---|---:|---:|---:|---:|---:|
| Macaque E1 | +0.080 | +0.150 | -0.043 | -0.215 | 0.215 |
| Macaque E2 | -0.261 | +0.316 | +0.484 | -0.044 | 0.316 |
| Macaque E3 | +0.070 | +0.240 | +0.108 | -0.020 | 0.240 |
| U.S. children | +0.230 | +0.627 | +0.279 | +0.436 | 0.627 |
| Tsimane' adults | +0.614 | +0.621 | +0.389 | +0.411 | 0.621 |
| U.S. adults | +1.088 | +0.769 | +1.029 | +0.824 | 0.824 |

All fitted self-couplings satisfy

```math
|J|<1.
```

For the fitted mean-field free energy,

```math
V''(m)=\frac{1}{1-m^2}-J.
```

Since `1/(1-m^2) >= 1`, every fitted diagonal potential is strictly convex for all `m in (-1,1)` whenever `J<1`. Therefore the fitted model has one well per coordinate and does not undergo a mean-field pitchfork/symmetry-breaking bifurcation.

The largest fitted coupling is about `0.824` (U.S.-adult orientation), still below the `J=1` critical threshold.

## Model selection

Within each cohort we compared:

1. static fields only;
2. diagonal kinetic model with self-persistence `J_c, J_q`;
3. a one-step cross-coupled model adding `J_cq, J_qc`.

BIC chooses:

- Macaque E1: static;
- Macaque E2: diagonal kinetic;
- Macaque E3: diagonal kinetic;
- U.S. children: diagonal kinetic;
- Tsimane' adults: diagonal kinetic;
- U.S. adults: diagonal kinetic.

The cross-coupled model is never selected; its BIC penalty relative to the best model is approximately 8--14 points across these cohorts. Thus an additional one-step `c <-> q` dynamic coupling is not required by these data beyond the hierarchical conditioning of `q` on structured access.

## E1 -> E2 mechanism test

With macaque subject fixed effects, the in-sample BICs are:

| model | BIC |
|---|---:|
| field-only phase shift | 725.59 |
| no shared phase shift | 726.83 |
| barrier-only phase shift | 734.04 |
| field + barrier phase shift | 735.22 |

The field-only advantage is only `Delta BIC = 1.24`, i.e. weak.

More importantly, leave-one-session-out prediction favors the no-shared-phase model:

| model | held-out log loss |
|---|---:|
| no shared phase shift | 0.6942 |
| barrier-only | 0.6962 |
| field-only | 0.7346 |
| field + barrier | 0.7447 |

Therefore the aggregate E1 -> E2 displacement should **not** be treated as a species-wide deterministic phase deformation. It is strongly subject-contingent; the earlier individual analysis already identified Coltrane's tail-embedded excursion.

## Developmental placement

In the public 33-child raw subset, Jeffreys-smoothed participant fields show:

- age vs access field: Spearman rho about `0.401`, p about `0.021`;
- age vs orientation field: rho about `0.109`, p about `0.546`.

Age-bin means:

| age bin | mean h_c | mean h_q |
|---|---:|---:|
| 3 to <4 | 0.059 | 0.402 |
| 4 to 5 | 0.790 | 0.421 |

Thus the developmental motion in this subset is predominantly along the access-field direction, not the CE-vs-Cross orientation field.

## Scientific interpretation

The current data support a decomposition into at least three costs:

```math
C_access, C_selection, C_stability.
```

The strongest current interpretation is:

> Additional evidence can rotate macaque strategy preference toward the same CE-vs-Cross orientation seen in children, while humans show much stronger access and persistence/stability. The fitted finite-data kinetic landscape remains single-well and does not require a true bifurcation.

This is an effective statistical landscape, not a thermodynamic free energy and not evidence of a biological critical point.
