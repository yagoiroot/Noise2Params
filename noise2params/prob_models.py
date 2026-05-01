"""
noise2params.prob_models — noise-event probability models.

Probability-model functions used by
:func:`noise2params.noise_image.event_noise_image_modeler_core`:

* :func:`theta_model`, :func:`theta_model_numba` — the parametric
  intensity-dependent leakage model  θ(λ) = c₁ + c₂·√λ + c₃·λ  (paper,
  Eq. for θ(λ); three-parameter form used throughout Noise2Params).
* :func:`pos_event_prob_gaussian`, :func:`neg_event_prob_gaussian` —
  closed-form Gaussian approximation to the per-pixel noise-event
  probabilities  P(Z₊ > 0)  and  P(Z₋ < 0).  Valid only for λ ≳ 10.
* :func:`pos_prob_saddle_bracket` / :func:`neg_prob_saddle_bracket` and
  their :func:`numba`-compiled counterparts
  :func:`pos_prob_saddle_bracket_numba` / :func:`neg_prob_saddle_bracket_numba`
  — the saddle-point approximation used for the primary synthetic dataset.
  The numba variants are used at inference-time and during per-pixel
  synthetic-image generation; the pure-NumPy variants are retained for
  reference and expose ``method={'newton', 'vec toms', 'pure vec'}``.
* :func:`pos_event_prob_vec_numba_2` — exact-Poisson vectorised
  probability.  Only the ``model='Poisson'`` branch of
  :func:`event_noise_image_modeler_core` uses it.
* :func:`kappa`, :func:`kappa_p`, :func:`kappa_pp` —
  cumulant-generating-function derivatives used by
  :func:`pos_prob_saddle`.

Mathematical notation follows the paper:
    * ``lam``   — mean photon count λ (per pixel, per integration window)
    * ``B``     — log-contrast threshold (paper parameter B)
    * ``theta`` — intensity-dependent leakage θ(λ)
See ``docs/PAPER_CROSSREF.md`` for a symbol-to-code mapping.

Dependencies
------------
Requires ``numpy``, ``scipy``, ``numba``, and the ``math``/``numpy`` helpers
re-exported by :mod:`noise2params.matlab_recreations` (``sqrt``, ``pi``,
``exp``) via the ``from matlab_recreations import *`` line below.
"""

import math
import numpy as np
import scipy
from numba import jit, njit, prange
import numba
from scipy.special import gammaln, logsumexp
from scipy.optimize import newton, root_scalar

# Re-export sqrt, pi, exp (used by the saddle-point functions below)
from noise2params.matlab_recreations import *  # noqa: F401,F403


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Gaussian approximation (paper Eq. for Gaussian noise-event probability)

def pos_event_prob_gaussian(lam=2.0, B=0.8, theta=1):
    """Closed-form Gaussian approximation to P(Z₊ > 0).

    Only valid for λ ≳ 10 (the Gaussian CLT regime).  Used as the
    ``model='Gaussian'`` branch of :func:`event_noise_image_modeler_core`.
    """
    p_plus = 0.5 + 0.5 * scipy.special.erf(((lam + theta) * (1 - np.exp(B))) /
                             np.sqrt(2 * lam * (1 + np.exp(2 * B))))
    return p_plus

def neg_event_prob_gaussian(lam=2.0, B=0.8, theta=1):
    """Closed-form Gaussian approximation to P(Z₋ < 0)."""
    p_plus = 0.5 - 0.5 * scipy.special.erf(((lam + theta) * (1 - np.exp(-B))) /
                             np.sqrt(2 * lam * (1 + np.exp(-2 * B))))
    return p_plus


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Leakage model theta(lambda)  —  paper Eq. for theta(lambda), 3-parameter form

def theta_model(x, fit_form, params):
    """Parametric model for the leakage term θ(λ).

    Only ``fit_form=4`` (three-parameter  θ = c₁ + c₂·√λ + c₃·λ ) is
    used by the Noise2Params reconstruction pipeline; the other fit forms
    are alternative parametric forms that may be used during parameter
    fitting (parameter fitting itself is not part of this public release).

    Parameters
    ----------
    x : float or ndarray
        Mean photon count λ.
    fit_form : int
        One of 1-8.  ``fit_form=4`` is the published form.
    params : sequence of float
        Parameters ``(c1, c2, c3)`` for fit_form=4; length varies for other
        forms.
    """
    if fit_form==1:
        c1, c2 = params
        return c1 + c2 * x ** 0.5
    if fit_form==2:
        c1, c2, c3 = params
        return c1 + c2 * (x ** c3)
    if fit_form==3:
        c1, c2, c3, c4, c5, c6, c7 = params
        return c1+c2*(x**0.5)+c3/(np.log(x+c4))+c5*x+c6/np.log(x+c7)
    if fit_form==4:
        c1, c2, c3 = params
        return c1 + c2* (x ** 0.5) + (c3*x)
    if fit_form==5:
        k1, k2, c1 = params
        return (c1*np.sqrt(x))+((k1*np.log(k2*x+1)-c1*np.sqrt(x))/(1+np.exp(8*(x-1))))
    if fit_form==6:
        k1, k2, c1, c2, c3 = params
        Alpha=k1*np.log(k2*x+1)
        Beta = c1 + c2 * (x ** c3)
        return Beta+(Alpha-Beta)/(1+np.exp(8*(x-1)))
    if fit_form==7:
        k1, k2, k3, c1, c2, c3 = params
        Alpha=k1*x*np.log(k2/np.abs(x*(x+k3)))
        Beta = c1 + c2 * (x ** c3)
        return Beta + (Alpha - Beta) / (1 + np.exp(10 * (x - .5)))
    if fit_form==8:
        c1, c2, c3, c4, c5, c6, c7, c8 ,c9  = params
        return c1 + c2 * (x ** c3) + c4 * x ** c5 + c6 * x ** c7 + c8 * x ** c9

@numba.njit
def theta_model_numba(x, params):
    """Numba-JIT version of :func:`theta_model`.

    Only the 3-parameter (``fit_form=4``) and 7-parameter (``fit_form=3``)
    forms are supported here; dispatched on ``len(params)``.
    """
    if len(params)==7:
        c1, c2, c3, c4, c5, c6, c7 = params
        return c1+c2*(x**0.5)+c3/(np.log(x+c4))+c5*x+c6/np.log(x+c7)
    if len(params)==3:
        c1, c2, c3 = params
        return c1 + c2 * (x ** 0.5) + (c3 * x)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Saddle-point approximation  —  cumulant generating function and derivatives

def kappa(s, lam, B, theta):
    """Cumulant-generating function κ(s).  Paper Eq. for κ(s)."""
    return lam*(np.exp(s)-1) \
         + lam*np.exp(B)*(np.exp(-s)-1) \
         + theta*(1-np.exp(B))*s

def kappa_p(s, lam, B, theta):
    """First derivative κ'(s)."""
    return lam*np.exp(s) \
         - lam*np.exp(B)*np.exp(-s) \
         + theta*(1-np.exp(B))

def kappa_pp(s, lam, B):
    """Second derivative κ''(s)."""
    return lam*np.exp(s) \
         + lam*np.exp(B)*np.exp(-s)   # note **plus** after the sign fix

def pos_prob_saddle(lam, B, theta):
    """Saddle-point approximation to P(Z₊ > 0) via scalar Newton.

    Solves φ'(s) = κ'(s) − 1/s = 0 and returns the saddle-point integral
    approximation.  Used for scalar inputs only; see
    :func:`pos_prob_saddle_bracket` and its numba variant for array use.
    """
    def phi_p(s): return kappa_p(s,lam,B,theta) - 1.0/s
    def phi_pp(s): return kappa_pp(s,lam,B) + 1.0/(s**2)
    s_star = newton(phi_p, x0=0.1, fprime=phi_pp, tol=1e-12, maxiter=1000)
    hess   = phi_pp(s_star)
    pref   = 1/(s_star*sqrt(2*pi*hess))
    return pref*np.exp(kappa(s_star,lam,B,theta))


def pos_prob_saddle_bracket(lam, B, theta, method='newton'):
    """
    Calculates positive probabilities using a saddle-point approximation method.

    This function is designed to compute probabilities for complex systems where direct
    calculation might be computationally expensive or impractical. It provides three
    methods: a scalar root-finding approach (``'vec toms'``), a fully vectorized
    bracketing/bisection approach (``'pure vec'``), and a vectorized Newton iteration
    (``'newton'``, default).  All three find the saddle point ``s_star`` of the
    cumulant generating function and return the resulting saddle-point probability.

    The saddle-point approximation involves solving a nonlinear equation to find ``s_star``,
    the point where a modified derivative of the cumulant generating function (``_phi_p``)
    equals zero. The function uses the second derivative at this point (``_phi_pp``) to
    construct the normalization term required for the final probability computation.

    Key mathematical ingredients include:
      * ``_kappa(s)``: The cumulant generating function.
      * ``_phi_p(s)``: A transformed derivative of ``_kappa(s)``, used to find the saddle
        point.
      * ``_phi_pp(s)``: The second derivative of ``_phi_p(s)``, which contributes to the
        final probability's normalization factor.

    The ``'vec toms'`` method performs scalar root-finding using the TOMS 748 algorithm
    and processes elements iteratively, which is more robust but slower. The ``'pure vec'``
    method, by contrast, fully vectorizes the process but at the cost of linear
    convergence.

    Currently, ``'newton'`` is slightly faster than ``'vec toms'``, which is slightly
    faster than ``'pure vec'``.  Newton is unstable past λ ~ 1000, but ``'vec toms'``
    remains stable.

    Parameters
    ----------
    lam : ndarray
        Parameter representing a rate or intensity in the system (mean photon count λ).
    B : ndarray
        Log-contrast threshold (paper parameter B).
    theta : ndarray
        Leakage term θ(λ).
    method : {'newton', 'vec toms', 'pure vec'}, optional
        Root-finding method for the saddle point.  Default ``'newton'``.

    Returns
    -------
    ndarray
        Saddle-point approximation to P(Z₊ > 0), shaped identically to the inputs.
    """

    def _kappa(s, lam, B, theta):
        return (lam*(np.exp(s)-1)
                + lam*(np.exp(-s*np.exp(B))-1)
                + theta*(1-np.exp(B))*s)
    # phi'(s)=kappa'(s)-1/s
    def _phi_p(s, lam, B, theta):
        return (lam*np.exp(s) - lam*np.exp(B)*np.exp(-s*np.exp(B)) +
                theta*(1-np.exp(B)) - 1/s)

    # phi''(s)=kappa''(s)-1/s^2
    def _phi_pp(s, lam, B):
        return lam*np.exp(s) + lam*np.exp(2*B)*np.exp(-s*np.exp(B)) + 1/s**2

    if method=='vec toms':
        def _find_root_toms748(l, B, th, a0=1e-12):
            """Scalar root via TOMS 748 with automatic upper bracket."""
            a = a0
            b_guess = 1.0
            while _phi_p(b_guess, l, B, th) <= 0:
                b_guess *= 4.0
            sol = root_scalar(_phi_p, args=(l, B, th),
                              bracket=[a, b_guess],
                              method='toms748',
                              xtol=1e-6, rtol=1e-6, maxiter=100)
            return sol.root

        lam_b, B_b, th_b = np.broadcast_arrays(lam, B, theta)
        out = np.empty_like(lam_b, dtype=float)

        it = np.nditer([lam_b, B_b, th_b, out],
                       flags=['multi_index', 'refs_ok'],
                       op_flags=[['readonly']] * 3 + [['writeonly']])

        for l, B, th, res in it:
            l = float(l);
            B = float(B);
            th = float(th)
            s_star = _find_root_toms748(l, B, th)
            hess = _phi_pp(s_star, l, B)
            pref = 1.0 / (s_star * sqrt(2.0 * pi * hess))
            e_kappa = np.exp(_kappa(s_star, l, B, th))
            res[...] = pref * e_kappa

        return out

    if method=='pure vec':
        # --- vectorised bracketing ------------------------------------------------
        a = np.full_like(lam, 1e-12, dtype=float)
        b = np.ones_like(lam, dtype=float)
        while True:
            mask = _phi_p(b, lam, B, theta) <= 0
            if not mask.any(): break
            b[mask] *= 2.0

        # --- vectorised bisection -------------------------------------------------
        max_iter=100
        for _ in range(max_iter):
            mid = 0.5 * (a + b)
            mask = _phi_p(mid, lam, B, theta) < 0
            a[mask] = mid[mask]
            b[~mask] = mid[~mask]

        s_star = 0.5 * (a + b)
        hess = _phi_pp(s_star, lam, B)
        pref = 1.0 / (s_star * np.sqrt(2.0 * np.pi * hess))
        return pref * np.exp(_kappa(s_star, lam, B, theta))

    if method == 'newton':
        lam, B, theta = np.broadcast_arrays(lam, B, theta)
        s = np.ones_like(lam, dtype=float)

        max_iter = 250
        tol = 1e-10

        for _ in range(max_iter):
            phi_p_val = _phi_p(s, lam, B, theta)
            phi_pp_val = _phi_pp(s, lam, B)
            phi_pp_val[phi_pp_val == 0] = 1e-22
            update = phi_p_val / phi_pp_val
            s -= update
            np.clip(s, 1e-12, None, out=s)
            if np.all(np.abs(update) < tol):
                break

        s_star = s
        hess = _phi_pp(s_star, lam, B)
        pref = 1.0 / (s_star * np.sqrt(2.0 * np.pi * hess))
        return pref * np.exp(_kappa(s_star, lam, B, theta))


def neg_prob_saddle_bracket(lam, B, theta, method='newton'):
    """Saddle-point approximation to P(Z₋ < 0).  Mirror of
    :func:`pos_prob_saddle_bracket` with sign flipped on B and on the
    prefactor."""

    def _kappa(s, lam, B, theta):
        return (lam*(np.exp(s)-1)
                + lam*(np.exp(-s*np.exp(-B))-1)
                + theta*(1-np.exp(-B))*s)
    def _phi_p(s, lam, B, theta):
        return (lam*np.exp(s) - lam*np.exp(-B)*np.exp(-s*np.exp(-B)) +
                theta*(1-np.exp(-B)) - 1/s)
    def _phi_pp(s, lam, B):
        return lam*np.exp(s) + lam*np.exp(-2*B)*np.exp(-s*np.exp(-B)) + 1/s**2

    if method=='vec toms':
        def _find_root_toms748(l, B, th, a0=-1e-12):
            a = a0
            b_guess = -1.0
            while _phi_p(b_guess, l, B, th) > 0:
                b_guess *= 4.0
            sol = root_scalar(_phi_p, args=(l, B, th),
                              bracket=[b_guess, a],
                              method='toms748', xtol=1e-7, rtol=1e-7, maxiter=200)
            return sol.root

        lam_b, B_b, th_b = np.broadcast_arrays(lam, B, theta)
        out = np.empty_like(lam_b, dtype=float)

        it = np.nditer([lam_b, B_b, th_b, out],
                       flags=['multi_index', 'refs_ok'],
                       op_flags=[['readonly']] * 3 + [['writeonly']])

        for l, B, th, res in it:
            l = float(l); B = float(B); th = float(th)
            s_star = _find_root_toms748(l, B, th)
            hess = _phi_pp(s_star, l, B)
            pref = -1.0 / (s_star * sqrt(2.0 * pi * hess))
            e_kappa = np.exp(_kappa(s_star, l, B, th))
            res[...] = pref * e_kappa

        return out

    if method == 'newton':
        lam, B, theta = np.broadcast_arrays(lam, B, theta)
        s = -np.ones_like(lam, dtype=float)

        max_iter = 250
        tol = 1e-10

        for _ in range(max_iter):
            phi_p_val = _phi_p(s, lam, B, theta)
            phi_pp_val = _phi_pp(s, lam, B)
            phi_pp_val[phi_pp_val == 0] = 1e-22
            update = phi_p_val / phi_pp_val
            s -= update
            s[s > -1e-12] = -1e-12
            if np.all(np.abs(update) < tol):
                break

        s_star = s
        hess = _phi_pp(s_star, lam, B)
        pref = -1.0 / (s_star * np.sqrt(2.0 * np.pi * hess))
        return pref * np.exp(_kappa(s_star, lam, B, theta))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Numba-parallel versions of the saddle-point solvers (primary runtime path).

@njit(parallel=True, fastmath=True)
def pos_prob_saddle_bracket_numba(lam, B, theta, max_iter=250, tol=1e-10):
    """Numba-accelerated parallel Newton saddle-point solver for P(Z₊ > 0).

    This is the hot path for synthetic noise-image generation over a
    (1280, 720) pixel array: ~200 µs per pixel in pure NumPy becomes
    <1 µs per pixel after Numba + parallelism.  See
    :func:`pos_prob_saddle_bracket` for the method description.
    """
    lam_flat = lam.ravel()
    B_flat = B.ravel()
    theta_flat = theta.ravel()

    n = lam_flat.size
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        l = lam_flat[i]
        b = B_flat[i]
        th = theta_flat[i]

        s = 1.0

        for _ in range(max_iter):
            exp_s = np.exp(s)
            exp_b = np.exp(b)
            exp_minus_s_exp_b = np.exp(-s * exp_b)

            phi_p_val = (l * exp_s - l * exp_b * exp_minus_s_exp_b +
                         th * (1 - exp_b) - 1 / s)
            phi_pp_val = l * exp_s + l * exp_b * exp_b * exp_minus_s_exp_b + 1 / (s * s)

            if phi_pp_val == 0:
                phi_pp_val = 1e-22

            update = phi_p_val / phi_pp_val
            s -= update

            if s < 1e-12:
                s = 1e-12

            if abs(update) < tol:
                break

        s_star = s
        hess = l * np.exp(s_star) + l * np.exp(2 * b) * np.exp(-s_star * np.exp(b)) + 1 / (s_star * s_star)
        kappa_ = (l * (np.exp(s_star) - 1) +
                 l * (np.exp(-s_star * np.exp(b)) - 1) +
                 th * (1 - np.exp(b)) * s_star)

        pref = 1.0 / (s_star * np.sqrt(2.0 * np.pi * hess))
        result[i] = pref * np.exp(kappa_)

    return result.reshape(lam.shape)


@njit(parallel=True, fastmath=True)
def neg_prob_saddle_bracket_numba(lam, B, theta, max_iter=250, tol=1e-10):
    """Numba-accelerated parallel Newton saddle-point solver for P(Z₋ < 0)."""
    lam_flat = lam.ravel()
    B_flat = B.ravel()
    theta_flat = theta.ravel()

    n = lam_flat.size
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        l = lam_flat[i]
        b = B_flat[i]
        th = theta_flat[i]

        s = -1.0

        for _ in range(max_iter):
            exp_s = np.exp(s)
            exp_neg_b = np.exp(-b)
            exp_minus_s_exp_neg_b = np.exp(-s * exp_neg_b)

            phi_p_val = (l * exp_s - l * exp_neg_b * exp_minus_s_exp_neg_b +
                         th * (1 - exp_neg_b) - 1 / s)
            phi_pp_val = l * exp_s + l * np.exp(-2 * b) * exp_minus_s_exp_neg_b + 1 / (s * s)

            if phi_pp_val == 0:
                phi_pp_val = 1e-22

            update = phi_p_val / phi_pp_val
            s -= update

            if s > -1e-12:
                s = -1e-12

            if abs(update) < tol:
                break

        s_star = s
        hess = l * np.exp(s_star) + l * np.exp(-2 * b) * np.exp(-s_star * np.exp(-b)) + 1 / (s_star * s_star)
        kappa_ = (l * (np.exp(s_star) - 1) +
                 l * (np.exp(-s_star * np.exp(-b)) - 1) +
                 th * (1 - np.exp(-b)) * s_star)

        pref = -1.0 / (s_star * np.sqrt(2.0 * np.pi * hess))
        result[i] = pref * np.exp(kappa_)

    return result.reshape(lam.shape)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Exact Poisson (vectorised, single-sum form) — used by the ``model='Poisson'``
# branch of event_noise_image_modeler_core.  NOTE: no Poisson-model synthetic
# dataset ships with the paper (see the repository README and docs/DATA.md).

@njit(parallel=True)
def _pos_event_single_sum_numba(lam_flat, B_flat, theta_flat,
                                n_max, n_0_max):
    """Parallel Numba kernel: single-sum exact Poisson P(Z₊ > 0)."""
    P = lam_flat.shape[0]
    out = np.empty(P, dtype=np.float64)

    max_n = n_max
    if n_0_max > max_n:
        max_n = n_0_max

    for k in prange(P):
        lam_k = lam_flat[k]
        B_k   = B_flat[k]
        th_k  = theta_flat[k]

        if lam_k <= 0.0:
            out[k] = 0.0
            continue

        A      = math.exp(B_k)       # e^B
        inv_exp = 1.0 - A            # 1 - e^B

        # Poisson pmf for N and N_0 (same lambda)
        pmf = np.empty(max_n, dtype=np.float64)
        p0 = math.exp(-lam_k)
        pmf[0] = p0
        for n in range(1, max_n):
            pmf[n] = pmf[n-1] * lam_k / n

        # Truncated tail for N up to n_max-1
        tail = np.empty(n_max + 1, dtype=np.float64)
        tail[n_max] = 0.0
        s = 0.0
        for n in range(n_max - 1, -1, -1):
            s += pmf[n]
            tail[n] = s

        # Single sum over n0
        prob = 0.0
        for i in range(n_0_max):
            lb = math.floor(A * i - th_k * inv_exp) + 1
            if lb < 0:
                lb = 0
            if lb >= n_max:
                continue
            prob += pmf[i] * tail[lb]

        out[k] = prob

    return out


def pos_event_prob_vec_numba_2(lam, B, theta, n_max=300, n_0_max=300):
    """Exact-Poisson P(Z₊ > 0), vectorised via :func:`_pos_event_single_sum_numba`.

    Implements the paper's exact Poisson formula (paper Eq. for the exact
    Poisson noise-event probability) via a single-sum truncation with
    caps ``n_max`` and ``n_0_max`` on the two Poisson tails.
    """
    lam_b, B_b, th_b = np.broadcast_arrays(lam, B, theta)
    shape = lam_b.shape

    lam_flat   = lam_b.ravel().astype(np.float64)
    B_flat     = B_b.ravel().astype(np.float64)
    theta_flat = th_b.ravel().astype(np.float64)

    result_flat = _pos_event_single_sum_numba(
        lam_flat, B_flat, theta_flat,
        n_max, n_0_max
    )

    result = result_flat.reshape(shape)
    return result.item() if result.shape == () else result
