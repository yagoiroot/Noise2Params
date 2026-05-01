"""
noise2params.matlab_recreations — saddle-point / Gaussian reference
implementations plus a star-import of ``sqrt``, ``pi``, ``exp`` from
:mod:`math`.

This module is imported with ``from noise2params.matlab_recreations import *``
by :mod:`noise2params.prob_models` so that the ``sqrt``, ``pi``, ``exp``
names used in the saddle-point prefactor there resolve without needing
``math.`` prefixes.

The remaining functions in this module are standalone reference
implementations of the saddle-point and Gaussian approximations (adapted
from MATLAB prototypes) and are not on the reconstruction pipeline's
call path; they are retained for verifiability against the paper's math.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize
from scipy.special import erf
from scipy.stats import norm
from math import sqrt, pi, exp
import mpmath as mpm
import mpmath as mp


def compute_probabilities_saddle(lambda_val, epsilon, b, g=1, f=1):
    """
    For a given lambda_val, epsilon, b, g, and a scaling factor f,
    compute the saddlepoint approximations for the probability of positive
    events and negative events.
    """

    # Define the cumulant generating function (kappa) and its first and second derivatives.
    def kappa(s, eps):
        # kappa(s, eps) = lambda*(exp(g*s)-1) + lambda*(exp(-exp(eps)*g*s)-1) + b*(1-exp(eps))*s
        return lambda_val * (np.exp(g * s) - 1) + \
            lambda_val * (np.exp(-np.exp(eps) * g * s) - 1) + \
            b * (1 - np.exp(eps)) * s

    def kappa_prime(s, eps):
        # kappa'(s, eps) = lambda*g*exp(g*s) - lambda*g*exp(eps)*exp(-exp(eps)*g*s) + b*(1-exp(eps))
        return lambda_val * g * np.exp(g * s) - \
            lambda_val * g * np.exp(eps) * np.exp(-np.exp(eps) * g * s) + \
            b * (1 - np.exp(eps))

    def kappa_prime2(s, eps):
        # kappa''(s, eps) = lambda*g^2*exp(g*s) + lambda*g^2*exp(2*eps)*exp(-exp(eps)*g*s)
        return lambda_val * g ** 2 * np.exp(g * s) + \
            lambda_val * g ** 2 * np.exp(2 * eps) * np.exp(-np.exp(eps) * g * s)

    # ------------------------------
    # For positive events:
    # Solve for the saddlepoint s* such that kappa_prime(s*, epsilon)=0.
    try:
        s_star_pos = newton(lambda s: kappa_prime(s, epsilon),
                            x0=0.0,
                            fprime=lambda s: kappa_prime2(s, epsilon),
                            tol=1e-15,
                            maxiter=10000)
    except RuntimeError:
        s_star_pos = np.nan

    # Use the saddlepoint approximation formula:
    # probability_plus = [1/sqrt(2*pi*kappa''(s*))] * exp(kappa(s*)) / s* * (1 + 1/(kappa''(s*)*s*^2))
    probability_plus = (1.0 / np.sqrt(2 * pi * kappa_prime2(s_star_pos, epsilon))) * \
                       np.exp(kappa(s_star_pos, epsilon)) / s_star_pos * \
                       (1 + 1.0 / (kappa_prime2(s_star_pos, epsilon) * s_star_pos ** 2))

    # ------------------------------
    # For negative events:
    # Modify epsilon by a factor f.
    epsilon_neg = epsilon * f
    try:
        s_star_neg = newton(lambda s: kappa_prime(s, -epsilon_neg),
                            x0=0.0,
                            fprime=lambda s: kappa_prime2(s, -epsilon_neg),
                            tol=1e-15,
                            maxiter=10000)
    except RuntimeError:
        s_star_neg = np.nan

    # Note the minus sign in the denominator
    probability_neg = (1.0 / np.sqrt(2 * pi * kappa_prime2(s_star_neg, -epsilon_neg))) * \
                      np.exp(kappa(s_star_neg, -epsilon_neg)) / (-s_star_neg) * \
                      (1 + 1.0 / (kappa_prime2(s_star_neg, -epsilon_neg) * s_star_neg ** 2))

    return probability_plus, probability_neg


def compute_probabilities_saddle_2(lambda_val, epsilon, b, g=1, f=1):
    """
    Potentially achieves better numerical stability than compute_probabilities_saddle.
    confirmed working at least where compute_probabilities_saddle works.
    For a given lambda_val, epsilon, b, g, and a scaling factor f,
    compute the saddlepoint approximations for the probability of positive
    events and negative events.
    """

    # print(f'lambda_val: {lambda_val}, epsilon: {epsilon}, b: {b}')

    # if lambda_val <= 0.0179 and b>40:
    #     return 0.0, 0.0

    def stable_diff_exp(A, B):
        # Compute exp(A) - exp(B) stably by factoring out the larger exponent.
        M = np.maximum(A, B)
        # print(f'A: {A}, B: {B}')
        output=np.exp(M) * (np.exp(A - M) - np.exp(B - M))
        # print(f'output: {output}')
        return output

    def kappa(s, eps):
        # kappa(s, eps) = lambda*(exp(g*s)-1) + lambda*(exp(-exp(eps)*g*s)-1) + b*(1-exp(eps))*s
        term1 = lambda_val * np.expm1(g * s)  # exp(g*s)-1
        term2 = lambda_val * np.expm1(-np.exp(eps) * g * s)  # exp(-exp(eps)*g*s)-1
        term3 = b * (-np.expm1(eps)) * s  # b*(1-exp(eps))*s
        return term1 + term2 + term3

    def kappa_prime(s, eps):
        # kappa'(s, eps) = lambda*g*exp(g*s) - lambda*g*exp(eps)*exp(-exp(eps)*g*s) + b*(1-exp(eps))
        A = g * s
        B = eps - np.exp(eps) * g * s
        # Compute exp(g*s) - exp(eps - exp(eps)*g*s) stably:
        diff = stable_diff_exp(A, B)
        term12 = lambda_val * g * diff
        term3 = b * (1 - np.exp(eps))
        return term12 + term3

    def kappa_prime2(s, eps):
        # kappa''(s, eps) = lambda*g^2*exp(g*s) + lambda*g^2*exp(2*eps)*exp(-exp(eps)*g*s)
        A = g * s
        # Combine the exponentials in the second term:
        term1 = lambda_val * g ** 2 * np.exp(A)
        term2 = lambda_val * g ** 2 * np.exp(2 * eps - np.exp(eps) * g * s)
        return term1 + term2

    # Solve for the saddlepoint s* for positive events (kappa'(s*, epsilon) = 0).
    try:
        s_star_pos = newton(lambda s: kappa_prime(s, epsilon),
                            x0=0.0,
                            fprime=lambda s: kappa_prime2(s, epsilon),
                            tol=1e-15,
                            maxiter=10000)
    except RuntimeError:
        s_star_pos = np.nan


    # Solve for the saddlepoint s* for negative events,
    # by modifying epsilon by a factor f.
    epsilon_neg = epsilon * f
    try:
        s_star_neg = newton(lambda s: kappa_prime(s, -epsilon_neg),
                            x0=0.0,
                            fprime=lambda s: kappa_prime2(s, -epsilon_neg),
                            tol=1e-15,
                            maxiter=10000)
    except RuntimeError:
        s_star_neg = np.nan


    def log_probability(s_star, eps):
        kp2 = kappa_prime2(s_star, eps)
        # Build the log-probability in parts:
        log_prefactor = -0.5 * (np.log(2 * np.pi) + np.log(kp2))
        log_exp_term = kappa(s_star, eps)
        log_s_star = -np.log(np.abs(s_star))
        log_correction = np.log(1 + 1 / (kp2 * s_star ** 2))
        return log_prefactor + log_exp_term + log_s_star + log_correction

    # Compute probabilities in the log domain to avoid overflow
    probability_plus = np.exp(log_probability(s_star_pos, epsilon))
    probability_neg = np.exp(log_probability(s_star_neg, -epsilon_neg))

    return probability_plus, probability_neg

def compute_probabilities_saddle_3(lambda_val, epsilon, b, g=1, f=1):
    print(f'lambda_val: {lambda_val}, epsilon: {epsilon}, b: {b}')

    # if lambda_val <= 0.0179 and b>40:
    #     return 0.0, 0.0

    def stable_diff_exp(A, B):
        # Compute exp(A) - exp(B) stably by factoring out the larger exponent.
        # M = np.maximum(A, B)
        # print(f'A: {A}, B: {B}')
        # output = np.exp(M) * (np.exp(A - M) - np.exp(B - M))
        # print(f'output: {output}')
        # A=mpm.mpf(A)
        # B=mpm.mpf(B)
        # print(f'A: {A}, B: {B}')
        output = mpm.exp(A) - mpm.exp(B)
        # print(f'stable_diff_exp output: {output}')
        return output

    def kappa(s, eps, lambda_val):
        # kappa(s, eps) = lambda*(exp(g*s)-1) + lambda*(exp(-exp(eps)*g*s)-1) + b*(1-exp(eps))*s
        term1 = lambda_val * mpm.expm1(g * s)  # exp(g*s)-1
        term2 = lambda_val * mpm.expm1(-mpm.exp(eps) * g * s)  # exp(-exp(eps)*g*s)-1
        term3 = b * (-mpm.expm1(eps)) * s  # b*(1-exp(eps))*s
        return term1 + term2 + term3

    def kappa_prime(s, eps, lambda_val, b):
        # print('call to k prime')
        # kappa'(s, eps) = lambda*g*exp(g*s) - lambda*g*exp(eps)*exp(-exp(eps)*g*s) + b*(1-exp(eps))
        # print(f's: {s}')
        b=mpm.mpf(b)
        g=mpm.mpf('1')
        eps = mpm.mpf(eps) ################working here
        lambda_val = mpm.mpf(lambda_val)
        # s = mpm.mpf(s)
        # print(f'k prime s: {s}, eps: {eps}, lambda_val: {lambda_val}')
        A = g * s
        # print(f'k prime, A: {A}')
        # B = eps - np.exp(eps) * g * s
        B = eps - mpm.exp(eps) * g * s
        # Compute exp(g*s) - exp(eps - exp(eps)*g*s) stably:
        diff = stable_diff_exp(A, B)
        term12 = lambda_val * g * diff
        term3 = b * (1 - mpm.exp(eps))
        output=term12+term3
        # print(f'k prime output: {output}')
        return output

    def kappa_prime2(s, eps, lambda_val):
        # kappa''(s, eps) = lambda*g^2*exp(g*s) + lambda*g^2*exp(2*eps)*exp(-exp(eps)*g*s)
        g=1
        A = g * s
        # print(f'k prime 2 s: {s}, eps: {eps}, A: {A}, lambda_val: {lambda_val}')
        lambda_val=mpm.mpf(lambda_val)
        g=mpm.mpf(g)
        # A=mpm.mpf(A)
        eps=mpm.mpf(eps)
        # s=mpm.mpf(s)
        # Combine the exponentials in the second term:
        # term1 = lambda_val * g ** 2 * np.exp(A)
        term1 = lambda_val * g ** mpm.mpf('2') * mpm.exp(A)
        # print(f'k prime 2 term1: {term1}')
        # term2 = lambda_val * g ** 2 * np.exp(2 * eps - np.exp(eps) * g * s)
        term2 = lambda_val * g ** mpm.mpf('2') * mpm.exp(mpm.mpf('2') * eps - mpm.exp(eps) * g * s)
        # print(f'k prime 2 term2: {term2}')
        output=term1 + term2
        # print(f'k prime 2 output: {output}')
        return output

    # Solve for the saddlepoint s* for positive events (kappa'(s*, epsilon) = 0).
    # try:
    #     s_star_pos = newton(lambda s: kappa_prime(s, epsilon, lambda_val),
    #                         x0=0.0,
    #                         fprime=lambda s: kappa_prime2(s, epsilon, lambda_val),
    #                         tol=1e-15,
    #                         maxiter=3)
    s0 = mpm.mpf(13)
    guesses=[mpm.mpf(14), mpm.mpf(14.25)]
    guesses_m=[mpm.mpf(13), mpm.mpf(16), mpm.mpf(20)]
    # try:
    s_star_pos = mpm.findroot(
        lambda s: kappa_prime(s, epsilon, lambda_val, b),
        # lambda s: kappa_prime_2(s, epsilon, lambda_val),
        # s0,
        # method='secant',
        guesses, solver='secant',
        # guesses_m, solver='muller',
        # tol=1e-14,  # tighten until you like
        tol=0.0001,
        maxsteps=1000000
    )
    # except RuntimeError:
        # s_star_pos = np.nan
    # print(f' \n s_star_pos: {s_star_pos} \n')

    # Solve for the saddlepoint s* for negative events,
    # by modifying epsilon by a factor f.
    epsilon_neg = epsilon * f

    try:
        # s_star_neg = newton(lambda s: kappa_prime(s, -epsilon_neg, lambda_val),
        #                     x0=0.0,
        #                     fprime=lambda s: kappa_prime2(s, -epsilon_neg, lambda_val),
        #                     tol=1e-15,
        #                     maxiter=1)
        s_star_neg = mpm.findroot(
            lambda s: kappa_prime(s, -epsilon, lambda_val, b),
            # lambda s: kappa_prime_2(s, epsilon, lambda_val),
            # s0,
            # method='secant',
            guesses, solver='secant',
            # guesses_m, solver='muller',
            # tol=1e-14,  # tighten until you like
            tol=0.001,
            maxsteps=5000
        )
    except RuntimeError:
        s_star_neg = np.nan
    except ValueError:
        s_star_neg = np.nan

    def log_probability(s_star, eps, lambda_val):
        kp2 = kappa_prime2(s_star, eps, lambda_val)
        # Build the log-probability in parts:
        log_prefactor = -0.5 * (mpm.log(2 * np.pi) + mpm.log(kp2))
        log_exp_term = kappa(s_star, eps, lambda_val)
        log_s_star = -mpm.log(np.abs(s_star))
        log_correction = mpm.log(1 + 1 / (kp2 * s_star ** 2))
        return log_prefactor + log_exp_term + log_s_star + log_correction

    # Compute probabilities in the log domain to avoid overflow
    probability_plus = mpm.exp(log_probability(s_star_pos, epsilon, lambda_val))
    probability_neg = mpm.exp(log_probability(s_star_neg, -epsilon, lambda_val))

    return probability_plus

def compute_probabilities_gaussian(lambda_val, epsilon, b, g=1, f=1):
    """
    Compute probabilities using a Gaussian approximation.
    For positive events epsilon is used as given; for negative events,
    epsilon is scaled by f.
    """
    p_plus = 0.5 - 0.5 * erf(((lambda_val + b) * (np.exp(epsilon) - 1)) /
                             np.sqrt(2 * lambda_val * (1 + np.exp(2 * epsilon))))
    epsilon_neg = epsilon * f
    p_neg = 0.5 + 0.5 * erf(((lambda_val + b) * (np.exp(-epsilon_neg) - 1)) /
                            np.sqrt(2 * lambda_val * (1 + np.exp(-2 * epsilon_neg))))
    return p_plus, p_neg

def run_all():
    """
    This function re-implements the provided Matlab code in Python.
    It defines helper functions for computing probabilities via saddlepoint approximations
    and Gaussian approximations, defines error functions, performs a multi-start optimization
    (using SLSQP) to fit the parameters to given measurements, and then plots the results.
    """

    # -------------------------------------------------------------------------
    # Define error functions that use compute_probabilities.
    def error2_func(lambda_val, epsilon, b, g, f):
        """
        error2 = probability_neg / probability_plus using compute_probabilities.
        """
        p_plus, p_neg = compute_probabilities(lambda_val, epsilon, b, g, f)
        return p_neg / p_plus

    def error_func(lambda_val, epsilon, b, g, f):
        """
        error = probability_neg (with epsilon) / probability_neg (with 2*epsilon)
        using compute_probabilities.
        """
        _, p_neg = compute_probabilities(lambda_val, epsilon, b, g, f)
        _, p_neg2 = compute_probabilities(lambda_val, 2 * epsilon, b, g, f)
        return p_neg / p_neg2

    def error_gaussian(lambda_val, epsilon, b, g, f):
        """
        error_gaussian using the Gaussian approximation.
        """
        _, p_neg = compute_probabilities_gaussian(lambda_val, epsilon, b, g, f)
        _, p_neg2 = compute_probabilities_gaussian(lambda_val, 2 * epsilon, b, g, f)
        return p_neg / p_neg2

    # -------------------------------------------------------------------------
    # Define the plotting function.
    def myplot(lambda_vec, epsilon, b, g, f):
        """
        For each value in lambda_vec, compute both the saddlepoint and Gaussian
        approximations of the probabilities, then plot the individual probabilities
        as well as their sums.
        """
        p = []
        n = []
        p2 = []
        n2 = []
        for lam in lambda_vec:
            pp, nn = compute_probabilities(lam, epsilon, b, g, f)
            pp2, nn2 = compute_probabilities_gaussian(lam, epsilon, b, g, f)
            p.append(pp)
            n.append(nn)
            p2.append(pp2)
            n2.append(nn2)
        p = np.array(p)
        n = np.array(n)
        p2 = np.array(p2)
        n2 = np.array(n2)

        # Figure 1: Plot individual probabilities
        plt.figure(1)
        plt.plot(lambda_vec, p, 'r', linewidth=5, label='P(positive) (saddle)')
        plt.plot(lambda_vec, n, 'b', linewidth=5, label='P(negative) (saddle)')
        plt.plot(lambda_vec, p2, 'r--', linewidth=5, label='P(positive) (gaussian)')
        plt.plot(lambda_vec, n2, 'b--', linewidth=5, label='P(negative) (gaussian)')
        plt.xlabel('lambda')
        plt.ylabel('Probability')
        plt.title('Probabilities vs. lambda')
        plt.legend()

        # Figure 2: Plot sum of probabilities
        plt.figure(2)
        plt.plot(lambda_vec, p + n, 'r', linewidth=5, label='(saddle) P(positive)+P(negative)')
        plt.plot(lambda_vec, p2 + n2, 'r--', linewidth=5, label='(gaussian) P(positive)+P(negative)')
        plt.xlabel('lambda')
        plt.ylabel('Probability Sum')
        plt.title('Sum of Probabilities vs. lambda')
        plt.legend()

        plt.show()

    # -------------------------------------------------------------------------
    # Main code
    # Set parameters (using names different from the reserved word "lambda")
    lambda_val = 20
    epsilon = 1
    b = 1
    g = 1
    f = 0.8

    p_plus1, p_neg1 = compute_probabilities(lambda_val, epsilon, b, g, f=1)
    print("Probability of positive events (with epsilon):", p_plus1)
    print("Probability of negative events (with epsilon):", p_neg1)
    # (The Matlab "if (0)" block with example calls is skipped.)

    # --- Optimization Section ---
    #
    # The Matlab code uses a multi-start constrained minimization to match two measurements.
    # In Python we define the objective function based on the error functions and then perform
    # multiple minimizations from different starting points.

    def objective(x):
        # x[0] = lambda, x[1] = epsilon, x[2] = b.
        lam, eps, b_val = x
        # Our objective function is
        #   (error(lam, eps, b_val, g, f) - 2)^2 + (error2(lam, eps, b_val, g, f) - 9)^2
        return (error_func(lam, eps, b_val, g, f) - 2) ** 2 + (error2_func(lam, eps, b_val, g, f) - 9) ** 2

    # Bounds: lambda in [0,100], epsilon in [0,5], b in [0,10]
    bounds = [(0, 100), (0, 5), (0, 10)]
    x0 = np.array([1, 1, 1])

    best_x = None
    best_val = np.inf
    n_starts = 100

    # Generate multiple starting points: include the given x0 and then additional random points.
    initial_points = [x0]
    rng = np.random.default_rng()
    for _ in range(n_starts - 1):
        init = np.array([rng.uniform(low, high) for (low, high) in bounds])
        initial_points.append(init)

    # Run the minimizations and pick the best result.
    for init in initial_points:
        res = minimize(objective, init, method='SLSQP', bounds=bounds,
                       options={'ftol': 1e-12, 'maxiter': 1000})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    if best_x is None:
        print("Optimization failed.")
        return
    else:
        print("Optimization result x (lambda, epsilon, b):", best_x)
        print("Objective function value:", best_val)

    # Update parameters with the optimized values.
    lambda_opt, epsilon_opt, b_opt = best_x

    # Compute and print probabilities for the optimized parameters.
    p_plus, p_neg = compute_probabilities(lambda_opt, epsilon_opt, b_opt, g, f)
    print("\nUsing optimized parameters:")
    print("Probability of positive events (with epsilon):", p_plus)
    print("Probability of negative events (with epsilon):", p_neg)
    print()

    p_plus_d, p_neg_d = compute_probabilities(lambda_opt, 2 * epsilon_opt, b_opt, g, f)
    print("Probability of positive events (with 2*epsilon):", p_plus_d)
    print("Probability of negative events (with 2*epsilon):", p_neg_d)

    # --- Plotting Section ---
    #
    # Create a lambda vector from 0 to 10 with step 0.01 and call the plotting function.
    lambda_range = np.arange(0, 10.01, 0.01)
    myplot(lambda_range, epsilon_opt, b_opt, g, f)


def compute_probabilities_improved(lambda_val, epsilon, b, g=1, f=1):
    """
    Compute tail probabilities using a refined saddlepoint approximation
    (the Lugannani–Rice formula) with tighter tolerances for improved accuracy.

    The function defines a cumulant-like function κ(s, ε) (and its first two
    derivatives) that is used to approximate the probability that the underlying
    random variable is above (or below) zero. For positive events the function
    uses the original epsilon. For negative events (as in the Matlab code), it
    uses a modified epsilon (scaled by f) with a sign flip.

    The Lugannani–Rice formulas used here are:

      For the upper tail (P(Z ≥ 0)):
         P(Z ≥ 0) ≈ 1 - Φ(r) + φ(r) (1/r - 1/w)

      For the lower tail (P(Z ≤ 0)):
         P(Z ≤ 0) ≈ Φ(r) - φ(r) (1/r - 1/w)

    where
         r = sgn(s*) √[-2 κ(s*)]   and   w = s* √[κ''(s*)]
         with s* the solution of κ'(s*, ε) = 0.

    Parameters:
      lambda_val : float
          Parameter λ.
      epsilon : float
          The ε parameter.
      b : float
          Parameter b.
      g : float
          Parameter g.
      f : float, optional
          Scaling factor for the epsilon used in the negative tail probability.
          (Default is 1.)

    Returns:
      probability_plus : float
          The approximated probability for positive events (Z > 0) using epsilon.
      probability_neg : float
          The approximated probability for negative events (Z < 0) using -epsilon*f.
    """

    # Define the cumulant-like function κ(s, eps) and its derivatives.
    def kappa(s, eps):
        return lambda_val * (np.exp(g * s) - 1) + \
            lambda_val * (np.exp(-np.exp(eps) * g * s) - 1) + \
            b * (1 - np.exp(eps)) * s

    def kappa_prime(s, eps):
        return lambda_val * g * np.exp(g * s) - \
            lambda_val * g * np.exp(eps) * np.exp(-np.exp(eps) * g * s) + \
            b * (1 - np.exp(eps))

    def kappa_prime2(s, eps):
        return lambda_val * g ** 2 * np.exp(g * s) + \
            lambda_val * g ** 2 * np.exp(2 * eps) * np.exp(-np.exp(eps) * g * s)

    # -------------------------------------------------------------------------
    # For the upper tail (positive events) we use the Lugannani–Rice formula.
    def lugannani_rice_upper(eps):
        # Find the saddlepoint s* by solving κ'(s, eps)=0 with a very tight tolerance.
        s_star = newton(lambda s: kappa_prime(s, eps),
                        x0=0.0,
                        fprime=lambda s: kappa_prime2(s, eps),
                        tol=1e-14,
                        maxiter=10000)
        K_val = kappa(s_star, eps)
        # For x = 0 the signed root is defined as:
        #   r = sgn(s_star) * sqrt(-2 κ(s_star))
        if -2 * K_val <= 0:
            r = 0.0  # Fall back if the value is not in the expected range.
        else:
            r = np.sign(s_star) * sqrt(-2 * K_val)
        # w is defined as:
        w = s_star * sqrt(kappa_prime2(s_star, eps))
        # Lugannani–Rice formula for the upper tail:
        #   P(Z ≥ 0) ≈ 1 - Φ(r) + φ(r)(1/r - 1/w)
        prob = 1 - norm.cdf(r) + norm.pdf(r) * (1.0 / r - 1.0 / w)
        return prob

    # -------------------------------------------------------------------------
    # For the lower tail (negative events), we adapt the formula.
    def lugannani_rice_lower(eps):
        # Solve for s* using the modified epsilon (which will be negative).
        s_star = newton(lambda s: kappa_prime(s, eps),
                        x0=0.0,
                        fprime=lambda s: kappa_prime2(s, eps),
                        tol=1e-14,
                        maxiter=10000)
        K_val = kappa(s_star, eps)
        if -2 * K_val <= 0:
            r = 0.0
        else:
            r = np.sign(s_star) * sqrt(-2 * K_val)
        w = s_star * sqrt(kappa_prime2(s_star, eps))
        # Lugannani–Rice formula for the lower tail:
        #   P(Z ≤ 0) ≈ Φ(r) - φ(r)(1/r - 1/w)
        prob = norm.cdf(r) - norm.pdf(r) * (1.0 / r - 1.0 / w)
        return prob

    # -------------------------------------------------------------------------
    # Compute probability for positive events (using epsilon as given)
    probability_plus = lugannani_rice_upper(epsilon)

    # Compute probability for negative events.
    # In the Matlab code the negative event probability is computed using:
    #    kappa(s, -epsilon)  with epsilon scaled by f.
    # Here we pass eps = -epsilon * f.
    probability_neg = lugannani_rice_lower(-epsilon * f)

    return probability_plus, probability_neg


def compute_probabilities_hybrid(lambda_val, epsilon, b, g=1, f=1):
    """
    Compute tail probabilities using two saddlepoint approximations:
      1. The basic saddlepoint approximation (as in the original Matlab code)
      2. A refined approximation using the Lugannani–Rice formula
    and then combine them via a simple average (hybrid approach).

    For the positive events the basic approximation is given by:
      P_plus_basic = (1/√(2π K''(s*)))*exp(K(s*)) / s* (1 + 1/(K''(s*) (s*)²))
    where s* is the saddlepoint solving K'(s*, ε)=0,
    with K(s,ε) defined as:
      K(s,ε) = λ (exp(g s)-1) + λ (exp(-exp(ε) g s)-1) + b (1-exp(ε)) s.

    The Lugannani–Rice formula approximates the upper tail by:
      P(Z ≥ 0) ≈ 1 - Φ(r) + φ(r) (1/r - 1/w)
    with
      r = sign(s*) √[-2 K(s*,ε)]   and   w = s* √[K''(s*,ε)].

    For negative events the same formulas are used but with ε replaced by -ε*f.

    The final hybrid probabilities are taken as the simple average of the two estimates.

    Parameters:
      lambda_val : float
          Parameter λ.
      epsilon : float
          Parameter ε.
      b : float
          Parameter b.
      g : float
          Parameter g.
      f : float, optional
          Scaling factor for the epsilon used in the negative tail probability.
          (Default is 1.)

    Returns:
      probability_plus : float
          Hybrid approximated probability for positive events (Z > 0).
      probability_neg : float
          Hybrid approximated probability for negative events (Z < 0).
    """

    # Define the cumulant-like function K and its first two derivatives.
    def K(s, eps):
        return lambda_val * (np.exp(g * s) - 1) + \
            lambda_val * (np.exp(-np.exp(eps) * g * s) - 1) + \
            b * (1 - np.exp(eps)) * s

    def K_prime(s, eps):
        return lambda_val * g * np.exp(g * s) - \
            lambda_val * g * np.exp(eps) * np.exp(-np.exp(eps) * g * s) + \
            b * (1 - np.exp(eps))

    def K_prime2(s, eps):
        return lambda_val * g ** 2 * np.exp(g * s) + \
            lambda_val * g ** 2 * np.exp(2 * eps) * np.exp(-np.exp(eps) * g * s)

    # -------------------------------
    # Basic saddlepoint approximation for positive events.
    tol_basic = 1e-12
    try:
        s_star_basic = newton(lambda s: K_prime(s, epsilon),
                              x0=0.0,
                              fprime=lambda s: K_prime2(s, epsilon),
                              tol=tol_basic,
                              maxiter=1000)
    except RuntimeError:
        s_star_basic = np.nan

    K_val_basic = K(s_star_basic, epsilon)
    K2_basic = K_prime2(s_star_basic, epsilon)
    basic_plus = (1.0 / np.sqrt(2 * pi * K2_basic)) * np.exp(K_val_basic) / s_star_basic * \
                 (1 + 1.0 / (K2_basic * s_star_basic ** 2))

    # -------------------------------
    # Lugannani–Rice approximation for positive events.
    tol_LR = 1e-14
    try:
        s_star_LR = newton(lambda s: K_prime(s, epsilon),
                           x0=0.0,
                           fprime=lambda s: K_prime2(s, epsilon),
                           tol=tol_LR,
                           maxiter=1000)
    except RuntimeError:
        s_star_LR = np.nan

    K_val_LR = K(s_star_LR, epsilon)
    # Ensure the quantity inside sqrt is positive
    if -2 * K_val_LR > 0:
        r = np.sign(s_star_LR) * sqrt(-2 * K_val_LR)
    else:
        r = 0.0
    w = s_star_LR * sqrt(K_prime2(s_star_LR, epsilon))
    LR_plus = 1 - norm.cdf(r) + norm.pdf(r) * (1.0 / r - 1.0 / w)

    # Hybrid positive probability as the simple average of the two estimates.
    probability_plus = 0.5 * (basic_plus + LR_plus)

    # -------------------------------
    # Now for negative events.
    # In the original Matlab code, negative events are computed using epsilon replaced by -epsilon*f.
    eps_neg = -epsilon * f

    # Basic saddlepoint approximation for negative events.
    try:
        s_star_basic_neg = newton(lambda s: K_prime(s, eps_neg),
                                  x0=0.0,
                                  fprime=lambda s: K_prime2(s, eps_neg),
                                  tol=tol_basic,
                                  maxiter=1000)
    except RuntimeError:
        s_star_basic_neg = np.nan

    K_val_basic_neg = K(s_star_basic_neg, eps_neg)
    K2_basic_neg = K_prime2(s_star_basic_neg, eps_neg)
    # Note the extra minus sign in the denominator as in the Matlab code.
    basic_neg = (1.0 / np.sqrt(2 * pi * K2_basic_neg)) * np.exp(K_val_basic_neg) / (-s_star_basic_neg) * \
                (1 + 1.0 / (K2_basic_neg * s_star_basic_neg ** 2))

    # Lugannani–Rice approximation for negative events.
    try:
        s_star_LR_neg = newton(lambda s: K_prime(s, eps_neg),
                               x0=0.0,
                               fprime=lambda s: K_prime2(s, eps_neg),
                               tol=tol_LR,
                               maxiter=1000)
    except RuntimeError:
        s_star_LR_neg = np.nan

    K_val_LR_neg = K(s_star_LR_neg, eps_neg)
    if -2 * K_val_LR_neg > 0:
        r_neg = np.sign(s_star_LR_neg) * sqrt(-2 * K_val_LR_neg)
    else:
        r_neg = 0.0
    w_neg = s_star_LR_neg * sqrt(K_prime2(s_star_LR_neg, eps_neg))
    LR_neg = norm.cdf(r_neg) - norm.pdf(r_neg) * (1.0 / r_neg - 1.0 / w_neg)

    probability_neg = 0.5 * (basic_neg + LR_neg)

    return probability_plus, probability_neg


def compute_probabilities_refined(lambda_val, epsilon, b, g=1, f=1):
    """
    Compute tail probabilities using a refined saddlepoint approximation
    that includes a third-order (skewness) correction to the standard
    Lugannani–Rice formula.

    The cumulant generating function is defined as:
        K(s, ε) = λ (exp(g*s)-1) + λ (exp(-exp(ε)*g*s)-1) + b (1-exp(ε))*s,
    with derivatives
        K'(s, ε)   = λ g exp(g s) - λ g exp(ε) exp(-exp(ε)*g s) + b (1-exp(ε))
        K''(s, ε)  = λ g² exp(g s) + λ g² exp(2ε) exp(-exp(ε)*g s)
        K'''(s, ε) = λ g³ exp(g s) - λ g³ exp(3ε) exp(-exp(ε)*g s).

    For the upper tail (Z > 0), we solve for the saddlepoint s* satisfying
        K'(s*, ε) = 0,
    then define:
        r = sign(s*) * sqrt(-2*K(s*, ε))
        w = s* sqrt(K''(s*, ε))
        δ = [K'''(s*, ε) / (6*(K''(s*, ε))^(3/2))]*(1/r - 1/w).

    The refined approximation for the upper tail is then:
        P(Z > 0) ≈ 1 - Φ(r) + φ(r)*(1/r - 1/w)*(1 + δ).

    For negative events (Z < 0), we compute similarly but with
        ε_neg = -ε * f,
    and use the relation:
        P(Z < 0) ≈ Φ(r) - φ(r)*(1/r - 1/w)*(1 + δ).

    Parameters:
      lambda_val : float
          Parameter λ.
      epsilon : float
          Parameter ε.
      b : float
          Parameter b.
      g : float
          Parameter g.
      f : float, optional
          Scaling factor for the epsilon used in the negative tail probability.
          (Default is 1.)

    Returns:
      probability_plus : float
          Approximated probability for positive events (Z > 0).
      probability_neg : float
          Approximated probability for negative events (Z < 0).
    """

    # Define the cumulant generating function and its derivatives.
    def K(s, eps):
        return lambda_val * (np.exp(g * s) - 1) + \
            lambda_val * (np.exp(-np.exp(eps) * g * s) - 1) + \
            b * (1 - np.exp(eps)) * s

    def K_prime(s, eps):
        return lambda_val * g * np.exp(g * s) - \
            lambda_val * g * np.exp(eps) * np.exp(-np.exp(eps) * g * s) + \
            b * (1 - np.exp(eps))

    def K_double_prime(s, eps):
        return lambda_val * g ** 2 * np.exp(g * s) + \
            lambda_val * g ** 2 * np.exp(2 * eps) * np.exp(-np.exp(eps) * g * s)

    def K_triple_prime(s, eps):
        return lambda_val * g ** 3 * np.exp(g * s) - \
            lambda_val * g ** 3 * np.exp(3 * eps) * np.exp(-np.exp(eps) * g * s)

    # Helper function: refined tail approximation.
    # tail = 'upper' yields P(Z > 0); tail = 'lower' yields P(Z < 0).
    def refined_tail(eps, tail='upper'):
        # Solve for saddlepoint s* such that K'(s*, eps) = 0.
        s_star = newton(lambda s: K_prime(s, eps),
                        x0=0.0,
                        fprime=lambda s: K_double_prime(s, eps),
                        tol=1e-14,
                        maxiter=1000)
        K_val = K(s_star, eps)
        K2 = K_double_prime(s_star, eps)
        K3 = K_triple_prime(s_star, eps)

        # Compute r and w.
        # (We expect K(s_star, eps) < 0 for the tail event.)
        r = np.sign(s_star) * sqrt(-2 * K_val)
        w = s_star * sqrt(K2)

        # Compute the skewness correction.
        delta = (K3 / (6 * (K2 ** 1.5))) * (1.0 / r - 1.0 / w)

        # Apply the appropriate formula.
        if tail == 'upper':
            # Upper tail (Z > 0):
            return 1 - norm.cdf(r) + norm.pdf(r) * (1.0 / r - 1.0 / w) * (1 + delta)
        elif tail == 'lower':
            # Lower tail (Z < 0):
            return norm.cdf(r) - norm.pdf(r) * (1.0 / r - 1.0 / w) * (1 + delta)
        else:
            raise ValueError("tail must be 'upper' or 'lower'")

    # For positive events, use epsilon as given.
    probability_plus = refined_tail(epsilon, tail='upper')

    # For negative events, use the modified epsilon: ε_neg = -ε * f.
    probability_neg = refined_tail(-epsilon * f, tail='lower')

    return probability_plus, probability_neg

#typicl way to read the outputs:

# p_plus1, p_neg1 = compute_probabilities(lambda_val, epsilon, b, g, f)
# print("Probability of positive events (with epsilon):", p_plus1)
# print("Probability of negative events (with epsilon):", p_neg1)
#
# p_plus, p_neg = compute_probabilities_improved(lambda_val, epsilon, b, g, f)
# print("Improved probability for positive events: ", p_plus)
# print("Improved probability for negative events: ", p_neg)
#
# p_plus, p_neg = compute_probabilities_hybrid(lambda_val, epsilon, b, g, f)
# print("Hybrid probability for positive events:", p_plus)
# print("Hybrid probability for negative events:", p_neg)
#
# p_plus, p_neg = compute_probabilities_refined(lambda_val, epsilon, b, g, f)
# print("Refined probability for positive events (Z > 0):", p_plus)
# print("Refined probability for negative events (Z < 0):", p_neg)

