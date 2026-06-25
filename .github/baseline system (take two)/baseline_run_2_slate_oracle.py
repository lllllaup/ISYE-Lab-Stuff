"""
*lets all take a moment and pray to any and every god out there that this works*


baseline_run_2.py

baseline model take twooooo

Big idea 
- we have the old feedback loop mode still (yay the other work wasn't wasted): users arrive, K items are shown, the system updates
  cumulative historical revenue R_j,t, then softmax recommends items with higher R more often.
- but now we have added more pricing models (oooh she's versatile):
    1. fixed: old baseline, one fixed price for everythinggggg (obviously not realistic but a useful reference point).
    2. oracle: platform knows the true demand curve q_j(p) and picks best price (the one that you like :) )
    3. mle: platform does NOT know true demand. It learns q_hat_j,t(p) from historical
       purchases and prices using that estimate.
- Add two preference worlds:
    1. gaussian: statistical penalty story. sparse minority item data = noisy MLE prices.
    2. zipfian: structural penalty story. heavy-tailed / niche minority items can get buried
       by majority behavior.

I tried to keep variable names pretty close to the first baseline file:
T, n_items, K, alpha, gamma, zipf_s, niche_strength, noise_sd, price, delta, lambda_,
T_explore, minority_top_quantile, seed, R, results, metadata, etc.

outputs (yayyyy):
    baseline_run_2_outputs/grid_results.csv
    baseline_run_2_outputs/grid_summary.csv
    baseline_run_2_outputs/*.png plots
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# lets get this starteddddd

def add_plot_title(ax, title, subtitle=None):
    ax.set_title(title, fontsize=14, weight="bold", pad=24)

    if subtitle is not None:
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            color="dimgray",
            wrap=True
        )

#HELPING HANDS
def make_zipf(n, zipf_s=1.0):
    """
    Heavy-tailed shared popularity distribution.

    old comment said normalize to sum to 1, but the old code normalized to max = 1.
    Keeping max normalization bc it keeps the old baseline scale pretty similar.
    """
    ranks = np.arange(1, n + 1)
    popularity = 1 / np.power(ranks, zipf_s)
    popularity = popularity / popularity.max()
    return popularity


def normal_survival(x, mu, sigma):
    """
    this is for the oracle pricing calculations, where we need to compute
    Pr(N(mu, sigma^2) > x).
    Using erfc so we don't need scipy.
    """
    if sigma <= 0:
        return float(mu > x)
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * math.erfc(z)


def softmax_probabilities(scores, temperature):
    """
    Convert item scores into recommendation probabilities.

    Smaller lambda = more aggressive exploitation.
    Larger lambda = more spread / exploration-ish.

    we're gonnna vary lambda to see how the strength of the feedback loop affects outcomes.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    scaled = scores / temperature
    scaled = scaled - np.max(scaled)  # numerical stability
    exp_scores = np.exp(scaled)

    # little tiny floor so rng.choice(replace=False) does not crash when softmax gets too sharpppp
    # and almost everything gets rounded to 0. This keeps a tiny bit of exploration alive
    # (very necessary for the MLE pricing model, which needs some exploration to learn)
    exp_scores = exp_scores + 1e-12
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities


def simulate_recommendation(R, K, lambda_, rng):
    #Recommend K items using softmax over cumulative historical revenue scores R
    probs = softmax_probabilities(R, temperature=lambda_)
    n_items = len(R)

    return rng.choice(
        np.arange(n_items),
        size=K,
        replace=False,
        p=probs
    )


def random_recommend_items(n_items, K, rng):
    #Recommend K random items for thee exploration phase
    return rng.choice(
        np.arange(n_items),
        size=K,
        replace=False
    )


#   WHAT ARE YOUR PREFEREANCESSSS? 

def make_preference(
    n=100,
    gamma=0.7,
    zipf_s=1.0,
    niche_strength=1.5,
    noise_sd=0.15,
    seed=0,
    preference_model="gaussian",
    minority_niche_share=0.20,
    structural_boost=2.0,
):
    """
    This is how the additive latent model works (work it)
        v_ij | g ~ Normal(gamma * mu_shared[j]
                          + (1 - gamma) * mu_specific[j,g],
                          noise_sd^2)

    preference_model options:
        gaussian = mostly the old baseline. group-specific preferences are random normal-ish.
        zipfian = intentionally creates a stronger heavy-tailed / structural minority niche.
                    This is for the postdoc's structural penalty story.
    """

    rng = np.random.default_rng(seed)
    mu_shared = make_zipf(n, zipf_s=zipf_s)

    if preference_model == "gaussian":
        # This is very close to the first baseline code but with a tad bit more normalization and control over niche strength.
        specific_M = rng.normal(loc=0, scale=1.0, size=n)
        specific_M = specific_M - specific_M.min()
        specific_M = specific_M / max(specific_M.max(), 1e-12)
        specific_M = specific_M * niche_strength

        specific_m = rng.normal(loc=0, scale=1.0, size=n)
        specific_m = specific_m - specific_m.min()
        specific_m = specific_m / max(specific_m.max(), 1e-12)
        specific_m = specific_m * niche_strength

    elif preference_model == "zipfian":
        # Structural / heavy-tailed version:
        # - majority likes globally popular items a lot
        # - minority has some strong niche items, often NOT the top global items
        # - this creates the "minority item has real demand but majority sample buries it" story
        specific_M = mu_shared.copy()
        specific_M = specific_M / max(specific_M.max(), 1e-12)
        specific_M = specific_M * niche_strength

        specific_m = rng.normal(loc=0, scale=0.20, size=n)
        specific_m = specific_m - specific_m.min()
        specific_m = specific_m / max(specific_m.max(), 1e-12)
        specific_m = 0.25 * niche_strength * specific_m

        n_niche = max(1, int(minority_niche_share * n))

        # pick minority niche items from the long tail, not from the obvious popular top 
        # (she's unique and doesn't like the mainstream stuff).
        tail_start = max(1, int(0.35 * n))
        tail_items = np.arange(tail_start, n)
        if len(tail_items) < n_niche:
            tail_items = np.arange(n)

        minority_niche_items = rng.choice(tail_items, size=n_niche, replace=False)

        # heavy-tailed niche strengths among minority niche items.
        niche_ranks = np.arange(1, n_niche + 1)
        niche_values = 1 / np.power(niche_ranks, zipf_s)
        niche_values = niche_values / max(niche_values.max(), 1e-12)
        niche_values = niche_values * niche_strength * structural_boost
        rng.shuffle(niche_values)

        specific_m[minority_niche_items] += niche_values

        # Make majority specifically weak on these minority niche items.
        # This is what creates the structural penalty.
        specific_M[minority_niche_items] *= 0.10

    else:
        raise ValueError("preference_model must be 'gaussian' or 'zipfian'.")

    # Keep gamma, but make it easy to test gamma = 0 or low gamma.
    # Cuz we lowkey may want to get rid of gamma and set gamma=0 and the shared popularity part disappears.
    mu_M = gamma * mu_shared + (1 - gamma) * specific_M
    mu_m = gamma * mu_shared + (1 - gamma) * specific_m

    return {
        "mu_shared": mu_shared,
        "mu_M": mu_M,
        "mu_m": mu_m,
        "specific_M": specific_M,
        "specific_m": specific_m,
        "noise_sd": noise_sd,
        "preference_model": preference_model,
    }


def define_minority_preferred_items(mu_M, mu_m, specific_m, minority_top_quantile=0.8, definition="specific_and_gap"):
    """
    Mark minority-preferred items.

    Old baseline used: specific_m >= quantile.
    New default is a wee bit stricter: high minority-specific preference AND mu_m > mu_M.
    This helps with the actual minority-item story.
    """
    threshold = np.quantile(specific_m, minority_top_quantile)

    if definition == "specific_only":
        minority_preferred_items = specific_m >= threshold
    elif definition == "gap_only":
        minority_preferred_items = mu_m > mu_M
    elif definition == "specific_and_gap":
        minority_preferred_items = (specific_m >= threshold) & (mu_m > mu_M)
        # fallback bc sometimes gamma makes groups too similar and this can be empty
        if minority_preferred_items.sum() == 0:
            minority_preferred_items = specific_m >= threshold
    else:
        raise ValueError("definition must be specific_only, gap_only, or specific_and_gap")

    return np.asarray(minority_preferred_items, dtype=bool)


# -----------------------------
# Pricing functions
# -----------------------------

def monopoly_oracle_q_j_p(j, p, mu_M, mu_m, alpha, noise_sd):
    """
    Item-alone / monopoly demand benchmark.

    This is the OLD oracle logic: it assumes item j is offered by itself,
    so the user buys whenever v_j > p. This is useful as a reference,
    but it is not slate-aware.
    """
    q_M = normal_survival(p, mu_M[j], noise_sd)
    q_m = normal_survival(p, mu_m[j], noise_sd)
    return alpha * q_M + (1 - alpha) * q_m


def monopoly_oracle_prices(mu_M, mu_m, alpha, noise_sd, price_grid):
    """
    OLD item-level oracle: for every item j, choose p_j^* = argmax_p p * Pr(v_j > p).
    Kept only as a diagnostic / benchmark.
    """
    n_items = len(mu_M)
    best_prices = np.zeros(n_items)
    best_revenues = np.zeros(n_items)
    q_at_best = np.zeros(n_items)

    for j in range(n_items):
        q_vals = np.array([monopoly_oracle_q_j_p(j, p, mu_M, mu_m, alpha, noise_sd) for p in price_grid])
        expected_revenue = price_grid * q_vals
        best_k = int(np.argmax(expected_revenue))
        best_prices[j] = price_grid[best_k]
        best_revenues[j] = expected_revenue[best_k]
        q_at_best[j] = q_vals[best_k]

    return best_prices, best_revenues, q_at_best


# Backward-compatible aliases so old plotting code still runs.
oracle_q_j_p = monopoly_oracle_q_j_p
oracle_prices = monopoly_oracle_prices


def _draw_population_values_for_slate(recommended_items, mu_M, mu_m, alpha, noise_sd, n_draws, rng):
    """
    Draw values for exactly the items in one slate from the true population.
    This is used by the slate-aware oracle.
    """
    recommended_items = np.asarray(recommended_items, dtype=int)
    is_majority = rng.random(n_draws) < alpha

    means_M = mu_M[recommended_items]
    means_m = mu_m[recommended_items]
    means = np.where(is_majority[:, None], means_M[None, :], means_m[None, :])

    values = rng.normal(loc=means, scale=noise_sd)
    values = np.maximum(values, 0.0)
    return values


def _expected_slate_revenue_from_prices(values, prices):
    """
    Expected revenue under the same competitive slate choice rule used in the simulation:
    user buys the item with the highest positive v_j - p_j, otherwise buys nothing.
    """
    prices = np.asarray(prices, dtype=float)
    utilities = values - prices[None, :]
    best_pos = np.argmax(utilities, axis=1)
    best_utility = utilities[np.arange(values.shape[0]), best_pos]
    bought = best_utility > 0
    revenue = np.where(bought, prices[best_pos], 0.0)
    return float(np.mean(revenue))


def slate_oracle_prices_for_slate(
    recommended_items,
    mu_M,
    mu_m,
    alpha,
    noise_sd,
    price_grid,
    rng,
    initial_prices=None,
    n_draws=500,
    coordinate_passes=2,
):
    """
    Slate-aware oracle pricing for the exact K recommended items.

    Unlike the monopoly oracle, this benchmark uses the same competitive choice rule
    as the simulation: the user compares all K shown items and buys the best positive
    utility item.

    We approximate expected revenue with Monte Carlo draws and use coordinate search
    over the price grid. This is not a full brute-force search over 40^K combinations,
    but it is much closer to the simulation than Pr(v_j > p_j).
    """
    recommended_items = np.asarray(recommended_items, dtype=int)
    price_grid = np.asarray(price_grid, dtype=float)
    K = len(recommended_items)

    values = _draw_population_values_for_slate(
        recommended_items=recommended_items,
        mu_M=mu_M,
        mu_m=mu_m,
        alpha=alpha,
        noise_sd=noise_sd,
        n_draws=n_draws,
        rng=rng,
    )

    if initial_prices is None:
        # Use the middle of the grid as a neutral default. The caller can also pass
        # monopoly oracle prices as a warmer start.
        current_indices = np.full(K, len(price_grid) // 2, dtype=int)
    else:
        current_indices = np.array([int(np.argmin(np.abs(price_grid - p))) for p in initial_prices], dtype=int)

    current_prices = price_grid[current_indices].copy()

    for _ in range(coordinate_passes):
        for pos in range(K):
            best_idx = int(current_indices[pos])
            best_revenue = -np.inf

            for candidate_idx, candidate_price in enumerate(price_grid):
                candidate_prices = current_prices.copy()
                candidate_prices[pos] = candidate_price
                candidate_revenue = _expected_slate_revenue_from_prices(values, candidate_prices)

                if candidate_revenue > best_revenue:
                    best_revenue = candidate_revenue
                    best_idx = int(candidate_idx)

            current_indices[pos] = best_idx
            current_prices[pos] = price_grid[best_idx]

    final_revenue = _expected_slate_revenue_from_prices(values, current_prices)
    return current_prices, current_indices, final_revenue


def estimate_competitive_oracle_item_prices(
    mu_M,
    mu_m,
    alpha,
    noise_sd,
    price_grid,
    K,
    rng,
    monopoly_price_by_item=None,
    n_slate_samples=250,
    n_draws=300,
    coordinate_passes=2,
):
    """
    Approximate an item-level competitive oracle price for diagnostics.

    A true slate-aware oracle price depends on the other items in the slate, so there is
    no single universal oracle price per item. For plotting MLE-vs-oracle diagnostics,
    we sample random slates, compute the slate-aware oracle price for each slate, and
    average the oracle price assigned to each item when it appears.
    """
    n_items = len(mu_M)
    price_sums = np.zeros(n_items, dtype=float)
    counts = np.zeros(n_items, dtype=float)

    if monopoly_price_by_item is None:
        monopoly_price_by_item, _, _ = monopoly_oracle_prices(mu_M, mu_m, alpha, noise_sd, price_grid)

    for _ in range(n_slate_samples):
        slate = rng.choice(np.arange(n_items), size=K, replace=False)
        initial_prices = monopoly_price_by_item[slate]
        prices, _, _ = slate_oracle_prices_for_slate(
            recommended_items=slate,
            mu_M=mu_M,
            mu_m=mu_m,
            alpha=alpha,
            noise_sd=noise_sd,
            price_grid=price_grid,
            rng=rng,
            initial_prices=initial_prices,
            n_draws=n_draws,
            coordinate_passes=coordinate_passes,
        )
        price_sums[slate] += prices
        counts[slate] += 1

    competitive_prices = monopoly_price_by_item.copy()
    observed = counts > 0
    competitive_prices[observed] = price_sums[observed] / counts[observed]
    return competitive_prices, counts


def mle_price_indices(
    exposures,
    purchases,
    price_grid,
    prior_successes=0.5,
    prior_trials=2.0,
    min_exposures_per_price=1,
    default_price=0.75,
    unseen_price_policy="ignore",
):
    """
    Empirical / MLE-ish pricing.

    ran into some weird stuff (with unseen prices like being rigged), so this hopefully fixes it and is more robust

    The important fix: do not let totally unseen high prices automatiavlly win just because the smoothing prior gives every price the same purchase probability.

    Why this matters:
        If q_hat for unseen prices starts at 1/4, then price=2.50 looks better
        than price=0.75 before we have seen anything. That's fake learning.

    unseen_price_policy:
        "ignore"       = unseen prices cannot win after exploration. This is the default.
        "conservative" = unseen prices get q_hat=0.0.
        "optimistic"   = old behavior-ish; unseen prices get the smoothed prior.

    If an item has no valid observed prices yet, we fall back to default_price.

    """
    n_items, n_prices = exposures.shape
    price_grid = np.asarray(price_grid, dtype=float)
    default_idx = int(np.argmin(np.abs(price_grid - default_price)))

    q_hat = np.zeros_like(exposures, dtype=float)
    estimated_revenue = np.full_like(exposures, fill_value=-np.inf, dtype=float)

    observed_enough = exposures >= min_exposures_per_price

    # Estimate demand only where we actually have enough data (lowkey i orignally did not have this)
    if prior_trials == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            q_hat_observed = purchases / exposures
            q_hat_observed = np.nan_to_num(q_hat_observed, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        q_hat_observed = (purchases + prior_successes) / (exposures + prior_trials)

    q_hat[observed_enough] = q_hat_observed[observed_enough]

    if unseen_price_policy == "ignore":
        estimated_revenue[observed_enough] = q_hat[observed_enough] * np.broadcast_to(price_grid, exposures.shape)[observed_enough]
    elif unseen_price_policy == "conservative":
        estimated_revenue = q_hat * price_grid.reshape(1, -1)
    elif unseen_price_policy == "optimistic":
        # Mostly here as a diagnostic to reproduce the old susss behavior
        if prior_trials == 0:
            q_hat = np.nan_to_num(purchases / np.maximum(exposures, 1), nan=0.0)
        else:
            q_hat = (purchases + prior_successes) / (exposures + prior_trials)
        estimated_revenue = q_hat * price_grid.reshape(1, -1)
    else:
        raise ValueError("unseen_price_policy must be ignore, conservative, or optimistic")

    best_indices = np.argmax(estimated_revenue, axis=1)

    # If an item has no valid price data, np.argmax would pick 0 bc all values are -inf.
    # That's also fake learning, so use the boring price until we learn something.
    no_valid_prices = ~np.isfinite(estimated_revenue).any(axis=1)
    best_indices[no_valid_prices] = default_idx

    return best_indices, q_hat, estimated_revenue


# MAIN SIMULATION
#lets simmulate a runnnnnn


def simulate_baseline_run_2(
    T=1000,
    n_items=100,
    K=5,
    alpha=0.8,
    gamma=0.7,
    zipf_s=1.0,
    niche_strength=1.5,
    noise_sd=0.15,
    price=0.75,
    delta=0.99,
    lambda_=0.25,
    T_explore=300,
    minority_top_quantile=0.8,
    seed=0,
    pricing_model="fixed",
    preference_model="gaussian",
    price_grid=None,
    prior_successes=0.5,
    prior_trials=2.0,
    min_exposures_per_price=1,
    unseen_price_policy="ignore",
    price_explore_random=True,
    epsilon_price=0.05,
    oracle_type="slate",
    slate_oracle_mc=500,
    slate_oracle_passes=2,
):
    """
    baseline simulation (Well the take two(maybe three) of it)

    pricing_model:
        fixed  = old baseline, price is fixed for all items.
        oracle = knows the true value distributions. By default this is a slate-aware
                 competitive oracle, so it prices the exact K-item slate using the
                 same argmax utility purchase rule as the simulation.
        mle    = learns q_hat_j,t(p) from exposures/purchases and prices from that.

    preference_model:
        gaussian = statistical penalty version.
        zipfian  = structural heavy-tailed version.
    """

    rng = np.random.default_rng(seed)

    if price_grid is None:
        price_grid = np.round(np.linspace(0.05, 2.50, 40), 3)
    else:
        price_grid = np.asarray(price_grid, dtype=float)

    prefs = make_preference(
        n=n_items,
        gamma=gamma,
        zipf_s=zipf_s,
        niche_strength=niche_strength,
        noise_sd=noise_sd,
        seed=seed,
        preference_model=preference_model,
    )

    mu_M = prefs["mu_M"]
    mu_m = prefs["mu_m"]
    specific_m = prefs["specific_m"]

    minority_preferred_items = define_minority_preferred_items(
        mu_M=mu_M,
        mu_m=mu_m,
        specific_m=specific_m,
        minority_top_quantile=minority_top_quantile,
        definition="specific_and_gap",
    )

    # R_j,t = cumulative historical revenue state variables used by the recommender (yay feedback loop state!)
    # Keeping this name bc it matches the old code and the theory notes
    R = np.zeros(n_items)

    # Monopoly / item-alone oracle benchmark. Kept for diagnostics and as a warm start.
    # The actual pricing_model="oracle" branch below can use a slate-aware oracle.
    monopoly_oracle_price_by_item, monopoly_oracle_r_star_by_item, monopoly_oracle_q_at_best = monopoly_oracle_prices(
        mu_M=mu_M,
        mu_m=mu_m,
        alpha=alpha,
        noise_sd=noise_sd,
        price_grid=price_grid,
    )

    # Backward-compatible names used elsewhere in the code. If oracle_type="slate",
    # these are NOT the prices shown by the oracle run; they are the old item-alone
    # benchmark used for reference.
    oracle_price_by_item = monopoly_oracle_price_by_item
    oracle_r_star_by_item = monopoly_oracle_r_star_by_item
    oracle_q_at_best = monopoly_oracle_q_at_best

    # MLE historical tables: exposures for each item-price pair.
    exposures = np.zeros((n_items, len(price_grid)))
    purchases = np.zeros((n_items, len(price_grid)))

    rows = []

    for t in range(T):
        # 1) sample user group
        group = "M" if rng.random() < alpha else "m"
        mean_values = mu_M if group == "M" else mu_m

        # individual-level noisy valuations around group means
        values = rng.normal(loc=mean_values, scale=noise_sd)
        values = np.maximum(values, 0.0)

        # 2) recommend items (what do you wnat to seeeee)
        if t < T_explore:
            recommended_items = random_recommend_items(n_items, K, rng)
            phase = "explore"
        else:
            recommended_items = simulate_recommendation(R, K, lambda_, rng)
            phase = "softmax"

        recommended_items = np.asarray(recommended_items, dtype=int)

        # 3) choose prices for the recommended items (dictator...al platform sets prices, user takes it or leaves it)
        if pricing_model == "fixed":
            shown_prices = np.full(K, price, dtype=float)
            shown_price_indices = np.array([
                int(np.argmin(np.abs(price_grid - price))) for _ in range(K)
            ])

        elif pricing_model == "oracle":
            if oracle_type == "slate":
                # Slate-aware oracle: price the exact K-item slate using the same
                # competitive argmax utility choice rule as the simulation.
                shown_prices, shown_price_indices, slate_oracle_expected_revenue = slate_oracle_prices_for_slate(
                    recommended_items=recommended_items,
                    mu_M=mu_M,
                    mu_m=mu_m,
                    alpha=alpha,
                    noise_sd=noise_sd,
                    price_grid=price_grid,
                    rng=rng,
                    initial_prices=monopoly_oracle_price_by_item[recommended_items],
                    n_draws=slate_oracle_mc,
                    coordinate_passes=slate_oracle_passes,
                )
            elif oracle_type == "monopoly":
                # Old oracle: item-alone price based on Pr(v_j > p_j).
                shown_prices = monopoly_oracle_price_by_item[recommended_items]
                shown_price_indices = np.array([
                    int(np.argmin(np.abs(price_grid - p))) for p in shown_prices
                ])
            else:
                raise ValueError("oracle_type must be 'slate' or 'monopoly'.")

        elif pricing_model == "mle":
            if t < T_explore and price_explore_random:
                # important: item exploration alone is not enough; MLE also needs price exploration
                shown_price_indices = rng.choice(len(price_grid), size=K, replace=True)
            else:
                best_price_indices, q_hat, estimated_revenue = mle_price_indices(
                    exposures=exposures,
                    purchases=purchases,
                    price_grid=price_grid,
                    prior_successes=prior_successes,
                    prior_trials=prior_trials,
                    min_exposures_per_price=min_exposures_per_price,
                    default_price=price,
                    unseen_price_policy=unseen_price_policy,
                )
                # Mostly greedy MLE pricing, but keep a small amount of price exploration alive
                # so unseen prices are not permanently banned after T_explore.
                shown_price_indices = []
                for item in recommended_items:
                    if rng.random() < epsilon_price:
                        shown_price_indices.append(rng.choice(len(price_grid)))
                    else:
                        shown_price_indices.append(best_price_indices[item])
                shown_price_indices = np.array(shown_price_indices, dtype=int)

            shown_prices = price_grid[shown_price_indices]

        else:
            raise ValueError("pricing_model must be fixed, oracle, or mle")

        # 4) user chooses whether to buy one of the recommended items (love it or list it)
        utilities = values[recommended_items] - shown_prices
        best_index = int(np.argmax(utilities))
        best_utility = float(utilities[best_index])
        best_item = int(recommended_items[best_index])
        best_price = float(shown_prices[best_index])
        best_price_index = int(shown_price_indices[best_index])

        if best_utility > 0:
            purchased = True
            purchase_item = best_item
            purchase_price = best_price
            purchase_price_index = best_price_index
            revenue = best_price
            user_utility = best_utility
        else:
            purchased = False
            purchase_item = -1
            purchase_price = 0.0
            purchase_price_index = -1
            revenue = 0.0
            user_utility = 0.0

        # 5) update MLE histories
        # exposures = shown item-price pairs
        if pricing_model == "mle":
            for item, p_idx in zip(recommended_items, shown_price_indices):
                exposures[int(item), int(p_idx)] += 1
            if purchased:
                purchases[purchase_item, purchase_price_index] += 1

        # 6) update cumulative historical revenue state R_j,t
        # This is the actual feedback loop stateeee 
        R = delta * R
        if purchased:
            R[purchase_item] += revenue

        # 7) per-round metrics (for diagnostics and analysis)
        minority_exposure_count = int(minority_preferred_items[recommended_items].sum())
        minority_exposure_share = minority_exposure_count / K
        random_minority_item_share = float(minority_preferred_items.mean())

        avg_recommended_value = float(values[recommended_items].mean())
        max_recommended_value = float(values[recommended_items].max())

        avg_shown_price = float(np.mean(shown_prices))
        avg_monopoly_oracle_price_for_shown = float(np.mean(monopoly_oracle_price_by_item[recommended_items]))
        avg_monopoly_oracle_r_star_for_shown = float(np.mean(monopoly_oracle_r_star_by_item[recommended_items]))

        # For oracle runs, this is the average actual oracle price shown. For fixed/MLE,
        # it is left as NaN because a slate-aware oracle price depends on the exact slate.
        avg_slate_oracle_price_for_shown = float(np.mean(shown_prices)) if pricing_model == "oracle" and oracle_type == "slate" else np.nan

        minority_items_shown = minority_preferred_items[recommended_items]
        if minority_items_shown.sum() > 0:
            avg_price_minority_items_shown = float(np.mean(shown_prices[minority_items_shown]))
        else:
            avg_price_minority_items_shown = np.nan

        rows.append({
            "t": t,
            "group": group,
            "phase": phase,
            "pricing_model": pricing_model,
            "preference_model": preference_model,
            "purchased": int(purchased),
            "purchase_item": purchase_item,
            "purchase_price": purchase_price,
            "revenue": revenue,
            "user_utility": user_utility,
            "minority_exposure_count": minority_exposure_count,
            "minority_exposure_share": minority_exposure_share,
            "random_minority_item_share": random_minority_item_share,
            "avg_recommended_value": avg_recommended_value,
            "max_recommended_value": max_recommended_value,
            "avg_shown_price": avg_shown_price,
            "avg_monopoly_oracle_price_for_shown": avg_monopoly_oracle_price_for_shown,
            "avg_monopoly_oracle_r_star_for_shown": avg_monopoly_oracle_r_star_for_shown,
            "avg_slate_oracle_price_for_shown": avg_slate_oracle_price_for_shown,
            "avg_price_minority_items_shown": avg_price_minority_items_shown,
            "minority_user": int(group == "m"),
            "majority_user": int(group == "M"),
            "minority_purchase_item": int(purchased and minority_preferred_items[purchase_item]),
            "majority_purchase_item": int(purchased and not minority_preferred_items[purchase_item]),
        })

    results = pd.DataFrame(rows)

    # Final MLE objects for diagnostics (lets analyzeeee)
    final_mle_price_indices, final_q_hat, final_estimated_revenue = mle_price_indices(
        exposures=exposures,
        purchases=purchases,
        price_grid=price_grid,
        prior_successes=prior_successes,
        prior_trials=prior_trials,
        min_exposures_per_price=min_exposures_per_price,
        default_price=price,
        unseen_price_policy=unseen_price_policy,
    )
    final_mle_prices = price_grid[final_mle_price_indices]

    metadata = {
        "preferences": prefs,
        "minority_preferred_items": minority_preferred_items,
        "final_revenue_scores": R,
        "price_grid": price_grid,
        "oracle_price_by_item": oracle_price_by_item,  # backward-compatible: monopoly/item-alone benchmark
        "oracle_r_star_by_item": oracle_r_star_by_item,
        "oracle_q_at_best": oracle_q_at_best,
        "monopoly_oracle_price_by_item": monopoly_oracle_price_by_item,
        "monopoly_oracle_r_star_by_item": monopoly_oracle_r_star_by_item,
        "monopoly_oracle_q_at_best": monopoly_oracle_q_at_best,
        "mle_exposures": exposures,
        "mle_purchases": purchases,
        "final_mle_prices": final_mle_prices,
        "final_q_hat": final_q_hat,
        "params": {
            "T": T,
            "n_items": n_items,
            "K": K,
            "alpha": alpha,
            "gamma": gamma,
            "zipf_s": zipf_s,
            "niche_strength": niche_strength,
            "noise_sd": noise_sd,
            "price": price,
            "delta": delta,
            "lambda_": lambda_,
            "T_explore": T_explore,
            "minority_top_quantile": minority_top_quantile,
            "seed": seed,
            "pricing_model": pricing_model,
            "preference_model": preference_model,
            "prior_successes": prior_successes,
            "prior_trials": prior_trials,
            "min_exposures_per_price": min_exposures_per_price,
            "unseen_price_policy": unseen_price_policy,
            "epsilon_price": epsilon_price,
            "oracle_type": oracle_type,
            "slate_oracle_mc": slate_oracle_mc,
            "slate_oracle_passes": slate_oracle_passes,
        }
    }

    return results, metadata


# Metrics

def add_rolling_metrics(results, window=250):
    # add rolling metrics so we can see how outcomes change over time (I learned that this is very important)
    df = results.copy()

    df["rolling_purchase_rate"] = df["purchased"].rolling(window, min_periods=1).mean()
    df["rolling_avg_utility"] = df["user_utility"].rolling(window, min_periods=1).mean()
    df["rolling_minority_exposure"] = df["minority_exposure_share"].rolling(window, min_periods=1).mean()
    df["rolling_revenue"] = df["revenue"].rolling(window, min_periods=1).mean()
    df["rolling_avg_price"] = df["avg_shown_price"].rolling(window, min_periods=1).mean()

    is_M = (df["group"] == "M").astype(int)
    is_m = (df["group"] == "m").astype(int)

    utility_M_sum = (df["user_utility"] * is_M).rolling(window, min_periods=1).sum()
    utility_m_sum = (df["user_utility"] * is_m).rolling(window, min_periods=1).sum()

    revenue_M_sum = (df["revenue"] * is_M).rolling(window, min_periods=1).sum()
    revenue_m_sum = (df["revenue"] * is_m).rolling(window, min_periods=1).sum()

    count_M = is_M.rolling(window, min_periods=1).sum()
    count_m = is_m.rolling(window, min_periods=1).sum()

    df["rolling_utility_M"] = utility_M_sum / count_M.replace(0, np.nan)
    df["rolling_utility_m"] = utility_m_sum / count_m.replace(0, np.nan)
    df["rolling_revenue_M"] = revenue_M_sum / count_M.replace(0, np.nan)
    df["rolling_revenue_m"] = revenue_m_sum / count_m.replace(0, np.nan)

    purchase_M_sum = (df["purchased"] * is_M).rolling(window, min_periods=1).sum()
    purchase_m_sum = (df["purchased"] * is_m).rolling(window, min_periods=1).sum()

    df["rolling_purchase_M"] = purchase_M_sum / count_M.replace(0, np.nan)
    df["rolling_purchase_m"] = purchase_m_sum / count_m.replace(0, np.nan)

    return df

def inspect_time_window(results, start=260, end=300):
    """
    Quick little goblin-check for weird jumps in the rolling plots.
    Shows what happened in a small time window.
    """
    window = results[(results["t"] >= start) & (results["t"] <= end)].copy()

    print(f"\nInspecting rows from t={start} to t={end}")
    print("Number of rows:", len(window))
    print("Group counts:")
    print(window["group"].value_counts())

    print("\nMinority user rows:")
    minority_rows = window[window["group"] == "m"]
    if len(minority_rows) == 0:
        print("No minority users in this window.")
    else:
        print(minority_rows[[
            "t",
            "group",
            "phase",
            "pricing_model",
            "purchased",
            "purchase_item",
            "purchase_price",
            "revenue",
            "user_utility",
            "minority_exposure_share",
            "avg_shown_price"
        ]])

    print("\nMean utility by group in this window:")
    print(window.groupby("group")["user_utility"].mean())


def summarize_by_group(results):
    #Summarize main metrics separately for majority and minority users. (segregation??)
    summary = results.groupby("group").agg(
        n_users=("group", "size"),
        purchase_rate=("purchased", "mean"),
        avg_utility=("user_utility", "mean"),
        avg_revenue=("revenue", "mean"),
        avg_recommended_value=("avg_recommended_value", "mean"),
        avg_best_recommended_value=("max_recommended_value", "mean"),
        avg_minority_exposure_share=("minority_exposure_share", "mean"),
        avg_shown_price=("avg_shown_price", "mean"),
    )
    return summary


def compute_fairness_metrics(results, alpha):
    #Compute fairness metrics comparing majority and minority users (is there discrimnation??)
    group_summary = summarize_by_group(results)

    M = group_summary.loc["M"] if "M" in group_summary.index else pd.Series(dtype=float)
    m = group_summary.loc["m"] if "m" in group_summary.index else pd.Series(dtype=float)

    utility_gap = M.get("avg_utility", np.nan) - m.get("avg_utility", np.nan)
    purchase_gap = M.get("purchase_rate", np.nan) - m.get("purchase_rate", np.nan)
    revenue_gap = M.get("avg_revenue", np.nan) - m.get("avg_revenue", np.nan)
    recommended_value_gap = M.get("avg_recommended_value", np.nan) - m.get("avg_recommended_value", np.nan)
    best_recommended_value_gap = M.get("avg_best_recommended_value", np.nan) - m.get("avg_best_recommended_value", np.nan)

    overall_minority_exposure = results["minority_exposure_share"].mean()
    minority_population_share = 1 - alpha
    random_minority_item_share = results["random_minority_item_share"].mean() if "random_minority_item_share" in results.columns else np.nan
    exposure_random_gap = random_minority_item_share - overall_minority_exposure
    # Old benchmark: compare item exposure to minority population share
    # Useful as a rough fairness reference, but it mixes users and items
    exposure_representation_gap = minority_population_share - overall_minority_exposure

    metrics = {
        "utility_gap_M_minus_m": utility_gap,
        "purchase_gap_M_minus_m": purchase_gap,
        "revenue_gap_M_minus_m": revenue_gap,
        "recommended_value_gap_M_minus_m": recommended_value_gap,
        "best_recommended_value_gap_M_minus_m": best_recommended_value_gap,
        "overall_minority_item_exposure": overall_minority_exposure,
        "minority_population_share": minority_population_share,
        "random_minority_item_share": random_minority_item_share,
        "exposure_representation_gap": exposure_representation_gap,
        "exposure_random_gap": exposure_random_gap,
        "total_revenue": results["revenue"].sum(),
        "avg_revenue_per_user": results["revenue"].mean(),
        "overall_purchase_rate": results["purchased"].mean(),
        "minority_item_purchase_share": results["minority_purchase_item"].mean(),
        "avg_shown_price": results["avg_shown_price"].mean(),
    }

    return pd.Series(metrics)


def compute_pricing_diagnostics(metadata):
    """
    Pricing diagnostics for MLE vs Oracle.
    For fixed/oracle runs, final_mle_prices still exists but isn't meaningful unless pricing_model=mle.
    """
    minority_items = metadata["minority_preferred_items"]
    oracle_prices_j = metadata.get("competitive_oracle_price_by_item", metadata["oracle_price_by_item"])
    oracle_r_star_j = metadata["oracle_r_star_by_item"]
    mle_prices_j = metadata["final_mle_prices"]
    exposures = metadata["mle_exposures"]
    pricing_model = metadata.get("params", {}).get("pricing_model", "unknown")
    mle_diagnostics_are_meaningful = pricing_model == "mle"

    if minority_items.sum() == 0:
        minority_price_error = np.nan
        minority_oracle_r_star = np.nan
        minority_exposures = np.nan
    else:
        minority_price_error = np.mean(np.abs(mle_prices_j[minority_items] - oracle_prices_j[minority_items])) if mle_diagnostics_are_meaningful else np.nan
        minority_oracle_r_star = np.mean(oracle_r_star_j[minority_items])
        minority_exposures = np.mean(exposures[minority_items].sum(axis=1))

    majority_like_items = ~minority_items
    if majority_like_items.sum() == 0:
        majority_price_error = np.nan
        majority_oracle_r_star = np.nan
        majority_exposures = np.nan
    else:
        majority_price_error = np.mean(np.abs(mle_prices_j[majority_like_items] - oracle_prices_j[majority_like_items])) if mle_diagnostics_are_meaningful else np.nan
        majority_oracle_r_star = np.mean(oracle_r_star_j[majority_like_items])
        majority_exposures = np.mean(exposures[majority_like_items].sum(axis=1))

    return pd.Series({
        "avg_oracle_price_all_items": np.mean(oracle_prices_j),
        "avg_final_mle_price_all_items": np.mean(mle_prices_j) if mle_diagnostics_are_meaningful else np.nan,
        "avg_abs_mle_oracle_price_error_all_items": np.mean(np.abs(mle_prices_j - oracle_prices_j)) if mle_diagnostics_are_meaningful else np.nan,
        "avg_abs_mle_oracle_price_error_minority_items": minority_price_error,
        "avg_abs_mle_oracle_price_error_majority_items": majority_price_error,
        "avg_oracle_r_star_minority_items": minority_oracle_r_star,
        "avg_oracle_r_star_majority_items": majority_oracle_r_star,
        "avg_mle_exposures_minority_items": minority_exposures,
        "avg_mle_exposures_majority_items": majority_exposures,
        "n_minority_preferred_items": int(minority_items.sum()),
    })



# Grid experiments

def run_parameter_grid_run_2(
    n_values=(50, 100),
    gamma_values=(0.0, 0.3, 0.7),
    delta_values=(0.95, 0.99),
    K_values=(1, 3, 5, 10),
    pricing_models=("fixed", "oracle", "mle"),
    preference_models=("gaussian", "zipfian"),
    alpha_values=(0.8, 0.9),
    lambda_values=(0.1, 0.25, 1.0),
    seeds=(0, 1),
    T=1000,
):
    """
    Pretty comprehensive but still not insanely slow.
    If this takes too long, cut seeds or remove some grid values.
    """
    rows = []

    for preference_model in preference_models:
        for pricing_model in pricing_models:
            for n_items in n_values:
                for gamma in gamma_values:
                    for delta in delta_values:
                        for K in K_values:
                            # make sure K isn't bigger than n_items bc lol
                            if K >= n_items:
                                continue
                            for alpha in alpha_values:
                                for lambda_ in lambda_values:
                                    for seed in seeds:
                                        results, metadata = simulate_baseline_run_2(
                                            T=T,
                                            n_items=n_items,
                                            K=K,
                                            alpha=alpha,
                                            gamma=gamma,
                                            delta=delta,
                                            lambda_=lambda_,
                                            seed=seed,
                                            pricing_model=pricing_model,
                                            preference_model=preference_model,
                                            T_explore=max(100, int(0.10 * T)),
                                        )

                                        metrics = compute_fairness_metrics(results, alpha=alpha)
                                        pricing_metrics = compute_pricing_diagnostics(metadata)

                                        row = {
                                            "preference_model": preference_model,
                                            "pricing_model": pricing_model,
                                            "n_items": n_items,
                                            "gamma": gamma,
                                            "delta": delta,
                                            "K": K,
                                            "alpha": alpha,
                                            "lambda_": lambda_,
                                            "seed": seed,
                                            "T": T,
                                        }
                                        row.update(metrics.to_dict())
                                        row.update(pricing_metrics.to_dict())
                                        rows.append(row)

    return pd.DataFrame(rows)


# Plotting with seaborn bc sm prettieerrrr

def save_single_run_plots(results, metadata, output_dir, label="single_run", window=250):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    rolling = add_rolling_metrics(results, window=window)
    params = metadata["params"]

    pricing_model = params["pricing_model"]
    preference_model = params["preference_model"]

    model_label = f"{pricing_model.upper()} pricing, {preference_model} preferences"

    setting_label = (
        f"T={params['T']}, K={params['K']}, alpha={params['alpha']}, "
        f"gamma={params['gamma']}, delta={params['delta']}, lambda={params['lambda_']}"
    )
    # Plot 1: revenue over time
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=rolling, x="t", y="rolling_revenue")
    plt.title(f"Rolling revenue over time ({model_label})")
    plt.xlabel("Time")
    plt.ylabel("Rolling revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_rolling_revenue.png"), dpi=200)
    plt.close()

    # Plot 2: minority exposure over time
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=rolling, x="t", y="rolling_minority_exposure")
    random_share = float(metadata["minority_preferred_items"].mean())
    plt.axhline(random_share, linestyle="--", label="Random item-exposure baseline")
    plt.axhline(1 - params["alpha"], linestyle=":", label="Minority population share")
    plt.title("Exposure of minority-preferred items over time")
    plt.xlabel("Time")
    plt.ylabel("Rolling minority item exposure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_minority_exposure.png"), dpi=200)
    plt.close()

    # Plot 3: group utility over time
    plot_df = rolling.melt(
        id_vars="t",
        value_vars=["rolling_utility_M", "rolling_utility_m"],
        var_name="group_metric",
        value_name="rolling_utility",
    )
    plot_df["group_metric"] = plot_df["group_metric"].replace({
        "rolling_utility_M": "Majority utility",
        "rolling_utility_m": "Minority utility",
    })
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=plot_df, x="t", y="rolling_utility", hue="group_metric")
    plt.title("Majority vs Minority utility over time")
    plt.suptitle((f"Minority users are only {round((1 - params['alpha']) * 100)}% of arrivals."))
    plt.xlabel("Time")
    plt.ylabel("Rolling average utility")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_group_utility.png"), dpi=200)
    plt.close()

    # Plot 4: price over time
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=rolling, x="t", y="rolling_avg_price")
    plt.title("Average shown price over time")
    plt.xlabel("Time")
    plt.ylabel("Rolling average shown price")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_avg_price.png"), dpi=200)
    plt.close()
def save_oracle_vs_mle_revenue_plot(
    output_dir,
    preference_model="zipfian",
    T=1000,
    n_items=100,
    K=3,
    alpha=0.9,
    gamma=0.3,
    zipf_s=1.0,
    price=0.75,
    delta=0.99,
    lambda_=0.1,
    seed=42,
    window=250,
):
    """
    Runs Oracle and MLE separately, then plots their rolling revenue
    on the same graph.
    """

    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for pricing_model in ["oracle", "mle"]:

        results, metadata = simulate_baseline_run_2(
            T=T,
            n_items=n_items,
            K=K,
            alpha=alpha,
            gamma=gamma,
            zipf_s=zipf_s,
            price=price,
            delta=delta,
            lambda_=lambda_,
            T_explore=max(100, int(0.10 * T)),
            seed=seed,
            pricing_model=pricing_model,
            preference_model=preference_model,
        )

        rolling = add_rolling_metrics(results, window=window)

        # important: force the model label after the rolling metrics are made
        rolling["pricing_model"] = pricing_model

        all_rows.append(rolling)

    plot_df = pd.concat(all_rows, ignore_index=True)

    # little sanity check so we know both models are actually there
    print("\nOracle vs MLE comparison plot check")
    print("Preference model:", preference_model)
    print(plot_df["pricing_model"].value_counts())

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(
        data=plot_df,
        x="t",
        y="rolling_revenue",
        hue="pricing_model",
        estimator=None,
        units="pricing_model",
        ax=ax,
    )

    add_plot_title(
        ax,
        title=f"Oracle vs MLE rolling revenue ({preference_model} preferences)",
        subtitle=(
            f"Same setup: T={T}, K={K}, alpha={alpha}, gamma={gamma}, "
            f"delta={delta}, lambda={lambda_}. "
            f"Oracle is slate-aware and uses the same competitive argmax choice rule as the simulation; "
            f"MLE estimates demand from historical exposures and purchases."
        ),
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling revenue per user")
    ax.legend(title="Pricing model")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"compare_oracle_mle_revenue_{preference_model}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
def save_mle_vs_oracle_price_diagnostic(
    output_dir,
    preference_model="zipfian",
    T=1000,
    n_items=100,
    K=3,
    alpha=0.9,
    gamma=0.3,
    zipf_s=1.0,
    price=0.75,
    delta=0.99,
    lambda_=0.1,
    seed=42,
    competitive_oracle_slate_samples=250,
    competitive_oracle_mc=300,
):
    """
    Compare final MLE prices vs Oracle prices item-by-item.
    This helps us see whether MLE is mispricing minority-preferred items specifically.
    """
    os.makedirs(output_dir, exist_ok=True)

    results, metadata = simulate_baseline_run_2(
        T=T,
        n_items=n_items,
        K=K,
        alpha=alpha,
        gamma=gamma,
        zipf_s=zipf_s,
        price=price,
        delta=delta,
        lambda_=lambda_,
        T_explore=max(100, int(0.10 * T)),
        seed=seed,
        pricing_model="mle",
        preference_model=preference_model,
    )

    minority_items = metadata["minority_preferred_items"]
    mle_prices_j = metadata["final_mle_prices"]

    # A true slate-aware oracle price depends on the other K-1 items shown with it.
    # For item-level diagnostics, approximate each item's competitive-oracle benchmark
    # by averaging over random slates that include that item.
    diagnostic_rng = np.random.default_rng(seed + 999)
    oracle_prices_j, oracle_slate_counts = estimate_competitive_oracle_item_prices(
        mu_M=metadata["preferences"]["mu_M"],
        mu_m=metadata["preferences"]["mu_m"],
        alpha=alpha,
        noise_sd=metadata["preferences"]["noise_sd"],
        price_grid=metadata["price_grid"],
        K=K,
        rng=diagnostic_rng,
        monopoly_price_by_item=metadata["monopoly_oracle_price_by_item"],
        n_slate_samples=competitive_oracle_slate_samples,
        n_draws=competitive_oracle_mc,
        coordinate_passes=2,
    )

    price_df = pd.DataFrame({
        "item": np.arange(n_items),
        "competitive_oracle_price": oracle_prices_j,
        "oracle_slate_count": oracle_slate_counts,
        "mle_price": mle_prices_j,
        "item_type": np.where(minority_items, "Minority-preferred", "Other items"),
    })

    price_df["abs_price_gap"] = np.abs(price_df["mle_price"] - price_df["competitive_oracle_price"])

    # Scatter: item-by-item price comparison
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=price_df,
        x="competitive_oracle_price",
        y="mle_price",
        hue="item_type",
        alpha=0.75,
    )
    max_price = max(price_df["competitive_oracle_price"].max(), price_df["mle_price"].max())    
    plt.plot([0, max_price], [0, max_price], linestyle="--")
    plt.title(f"Final MLE prices vs slate-aware Oracle prices ({preference_model})")
    plt.xlabel("Slate-aware Oracle price")
    plt.ylabel("Final MLE price")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"diagnostic_mle_vs_oracle_prices_{preference_model}.png"),
        dpi=200
    )
    plt.close()

    # Bar/box-ish diagnostic: price error by item type
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=price_df,
        x="item_type",
        y="abs_price_gap",
        errorbar="se",
    )
    plt.title(f"MLE vs slate-aware Oracle price gap by item type ({preference_model})")
    plt.xlabel("Item type")
    plt.ylabel("Absolute price gap")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"diagnostic_mle_price_error_by_item_type_{preference_model}.png"),
        dpi=200
    )
    plt.close()

    price_df.to_csv(
        os.path.join(output_dir, f"diagnostic_prices_{preference_model}.csv"),
        index=False
    )
def save_grid_plots(grid_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Average over seeds.
    grid_summary = grid_results.groupby([
        "preference_model", "pricing_model", "n_items", "gamma", "delta", "K", "alpha", "lambda_"
    ]).agg(
        total_revenue=("total_revenue", "mean"),
        avg_revenue_per_user=("avg_revenue_per_user", "mean"),
        utility_gap=("utility_gap_M_minus_m", "mean"),
        revenue_gap=("revenue_gap_M_minus_m", "mean"),
        exposure_gap=("exposure_representation_gap", "mean"),
        exposure_random_gap=("exposure_random_gap", "mean"),
        minority_exposure=("overall_minority_item_exposure", "mean"),
        price_error_minority=("avg_abs_mle_oracle_price_error_minority_items", "mean"),
        price_error_all=("avg_abs_mle_oracle_price_error_all_items", "mean"),
    ).reset_index()

    grid_summary.to_csv(os.path.join(output_dir, "grid_summary.csv"), index=False)

    # Plot 1: pricing model comparison by preference model
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=grid_results,
        x="pricing_model",
        y="avg_revenue_per_user",
        hue="preference_model",
        errorbar="se",
    )
    plt.title("Average revenue per user: fixed vs oracle vs MLE")
    plt.xlabel("Pricing model")
    plt.ylabel("Avg revenue per user")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_revenue_by_pricing_model.png"), dpi=200)
    plt.close()

    # Plot 2: gamma sensitivity
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=grid_summary,
        x="gamma",
        y="avg_revenue_per_user",
        hue="pricing_model",
        style="preference_model",
        marker="o",
    )
    plt.title("Revenue as gamma changes")
    plt.xlabel("gamma (shared popularity weight)")
    plt.ylabel("Avg revenue per user")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_gamma_revenue.png"), dpi=200)
    plt.close()

    # Plot 3: K sensitivity / how many items shown
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=grid_summary,
        x="K",
        y="minority_exposure",
        hue="pricing_model",
        style="preference_model",
        marker="o",
    )
    plt.title(f"Minority-preferred item exposure over time")
    plt.suptitle((f"Dashed line = random item-exposure baseline; dotted line = minority population share."))
    plt.xlabel("K items shown")
    plt.ylabel("Minority item exposure share")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_K_minority_exposure.png"), dpi=200)
    plt.close()

    # Plot 4: delta sensitivity
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=grid_results,
        x="delta",
        y="exposure_random_gap",
        hue="pricing_model",
        errorbar="se",
    )
    plt.title("Minority-item exposure gap as delta changes")
    plt.xlabel("delta (memory / discount factor)")
    plt.ylabel("random minority-item share - observed exposure")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_delta_exposure_gap.png"), dpi=200)
    plt.close()

    # Plot 5: MLE price error, mostly meaningful for MLE rows
    mle_only = grid_results[grid_results["pricing_model"] == "mle"].copy()
    if len(mle_only) > 0:
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=mle_only,
            x="preference_model",
            y="avg_abs_mle_oracle_price_error_minority_items",
            hue="gamma",
            errorbar="se",
        )
        plt.title("MLE vs Oracle price error for minority-preferred items")
        plt.xlabel("Preference model")
        plt.ylabel("Avg abs price error")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "grid_mle_minority_price_error.png"), dpi=200)
        plt.close()


    # Cleaner faceted gamma plot: same info as above, less soup.
    g = sns.relplot(
        data=grid_summary,
        x="gamma",
        y="avg_revenue_per_user",
        hue="pricing_model",
        col="preference_model",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=4,
        aspect=1.15,
    )
    g.fig.suptitle("Revenue as gamma changes, split by preference model", y=1.05)
    g.set_axis_labels("gamma", "Avg revenue per user")
    g.tight_layout()
    g.savefig(os.path.join(output_dir, "grid_gamma_revenue_faceted.png"), dpi=200)
    plt.close(g.fig)

    # Cleaner faceted K plot.
    g = sns.relplot(
        data=grid_summary,
        x="K",
        y="minority_exposure",
        hue="pricing_model",
        col="preference_model",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=4,
        aspect=1.15,
    )
    g.fig.suptitle("Minority-preferred item exposure as K changes, split by preference model", y=1.05)
    g.set_axis_labels("K items shown", "Minority item exposure share")
    g.tight_layout()
    g.savefig(os.path.join(output_dir, "grid_K_minority_exposure_faceted.png"), dpi=200)
    plt.close(g.fig)

    return grid_summary


# MAIN
# where the magic happens

def main():
    output_dir = "baseline_run_2_revised_outputs_slate_oracle"
    os.makedirs(output_dir, exist_ok=True)

    # Smaller T than the old 10,000 so it runs faster while testing.
    # Bump this up after the code feels stable.
    T = 500

    print("Running a quick single-run demo for each pricing model...")
    demo_rows = []
    for pricing_model in ["fixed", "oracle", "mle"]:
        results, metadata = simulate_baseline_run_2(
            T=T,
            n_items=100,
            K=3,
            alpha=0.9,
            gamma=0.3,
            zipf_s=1.0,
            price=0.75,
            delta=0.99,
            lambda_=0.1,
            T_explore=max(100, int(0.10 * T)),
            seed=42,
            pricing_model=pricing_model,
            preference_model="zipfian",
        )

        print("\n---", pricing_model, "---")
        print("Results shape:", results.shape)
        print("Summary by group:")
        print(summarize_by_group(results))
        print("Fairness metrics:")
        if pricing_model == "fixed":
            inspect_time_window(results, start=260, end=300)
        print(compute_fairness_metrics(results, alpha=metadata["params"]["alpha"]))

        save_single_run_plots(
            results=results,
            metadata=metadata,
            output_dir=output_dir,
            label=f"demo_{pricing_model}_zipfian",
            window=250,
        )

        row = {
            "pricing_model": pricing_model,
            "preference_model": "zipfian",
        }
        row.update(compute_fairness_metrics(results, alpha=metadata["params"]["alpha"]).to_dict())
        row.update(compute_pricing_diagnostics(metadata).to_dict())
        demo_rows.append(row)

    save_oracle_vs_mle_revenue_plot(
        output_dir=output_dir,
        preference_model="zipfian",
        T=T,
        n_items=100,
        K=3,
        alpha=0.9,
        gamma=0.3,
        delta=0.99,
        lambda_=0.1,
        seed=42,
        window=250,
    )

    save_oracle_vs_mle_revenue_plot(
        output_dir=output_dir,
        preference_model="gaussian",
        T=T,
        n_items=100,
        K=3,
        alpha=0.9,
        gamma=0.3,
        delta=0.99,
        lambda_=0.1,
        seed=42,
        window=250,
    )    
    save_mle_vs_oracle_price_diagnostic(
        output_dir=output_dir,
        preference_model="zipfian",
        T=T,
        n_items=100,
        K=3,
        alpha=0.9,
        gamma=0.3,
        delta=0.99,
        lambda_=0.1,
        seed=42,
    )

    save_mle_vs_oracle_price_diagnostic(
        output_dir=output_dir,
        preference_model="gaussian",
        T=T,
        n_items=100,
        K=3,
        alpha=0.9,
        gamma=0.3,
        delta=0.99,
        lambda_=0.1,
        seed=42,
    ) 
    demo_summary = pd.DataFrame(demo_rows)
    demo_summary.to_csv(os.path.join(output_dir, "demo_summary.csv"), index=False)
    print("\nDemo summary:")
    print(demo_summary)

    print("\nRunning parameter grid... this may take a bit.")
    grid_results = run_parameter_grid_run_2(
        n_values=(50,),
        gamma_values=(0.0, 0.3, 0.7),
        delta_values=(0.95, 0.99),
        K_values=(1, 3, 5),
        pricing_models=("fixed", "oracle", "mle"),
        preference_models=("gaussian", "zipfian"),
        alpha_values=(0.9,),
        lambda_values=(0.1, 0.25),
        seeds=(0,),
        T=300,
    )

    grid_results.to_csv(os.path.join(output_dir, "grid_results.csv"), index=False)
    grid_summary = save_grid_plots(grid_results, output_dir=output_dir)

    print("\nGrid results saved to:", os.path.join(output_dir, "grid_results.csv"))
    print("Grid summary saved to:", os.path.join(output_dir, "grid_summary.csv"))
    print("Plots saved in:", output_dir)
    print("\nGrid summary head:")
    print(grid_summary.head())


if __name__ == "__main__":
    main()
