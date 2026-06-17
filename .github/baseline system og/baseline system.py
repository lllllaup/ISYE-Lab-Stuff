import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_zipf(n, zipf_s=1.0):
    ranks = np.arange(1, n + 1)
    popularity = 1 / np.power(ranks, zipf_s)
    popularity = popularity / popularity.max()  # Normalize to sum to 1
    return popularity

def make_preference(
        #each groups prefercen for item j is:
            # gamma *shared_popularity[j] + (group_specific_preference[g][j])
        #gamma near 1 means groups mostly agree on mainstram popular items, and gramma 0 means polarized
        n=100,
        gamma=0.7,
        zipf_s=1.0,
        niche_strength=1.5,
        noise_sd=0.15,
        seed=0
):
    
    rng = np.random.default_rng(seed)

    mu_shared = make_zipf(n, zipf_s) #mainstream popularity: h it's eavy-tailed

    #MAJORITY GROUP
    specific_M = rng.normal(loc=0, scale=1.0, size=n) #group-specific preference: normal distribution
    specific_M = specific_M - specific_M.min() #shift to make all values non-negative
    specific_M = specific_M / specific_M.max() #normalize 
    specific_M = specific_M * niche_strength #amplify niche preferences
    mu_M = gamma * mu_shared + (1 - gamma) * specific_M #combine mainstream and niche preferences

    #MINORITY GROUP
    specific_m = rng.normal(loc=0, scale=1.0, size=n) #group-specific preference: normal distribution
    specific_m = specific_m - specific_m.min() #shift to make all values non-negative   
    specific_m = specific_m / specific_m.max() #normalize
    specific_m = specific_m * niche_strength #amplify niche preferences
    mu_m = gamma * mu_shared + (1 - gamma) * specific_m #combine mainstream and niche preferences

    return {
        'mu_shared': mu_shared,
        'mu_M': mu_M,
        'mu_m': mu_m,
        'specific_M': specific_M,
        'specific_m': specific_m,
        'noise_sd': noise_sd
    }

def softmax_probabilities(scores, temperature):
    #this converts item scores into reommendation probabiltiies 
        #smaller temperature makes distribution more agressive explotiation
        #larger temper means more exploration/spread

    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
        
    scaled = scores / temperature

    #numerical stabiltiy
    scaled = scaled - np.max(scaled)

    exp_scores = np.exp(scaled)
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities

def simulate_recommendation(R, K, lambda_, rng):
     #Recommend K items using softmax over ihistorical revenue scores R

     probs = softmax_probabilities(R, temperature=lambda_)
     n_items = len(R)

     return rng.choice(
        np.arange(n_items),
        size=K,
        replace=False,
        p=probs
     )

def random_recommend_items(n_items, K, rng):
    #reccomend k random items
    return rng.choice(
        np.arange(n_items),
        size=K,
        replace=False
    )   

def simulate_baseline(
    T=10_000, #number of time periods/users
    n_items=100,   #number of items
    K=5,    #number of items recommended each time #lower (try diff values)
    alpha=0.8,  #probality user is from majority group
    gamma=0.7,  #weight on shared mainstream popularity
    zipf_s=1.0, #zipf concentration of mainstream popularity
    niche_strength=1.5, #strength of group-specific preferences
    noise_sd=0.15,  #user level randomness in valuations
    price=0.75, #fixed price for all items
    delta=0.99, #discount factor for historial revenue in recommendation algorithm
    lambda_=0.25,   #softmax temperature for recommendation algorithm
    T_explore=500,   #number of intial random exploration rounds
    minority_top_quantile=0.8,    #defines minority preferred items as those in the top quantile of minority group specific preferences
    seed=0  #random seed for reproducibility
):
     
    rng = np.random.default_rng(seed)
    prefs = make_preference(
        n=n_items,
        gamma=gamma,
        zipf_s=zipf_s,
        niche_strength=niche_strength,
        noise_sd=noise_sd,
        seed=seed
    )

    mu_M = prefs['mu_M']
    mu_m = prefs['mu_m']
    specific_m = prefs['specific_m']

    #define minority ptrferred item
    threshold = np.quantile(specific_m, minority_top_quantile)
    minority_preferred_items = specific_m >= threshold
    minority_preferred_items = np.asarray(minority_preferred_items, dtype=bool)

    #historical revenue score for each item, initialized to 0
    R = np.zeros(n_items)

    rows = []

    for t in range(T):
        #sample user group 

        group = 'M' if rng.random() < alpha else 'm'

        if group == 'M':
            mean_values = mu_M
        else:
            mean_values = mu_m

        #each arriving user has noisy valuations around group means
        values = rng.normal(loc=mean_values, scale=noise_sd)
        values = np.maximum(values, 0.0) #ensure non-negative valuations

        #2) recommend items
        if t < T_explore:
            recommended_items = random_recommend_items(n_items, K, rng)
            phase = 'explore'
        else:
            recommended_items = simulate_recommendation(R, K, lambda_, rng)
            phase = 'softmax'

        recommended_items = np.asarray(recommended_items, dtype=int)

        #3) user chooses whether to buy each recommended item
        utilities = values[recommended_items] - price
        best_idex = np.argmax(utilities)
        best_utility = utilities[best_idex]
        best_item = recommended_items[best_idex]

        if best_utility > 0:
            purchased = True
            purchase_item = best_item
            revenue = price
            user_utility = best_utility
        else:
            purchased = False
            purchase_item = -1
            revenue = 0.0
            user_utility = 0.0

        #4) update historical revenue scores with discounting
        R = delta * R
        if purchased:
            R[purchase_item] += revenue     
        
        #5) Compute per round metrics
        minority_exposure_count = minority_preferred_items[recommended_items].sum()
        minority_exposure_share = minority_exposure_count / K

        avg_recommended_value = values[recommended_items].mean()
        max_recommended_value = values[recommended_items].max()

        rows.append({
            't': t,
            'group': group,
            'phase': phase,
            'purchased': int(purchased),
            'purchase_item': purchase_item,
            'revenue': revenue,
            'user_utility': user_utility,
            'minority_exposure_count': minority_exposure_count,
            'minority_exposure_share': minority_exposure_share,
            'avg_recommended_value': avg_recommended_value,
            'max_recommended_value': max_recommended_value,
            'minority_user': int(group == 'm'),
            'majority_user': int(group == 'M')
        })

    results = pd.DataFrame(rows)

    metadata = {
        "preferences": prefs,
        "minority_preferred_items": minority_preferred_items,
        "final_revenue_scores": R,
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
            "seed": seed
        }
    }

    return results, metadata


def add_rolling_metrics(results, window=500):
    """
    Add rolling metrics so we can see how outcomes change over time.
    """
    df = results.copy()

    df["rolling_purchase_rate"] = df["purchased"].rolling(window, min_periods=1).mean()
    df["rolling_avg_utility"] = df["user_utility"].rolling(window, min_periods=1).mean()
    df["rolling_minority_exposure"] = df["minority_exposure_share"].rolling(window, min_periods=1).mean()
    df["rolling_revenue"] = df["revenue"].rolling(window, min_periods=1).mean()

    # Group-specific rolling means.
    # We calculate these as:
    # rolling sum of utility for the group / rolling count of users in the group

    is_M = (df["group"] == "M").astype(int)
    is_m = (df["group"] == "m").astype(int)

    utility_M_sum = (df["user_utility"] * is_M).rolling(window, min_periods=1).sum()
    utility_m_sum = (df["user_utility"] * is_m).rolling(window, min_periods=1).sum()

    count_M = is_M.rolling(window, min_periods=1).sum()
    count_m = is_m.rolling(window, min_periods=1).sum()

    df["rolling_utility_M"] = utility_M_sum / count_M
    df["rolling_utility_m"] = utility_m_sum / count_m

    purchase_M_sum = (df["purchased"] * is_M).rolling(window, min_periods=1).sum()
    purchase_m_sum = (df["purchased"] * is_m).rolling(window, min_periods=1).sum()

    df["rolling_purchase_M"] = purchase_M_sum / count_M
    df["rolling_purchase_m"] = purchase_m_sum / count_m

    return df

def run_parameter_grid(
    alpha_values=(0.5, 0.6, 0.7, 0.8, 0.9),
    lambda_values=(0.1, 0.25, 0.5, 1.0),
    seeds=(0, 1, 2),
    T=10_000
):
    rows = []

    for alpha in alpha_values:
        for lambda_ in lambda_values:
            for seed in seeds:
                results, metadata = simulate_baseline(
                    T=T,
                    alpha=alpha,
                    lambda_=lambda_,
                    seed=seed
                )

                metrics = compute_fairness_metrics(results, alpha=alpha)

                row = {
                    "alpha": alpha,
                    "lambda_": lambda_,
                    "seed": seed
                }

                row.update(metrics.to_dict())
                rows.append(row)

    return pd.DataFrame(rows)

def summarize_by_group(results):
    """
    Summarize the main metrics separately for majority and minority users.
    """
    summary = results.groupby("group").agg(
        n_users=("group", "size"),
        purchase_rate=("purchased", "mean"),
        avg_utility=("user_utility", "mean"),
        avg_revenue=("revenue", "mean"),
        avg_recommended_value=("avg_recommended_value", "mean"),
        avg_best_recommended_value=("max_recommended_value", "mean"),
        avg_minority_exposure_share=("minority_exposure_share", "mean")
    )

    return summary


def compute_fairness_metrics(results, alpha):
    """
    Compute fairness metrics comparing majority and minority users.
    """
    group_summary = summarize_by_group(results)

    M = group_summary.loc["M"]
    m = group_summary.loc["m"]

    utility_gap = M["avg_utility"] - m["avg_utility"]
    purchase_gap = M["purchase_rate"] - m["purchase_rate"]
    recommended_value_gap = M["avg_recommended_value"] - m["avg_recommended_value"]
    best_recommended_value_gap = (
        M["avg_best_recommended_value"] - m["avg_best_recommended_value"]
    )

    overall_minority_exposure = results["minority_exposure_share"].mean()
    minority_population_share = 1 - alpha
    exposure_representation_gap = minority_population_share - overall_minority_exposure

    metrics = {
        "utility_gap_M_minus_m": utility_gap,
        "purchase_gap_M_minus_m": purchase_gap,
        "recommended_value_gap_M_minus_m": recommended_value_gap,
        "best_recommended_value_gap_M_minus_m": best_recommended_value_gap,
        "overall_minority_item_exposure": overall_minority_exposure,
        "minority_population_share": minority_population_share,
        "exposure_representation_gap": exposure_representation_gap,
        "total_revenue": results["revenue"].sum(),
        "avg_revenue_per_user": results["revenue"].mean(),
        "overall_purchase_rate": results["purchased"].mean()
    }

    return pd.Series(metrics)
def main():
    results, metadata = simulate_baseline(
        T=10_000,
        n_items=100,
        K=5,
        alpha=0.8,
        gamma=0.7,
        zipf_s=1.0,
        price=0.75,
        delta=0.99,
        lambda_=0.25,
        T_explore=500,
        seed=42
    )

    print("Results shape:", results.shape)
    print(results.head())
    print(results.tail())

    print("\nSummary by group:")
    print(summarize_by_group(results))

    print("\nFairness metrics:")
    print(compute_fairness_metrics(results, alpha=metadata["params"]["alpha"]))

    rolling = add_rolling_metrics(results, window=500)

    #plot 1 (purchase rate) over time
    plt.figure(figsize=(10, 5))
    plt.plot(rolling["t"], rolling["rolling_purchase_rate"])
    plt.xlabel("Time")
    plt.ylabel("Rolling purchase rate")
    plt.title("Overall purchase rate over time")
    plt.tight_layout()
    plt.show()
    
    #plot 2: minority preferrred item exposure over time
    plt.figure(figsize=(10, 5))
    plt.plot(rolling["t"], rolling["rolling_minority_exposure"])
    plt.axhline(1 - metadata["params"]["alpha"], linestyle="--", label="Minority population share")
    plt.xlabel("Time")
    plt.ylabel("Rolling minority-preferred item exposure")
    plt.title("Exposure of minority-preferred items over time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #plot 3: group utility overtime
    plt.figure(figsize=(10, 5))
    plt.plot(rolling["t"], rolling["rolling_utility_M"], label="Majority utility")
    plt.plot(rolling["t"], rolling["rolling_utility_m"], label="Minority utility")
    plt.xlabel("Time")
    plt.ylabel("Rolling average utility")
    plt.title("Group utility over time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #run experiemtns ober alpha and lambda
    grid_results = run_parameter_grid()

    grid_summary = grid_results.groupby(["alpha", "lambda_"]).agg(
        utility_gap=("utility_gap_M_minus_m", "mean"),
        purchase_gap=("purchase_gap_M_minus_m", "mean"),
        exposure_gap=("exposure_representation_gap", "mean"),
        total_revenue=("total_revenue", "mean")
    ).reset_index()

    print("\nGrid summary:")
    print(grid_summary)

    #plot unfairness as alpha changes
    plt.figure(figsize=(10, 5))

    for lambda_value in sorted(grid_summary["lambda_"].unique()):
        subset = grid_summary[grid_summary["lambda_"] == lambda_value]
        plt.plot(
            subset["alpha"],
            subset["utility_gap"],
            marker="o",
            label=f"lambda={lambda_value}"
        )

    plt.xlabel("Majority population share alpha")
    plt.ylabel("Utility gap: majority - minority")
    plt.title("Utility unfairness as majority share increases")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()