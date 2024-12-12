import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Update Posterior Function
def update_posterior(alpha_prior, beta_prior, successes, failures):
    alpha_posterior = alpha_prior + successes
    beta_posterior = beta_prior + failures
    return stats.beta(alpha_posterior, beta_posterior)

# Function to Simulate Weekly Updates
def simulate_weekly_updates(daily_visitors, baseline_rate, weeks, variations):
    priors = [{"alpha": 1, "beta": 1} for _ in range(variations)]
    posteriors = [stats.beta(1, 1) for _ in range(variations)]
    results = []

    for week in range(1, weeks + 1):
        weekly_results = []

        for i in range(variations):
            visitors_per_variation = daily_visitors * 7
            successes = np.random.binomial(visitors_per_variation, baseline_rate)
            failures = visitors_per_variation - successes

            priors[i]["alpha"] += successes
            priors[i]["beta"] += failures
            posteriors[i] = update_posterior(priors[i]["alpha"], priors[i]["beta"], successes, failures)

            weekly_results.append({
                "expected_conversion": posteriors[i].mean(),
                "credible_interval": posteriors[i].interval(0.95)
            })

        results.append({"week": week, "metrics": weekly_results})
    return results

# Visualization Function
def visualize_results(results, variations):
    for i in range(variations):
        weeks = [result["week"] for result in results]
        means = [result["metrics"][i]["expected_conversion"] for result in results]
        lower_bounds = [result["metrics"][i]["credible_interval"][0] for result in results]
        upper_bounds = [result["metrics"][i]["credible_interval"][1] for result in results]

        plt.figure(figsize=(10, 6))
        plt.plot(weeks, means, label=f"Variation {i + 1}")
        plt.fill_between(weeks, lower_bounds, upper_bounds, alpha=0.2)
        plt.title(f"Expected Conversion and Credible Intervals for Variation {i + 1}")
        plt.xlabel("Week")
        plt.ylabel("Conversion Rate")
        plt.legend()
        st.pyplot(plt)

# Streamlit App Layout
st.title("Bayesian A/B Test Calculator")

# User Inputs
daily_visitors = st.number_input("Daily Visitors (total across variations):", min_value=1, value=200)
baseline_rate = st.number_input("Baseline Conversion Rate (in %):", min_value=0.0, max_value=100.0, value=10.0) / 100
weeks = st.slider("Number of Weeks to Simulate:", min_value=1, max_value=12, value=4)
variations = st.slider("Number of Variations (including control):", min_value=2, max_value=10, value=3)

# Run Simulation
if st.button("Run Simulation"):
    st.write("Simulating A/B test results...")
    results = simulate_weekly_updates(daily_visitors, baseline_rate, weeks, variations)
    visualize_results(results, variations)

    st.write("Summary of Results:")
    for result in results:
        st.write(f"Week {result['week']}:")
        for i, metrics in enumerate(result["metrics"]):
            st.write(
                f"  Variation {i + 1}: Expected Conversion = {metrics['expected_conversion']:.4f}, "
                f"Credible Interval = [{metrics['credible_interval'][0]:.4f}, {metrics['credible_interval'][1]:.4f}]"
            )
            
            
#to run the app: cd path/to/folder containing the app

# then on the terminal : streamlit run app.py