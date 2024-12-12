

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Function to simulate the Bayesian A/B test
def bayesian_ab_test(control_conversions, treatment_conversions, control_visitors, treatment_visitors, prior_alpha=1, prior_beta=1):
    # Set up priors (Beta distribution)
    control_prior = beta(prior_alpha, prior_beta)
    treatment_prior = beta(prior_alpha, prior_beta)

    # Update priors with observed data (conversions and visitors)
    control_posterior = beta(prior_alpha + control_conversions, prior_beta + control_visitors - control_conversions)
    treatment_posterior = beta(prior_alpha + treatment_conversions, prior_beta + treatment_visitors - treatment_conversions)

    # Sampling from posterior distributions
    control_samples = control_posterior.rvs(10000)
    treatment_samples = treatment_posterior.rvs(10000)

    # Probability that treatment is better than control
    probability_treatment_better = np.mean(treatment_samples > control_samples)

    # Calculate the 95% credible interval for both groups
    control_credible_interval = np.percentile(control_samples, [2.5, 97.5])
    treatment_credible_interval = np.percentile(treatment_samples, [2.5, 97.5])

    return probability_treatment_better, control_credible_interval, treatment_credible_interval

# Streamlit UI
st.title('Bayesian A/B Testing Calculator')

# Inputs for the experiment
control_conversions = st.number_input('Number of Conversions in Control', min_value=0, value=100)
control_visitors = st.number_input('Number of Visitors in Control', min_value=1, value=1000)

treatment_conversions = st.number_input('Number of Conversions in Treatment', min_value=0, value=120)
treatment_visitors = st.number_input('Number of Visitors in Treatment', min_value=1, value=1000)

prior_alpha = st.number_input('Prior Alpha (default 1)', min_value=0, value=1)
prior_beta = st.number_input('Prior Beta (default 1)', min_value=0, value=1)

# Run the Bayesian A/B test simulation
if st.button('Run Test'):
    prob_treatment_better, control_interval, treatment_interval = bayesian_ab_test(
        control_conversions, treatment_conversions, control_visitors, treatment_visitors, prior_alpha, prior_beta
    )

    # Display results
    st.write(f'Probability that the Treatment is better than the Control: {prob_treatment_better:.2f}')
    st.write(f'Control Group 95% Credible Interval: {control_interval}')
    st.write(f'Treatment Group 95% Credible Interval: {treatment_interval}')

    # Plot the posterior distributions
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.linspace(0, 1, 1000)
    ax.plot(x, beta(prior_alpha, prior_beta).pdf(x), label='Prior (Beta(1,1))', linestyle='dashed')
    ax.plot(x, beta(prior_alpha + control_conversions, prior_beta + control_visitors - control_conversions).pdf(x), label='Control Posterior')
    ax.plot(x, beta(prior_alpha + treatment_conversions, prior_beta + treatment_visitors - treatment_conversions).pdf(x), label='Treatment Posterior')
    ax.set_title('Posterior Distributions of Conversion Rates')
    ax.legend()
    st.pyplot(fig)









'''

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
            
            
#to run the app in local: cd path/to/folder containing the app

# then : streamlit run app.py


'''