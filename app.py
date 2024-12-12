

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








#to run the app in local: cd path/to/folder containing the app

# then : streamlit run app.py


