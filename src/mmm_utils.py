"""
MMM Utility Functions

Shared functions for Marketing Mix Modeling pipeline including:
- Adstock transformation (geometric decay)
- Hill saturation curves (diminishing returns)
- Utility helpers for model operations
"""

import numpy as np
import pandas as pd


def geometric_adstock(spend_data, decay_rate):
    """
    Apply geometric adstock transformation to spending data.
    
    Models the 'echo effect' where past advertising spend carries over
    to future periods with exponential decay.
    
    Parameters
    ----------
    spend_data : array-like
        Raw spending data (typically weekly spend amounts)
    decay_rate : float
        Decay rate between 0.0 and 1.0
        - 0.0: No carryover (immediate effect only)
        - 0.8: 80% carries to next week (long memory)
        - 1.0: 100% carries (infinite memory, unrealistic)
    
    Returns
    -------
    np.ndarray
        Adstocked spending values
    
    Example
    -------
    >>> spend = np.array([1000, 2000, 1500])
    >>> adstocked = geometric_adstock(spend, decay_rate=0.8)
    """
    adstocked_spend = np.zeros(len(spend_data))
    carryover = 0
    
    for t in range(len(spend_data)):
        # Today's effect = Today's Spend + (Yesterday's Effect * Decay)
        adstocked_spend[t] = spend_data[t] + (carryover * decay_rate)
        carryover = adstocked_spend[t]
    
    return adstocked_spend


def hill_saturation(spend_data, alpha, beta):
    """
    Apply Hill saturation transformation to spending data.
    
    Models diminishing returns using Hill equation (S-curve response).
    Output is bounded between 0 and 1, representing effectiveness ratio.
    
    Parameters
    ----------
    spend_data : array-like or float
        Spending amount(s)
    alpha : float
        Half-saturation point (spend level where response = 0.5)
        Represents the inflection point of the curve
    beta : float
        Shape parameter controlling curve steepness
        Higher beta = steeper curve (more abrupt saturation)
    
    Returns
    -------
    np.ndarray or float
        Saturation-adjusted effectiveness (0 to 1 scale)
    
    Notes
    -----
    Hill equation: y = 1 / (1 + (x / alpha)^(-beta))
    
    Example
    -------
    >>> spend = np.array([1000, 5000, 10000, 20000])
    >>> saturated = hill_saturation(spend, alpha=5000, beta=1.5)
    # Output shows diminishing returns as spend increases
    """
    # Avoid division by zero for zero spends
    safe_spend = np.maximum(spend_data, 1e-9)
    return 1 / (1 + (safe_spend / alpha) ** (-beta))


def adstock_with_saturation(spend_data, decay_rate, alpha, beta):
    """
    Apply both adstock and saturation transformations sequentially.
    
    This models realistic marketing response: spending carries over time
    (adstock) but also exhibits diminishing returns (saturation).
    
    Parameters
    ----------
    spend_data : array-like
        Raw spending data
    decay_rate : float
        Adstock decay rate (0.0 to 1.0)
    alpha : float
        Hill saturation point
    beta : float
        Hill shape parameter
    
    Returns
    -------
    np.ndarray
        Transformed spending values (0 to 1 scale)
    
    Example
    -------
    >>> tv_spend = np.array([3000, 4000, 2500, 5000])
    >>> tv_effect = adstock_with_saturation(tv_spend, 0.85, 10000, 1.5)
    """
    adstocked = geometric_adstock(spend_data, decay_rate)
    saturated = hill_saturation(adstocked, alpha, beta)
    return saturated


def create_adstock_features(df, spend_columns, decay_params):
    """
    Create adstocked features for a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with spend columns
    spend_columns : dict
        Mapping of column names to decay rates
        e.g., {'TV_Spend': 0.85, 'Social_Spend': 0.3, 'Radio_Spend': 0.5}
    decay_params : dict
        Additional parameters for each channel (future use)
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with new adstocked columns appended
    
    Example
    -------
    >>> decay_rates = {'TV_Spend': 0.85, 'Social_Spend': 0.3}
    >>> df_adstocked = create_adstock_features(df, decay_rates, {})
    """
    df_copy = df.copy()
    
    for col, decay_rate in spend_columns.items():
        new_col = col.replace('Spend', 'Adstock')
        df_copy[new_col] = geometric_adstock(df[col].values, decay_rate)
    
    return df_copy


def calculate_roi(spend, revenue, base_revenue=0):
    """
    Calculate Return on Investment (ROI).
    
    Parameters
    ----------
    spend : float
        Amount spent
    revenue : float
        Revenue generated
    base_revenue : float, optional
        Baseline revenue (default 0)
    
    Returns
    -------
    float
        ROI as a ratio (e.g., 2.5 means 2.5x return)
    
    Example
    -------
    >>> roi = calculate_roi(spend=10000, revenue=25000)
    >>> print(f"ROI: {roi:.2f}x")  # Output: ROI: 2.50x
    """
    incremental_revenue = revenue - base_revenue
    if spend == 0:
        return np.inf if incremental_revenue > 0 else 0
    return incremental_revenue / spend


def validate_budget_constraint(allocation, total_budget, tolerance=1e-6):
    """
    Validate that budget allocation sums to total budget.
    
    Parameters
    ----------
    allocation : array-like
        Budget allocation across channels
    total_budget : float
        Total available budget
    tolerance : float
        Allowed numerical error
    
    Returns
    -------
    bool
        True if allocation is valid, False otherwise
    """
    return np.abs(np.sum(allocation) - total_budget) <= tolerance


def normalize_allocation(allocation, total_budget):
    """
    Normalize an allocation to exactly match total budget.
    
    Useful for correcting numerical precision errors.
    
    Parameters
    ----------
    allocation : array-like
        Budget allocation
    total_budget : float
        Target total budget
    
    Returns
    -------
    np.ndarray
        Normalized allocation
    """
    current_total = np.sum(allocation)
    if current_total == 0:
        return np.array(allocation)
    return np.array(allocation) * (total_budget / current_total)
