"""
Feature importance plotting utilities for NBA betting models.

This module provides functions to generate feature importance visualizations
that are shared between training pipelines and notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Union


def separate_team_features(coeff_series: pd.Series) -> Tuple[pd.Series, list]:
    """
    Separate team abbreviation effects from team statistical features.
    
    Args:
        coeff_series: Series with feature names as index and coefficients as values
        
    Returns:
        Tuple of (team_stats_series, team_abbreviation_coefficients_list)
    """
    team_stats = {}
    team_abbreviations = []
    
    for feature, coef in coeff_series.items():
        if feature.startswith(('home_', 'away_')):
            parts = feature.split('_')
            if len(parts) == 2 and len(parts[1]) == 3 and parts[1].isupper():
                team_abbreviations.append(coef)
            else:
                team_stats[feature] = coef
        else:
            team_stats[feature] = coef
    
    return pd.Series(team_stats), team_abbreviations


def plot_linear_feature_importance(
    linear_coef: pd.Series,
    title: str = "Linear Model Feature Importance",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a horizontal bar plot for linear model feature importance.
    
    Args:
        linear_coef: Series with feature coefficients
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    team_stats_coef, team_abbrev_coefs = separate_team_features(linear_coef)
    
    team_effects_avg = np.mean(np.abs(team_abbrev_coefs)) if team_abbrev_coefs else 0
    team_effects_std = np.std(np.abs(team_abbrev_coefs)) if team_abbrev_coefs else 0
    
    team_stats_coef_with_teams = team_stats_coef.copy()
    if team_abbrev_coefs:
        team_stats_coef_with_teams['team_effects_avg'] = team_effects_avg
    
    fig, ax = plt.subplots(figsize=figsize)
    
    linear_sorted = team_stats_coef_with_teams.abs().sort_values(ascending=True)
    colors_linear = ['red' if team_stats_coef_with_teams[x] < 0 else 'blue' 
                    for x in linear_sorted.index]
    
    x_vals = linear_sorted.values
    y_pos = range(len(linear_sorted))
    ax.barh(y_pos, x_vals, color=colors_linear)
    
    for i, (feature, val) in enumerate(zip(linear_sorted.index, linear_sorted.values)):
        if feature == 'team_effects_avg' and team_effects_std > 0:
            ax.errorbar(val, i, xerr=team_effects_std, fmt='none', 
                       color='black', capsize=5, linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(linear_sorted.index)
    ax.set_xlabel('Absolute Coefficient Value')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    max_val = linear_sorted.max()
    text_space = max_val * 0.15
    ax.set_xlim(0, max_val + text_space + 1)
    
    for i, (feature, val) in enumerate(zip(linear_sorted.index, linear_sorted.values)):
        original_val = team_stats_coef_with_teams[feature]
        color = 'red' if original_val < 0 else 'blue'
        if feature == 'team_effects_avg':
            text_x = val + team_effects_std + 0.2
            ax.text(text_x, i, f'{original_val:.3f}±{team_effects_std:.3f}', 
                   ha='left', va='center', color=color, fontweight='bold')
        else:
            ax.text(val + 0.15, i, f'{original_val:.3f}', 
                   ha='left', va='center', color=color, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_catboost_feature_importance(
    catboost_importance: pd.Series,
    title: str = "CatBoost Feature Importance",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a horizontal bar plot for CatBoost feature importance.
    
    Args:
        catboost_importance: Series with feature importance scores
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    catboost_sorted = catboost_importance.sort_values(ascending=True)
    ax.barh(range(len(catboost_sorted)), catboost_sorted.values, color='green')
    ax.set_yticks(range(len(catboost_sorted)))
    ax.set_yticklabels(catboost_sorted.index)
    ax.set_xlabel('Feature Importance Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    catboost_max_val = catboost_sorted.max()
    catboost_text_space = catboost_max_val * 0.12
    ax.set_xlim(0, catboost_max_val + catboost_text_space)
    
    for i, (idx, val) in enumerate(zip(catboost_sorted.index, catboost_sorted.values)):
        ax.text(val + 0.15, i, f'{val:.4f}', 
               ha='left', va='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_combined_feature_importance(
    linear_coef: pd.Series,
    catboost_importance: pd.Series,
    top_n: int = 15,
    title: str = "Top Features: Linear vs CatBoost",
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Create a combined comparison plot of top features from both models.
    
    Args:
        linear_coef: Linear model coefficients
        catboost_importance: CatBoost feature importance scores
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    team_stats_coef, _ = separate_team_features(linear_coef)
    
    top_linear_stats = team_stats_coef.abs().nlargest(top_n)
    top_catboost = catboost_importance.nlargest(top_n)
    
    combined_top = pd.DataFrame({
        'Linear_Abs_Coeff': top_linear_stats,
        'CatBoost_Importance': top_catboost
    }).fillna(0)
    
    combined_top['max_importance'] = combined_top.max(axis=1)
    combined_top = combined_top.sort_values('max_importance', ascending=True).drop('max_importance', axis=1)
    
    combined_top.plot(kind='barh', ax=ax, width=0.8)
    ax.set_xlabel('Importance Score / Absolute Coefficient')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(['Linear Regression (Abs Coeff)', 'CatBoost (Importance)'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_individual_models_comparison(
    linear_coef: pd.Series,
    catboost_importance: pd.Series,
    title_linear: str = "Linear Regression Feature Importance\n(Team Abbreviations Aggregated)",
    title_catboost: str = "CatBoost Feature Importance\n(Sorted by Importance)",
    figsize: Tuple[int, int] = (20, 10)
) -> plt.Figure:
    """
    Create side-by-side comparison of individual model feature importance.
    
    Args:
        linear_coef: Linear model coefficients
        catboost_importance: CatBoost feature importance scores
        title_linear: Title for linear regression subplot
        title_catboost: Title for CatBoost subplot
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    team_stats_coef, team_abbrev_coefs = separate_team_features(linear_coef)
    
    team_effects_avg = np.mean(np.abs(team_abbrev_coefs)) if team_abbrev_coefs else 0
    team_effects_std = np.std(np.abs(team_abbrev_coefs)) if team_abbrev_coefs else 0
    
    team_stats_coef_with_teams = team_stats_coef.copy()
    if team_abbrev_coefs:
        team_stats_coef_with_teams['team_effects_avg'] = team_effects_avg
    
    linear_sorted = team_stats_coef_with_teams.abs().sort_values(ascending=True)
    colors_linear = ['red' if team_stats_coef_with_teams[x] < 0 else 'blue' 
                    for x in linear_sorted.index]
    
    x_vals = linear_sorted.values
    y_pos = range(len(linear_sorted))
    ax1.barh(y_pos, x_vals, color=colors_linear)
    
    for i, (feature, val) in enumerate(zip(linear_sorted.index, linear_sorted.values)):
        if feature == 'team_effects_avg' and team_effects_std > 0:
            ax1.errorbar(val, i, xerr=team_effects_std, fmt='none', 
                        color='black', capsize=5, linewidth=2)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(linear_sorted.index)
    ax1.set_xlabel('Absolute Coefficient Value')
    ax1.set_title(title_linear, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    max_val = linear_sorted.max()
    text_space = max_val * 0.15
    ax1.set_xlim(0, max_val + text_space + 1)
    
    for i, (feature, val) in enumerate(zip(linear_sorted.index, linear_sorted.values)):
        original_val = team_stats_coef_with_teams[feature]
        color = 'red' if original_val < 0 else 'blue'
        if feature == 'team_effects_avg':
            text_x = val + team_effects_std + 0.2
            ax1.text(text_x, i, f'{original_val:.3f}±{team_effects_std:.3f}', 
                    ha='left', va='center', color=color, fontweight='bold')
        else:
            ax1.text(val + 0.15, i, f'{original_val:.3f}', 
                    ha='left', va='center', color=color, fontweight='bold')
    
    catboost_sorted = catboost_importance.sort_values(ascending=True)
    ax2.barh(range(len(catboost_sorted)), catboost_sorted.values, color='green')
    ax2.set_yticks(range(len(catboost_sorted)))
    ax2.set_yticklabels(catboost_sorted.index)
    ax2.set_xlabel('Feature Importance Score')
    ax2.set_title(title_catboost, fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    catboost_max_val = catboost_sorted.max()
    catboost_text_space = catboost_max_val * 0.12
    ax2.set_xlim(0, catboost_max_val + catboost_text_space)
    
    for i, (idx, val) in enumerate(zip(catboost_sorted.index, catboost_sorted.values)):
        ax2.text(val + 0.15, i, f'{val:.4f}', 
                ha='left', va='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    return fig


def save_feature_importance_plots(
    linear_coef: Optional[pd.Series] = None,
    catboost_importance: Optional[pd.Series] = None,
    output_dir: Union[str, Path] = ".",
    prefix: str = "feature_importance",
    dpi: int = 300
) -> list[Path]:
    """
    Generate and save feature importance plots.
    
    Args:
        linear_coef: Linear model coefficients (optional)
        catboost_importance: CatBoost feature importance (optional)
        output_dir: Directory to save plots
        prefix: Filename prefix
        dpi: Plot resolution
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    saved_files = []
    
    if linear_coef is not None and catboost_importance is not None:
        fig_individual = plot_individual_models_comparison(linear_coef, catboost_importance)
        individual_path = output_dir / f"{prefix}_individual_models.png"
        fig_individual.savefig(individual_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_individual)
        saved_files.append(individual_path)
        
        fig_combined = plot_combined_feature_importance(linear_coef, catboost_importance)
        combined_path = output_dir / f"{prefix}_top_combined.png"
        fig_combined.savefig(combined_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_combined)
        saved_files.append(combined_path)
        
    elif linear_coef is not None:
        fig_linear = plot_linear_feature_importance(linear_coef)
        linear_path = output_dir / f"{prefix}_linear.png"
        fig_linear.savefig(linear_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_linear)
        saved_files.append(linear_path)
        
    elif catboost_importance is not None:
        fig_catboost = plot_catboost_feature_importance(catboost_importance)
        catboost_path = output_dir / f"{prefix}_catboost.png"
        fig_catboost.savefig(catboost_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_catboost)
        saved_files.append(catboost_path)
    
    return saved_files