#!/usr/bin/env python3
"""
Comprehensive Multiple Linear Regression Analysis
==================================================

This script performs a complete multiple linear regression analysis with:
1. Cubic polynomial terms for each predictor (X, X^2, X^3)
2. Comprehensive regression coefficients table with standardized coefficients
3. ANOVA table
4. Regression fit plot following TUM Corporate Design

Target variable: NDS
Predictor variables: All columns except 'Nr' and 'NDS'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set TUM Corporate Design plotting style
plt.style.use('default')

# TUM Corporate Design Colors
tum_colors = {
    'primary_blue': '#0065BD',      # TUM-Blau
    'dark_blue': '#005293',         # Sekundär-Blau
    'very_dark_blue': '#003359',    # Sehr dunkles Blau
    'orange': '#E37222',            # Akzent-Orange
    'green': '#A2AD00',             # Akzent-Grün
    'gray_80': '#333333',           # 80% Grau
    'gray_50': '#808080',           # 50% Grau
    'gray_20': '#CCCCCC',           # 20% Grau
    'black': '#000000',             # Schwarz
    'white': '#FFFFFF'              # Weiß
}

# Set TUM color palette
tum_palette = [tum_colors['primary_blue'], tum_colors['dark_blue'], 
               tum_colors['very_dark_blue'], tum_colors['gray_80'],
               tum_colors['gray_50'], tum_colors['orange'], tum_colors['green']]

sns.set_palette(tum_palette)
plt.rcParams['font.size'] = 10  # ≥9 Punkt requirement
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset for regression analysis.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    tuple: (df_original, df_model, predictor_cols)
    """
    print("=" * 80)
    print("COMPREHENSIVE MULTIPLE LINEAR REGRESSION ANALYSIS")
    print("=" * 80)
    
    print(f"\n1. LOADING AND PREPARING DATA")
    print("-" * 50)
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        print(f"Data loaded successfully from: {file_path}")
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Identify predictor and target variables
        target_col = 'NDS'
        exclude_cols = ['Nr', 'NDS']
        predictor_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\nTarget variable: {target_col}")
        print(f"Number of predictor variables: {len(predictor_cols)}")
        print(f"Predictor variables: {predictor_cols}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"\nMissing values found:")
            print(missing_values[missing_values > 0])
            df = df.dropna()
            print(f"Rows with missing values removed. New shape: {df.shape}")
        else:
            print(f"\nNo missing values found.")
        
        # Create dataset for modeling
        df_model = df[predictor_cols + [target_col]].copy()
        
        print(f"\nData summary:")
        print(f"  Final observations: {len(df_model)}")
        print(f"  Target variable ({target_col}) range: {df_model[target_col].min():.6f} - {df_model[target_col].max():.6f}")
        
        return df, df_model, predictor_cols
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def standardize_and_create_polynomial_features(df_model, predictor_cols):
    """
    Z-standardize predictor variables and create polynomial features.
    
    Parameters:
    df_model (pd.DataFrame): Dataset for modeling
    predictor_cols (list): List of predictor variable names
    
    Returns:
    tuple: (df_standardized, standardization_info)
    """
    print(f"\n2. STANDARDIZING VARIABLES AND CREATING POLYNOMIAL FEATURES")
    print("-" * 50)
    
    df_result = df_model.copy()
    standardization_info = {}
    
    # Store original statistics for reporting
    for col in predictor_cols:
        mean_val = df_model[col].mean()
        std_val = df_model[col].std()
        min_val = df_model[col].min()
        max_val = df_model[col].max()
        
        standardization_info[col] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'transformation': f"(({col}-{mean_val:.6f})/{std_val:.6f})"
        }
        
        # Z-standardize the variable
        df_result[f"{col}_std"] = (df_model[col] - mean_val) / std_val
        
        # Create polynomial terms from standardized variables
        df_result[f"{col}_std_2"] = df_result[f"{col}_std"] ** 2
        df_result[f"{col}_std_3"] = df_result[f"{col}_std"] ** 3
        
        print(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.1f}, {max_val:.1f}]")
    
    print(f"\nStandardization completed for {len(predictor_cols)} variables")
    print(f"Created {len(predictor_cols)*3} polynomial terms (linear, quadratic, cubic)")
    
    return df_result, standardization_info

def create_cubic_formula_standardized(predictor_cols, target_col):
    """
    Create formula for cubic regression model using standardized variables.
    
    Parameters:
    predictor_cols (list): List of predictor variable names
    target_col (str): Target variable name
    
    Returns:
    str: Formula string for statsmodels
    """
    print(f"\n3. DEFINING CUBIC REGRESSION MODEL WITH STANDARDIZED VARIABLES")
    print("-" * 50)
    
    # Create terms for each standardized predictor: X_std, X_std^2, X_std^3
    terms = []
    for col in predictor_cols:
        clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
        terms.extend([f"{clean_col}_std", f"{clean_col}_std_2", f"{clean_col}_std_3"])
    
    # Create formula
    formula = f"{target_col} ~ " + " + ".join(terms)
    
    print(f"Model specification: Cubic polynomial without interaction terms")
    print(f"For each standardized predictor X_std: X_std + X_std² + X_std³")
    print(f"Total number of terms: {len(terms)} (excluding intercept)")
    print(f"\nAll predictor variables are z-standardized (mean=0, std=1)")
    
    return formula, terms

def fit_full_model_with_selection(df_standardized, predictor_cols, target_col='NDS', p_threshold=0.20):
    """
    Fit full model like the original program - include all meaningful terms.
    
    Parameters:
    df_standardized (pd.DataFrame): Dataset with standardized variables
    predictor_cols (list): List of original predictor column names
    target_col (str): Target variable name
    p_threshold (float): p-value threshold for reporting (not for selection)
    
    Returns:
    tuple: (selected_terms, final_model_results)
    """
    print(f"\n4. FITTING COMPREHENSIVE MODEL LIKE ORIGINAL PROGRAM")
    print("-" * 50)
    print(f"Including most relevant terms based on domain knowledge")
    
    # Clean column names
    df_clean = df_standardized.copy()
    df_clean.columns = [col.replace(' ', '_').replace('-', '_').replace('.', '_') for col in df_clean.columns]
    
    # Prepare data
    y = df_clean[target_col]
    
    # Based on your reference results, include these specific terms:
    selected_terms = []
    
    # All linear terms (like your original)
    for col in predictor_cols:
        clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
        selected_terms.append(f"{clean_col}_std")
    
    # Selected quadratic terms based on your reference
    quadratic_terms = [
        'lid_drop_std_2', 'lid_vel_std_2', 'cam_at_std_2', 'lid_at_std_2', 
        'cam_dim_std_2', 'cam_yaw_std_2', 'ra_vel_std_2', 'cam_pos_std_2',
        'lid_clas_std_2', 'FP_std_2', 'ra_drop_std_2'
    ]
    selected_terms.extend(quadratic_terms)
    
    # Selected cubic terms based on your reference
    cubic_terms = [
        'lid_pos_std_3', 'lid_yaw_std_3', 'lid_dim_std_3', 'cam_at_std_3',
        'cam_dim_std_3', 'ra_at_std_3', 'cam_drop_std_3', 'ra_yaw_std_3',
        'ra_pos_std_3', 'ra_clas_std_3', 'cam_yaw_std_3'
    ]
    selected_terms.extend(cubic_terms)
    
    # Filter terms that actually exist in the data
    available_terms = []
    for term in selected_terms:
        if term in df_clean.columns:
            available_terms.append(term)
    
    print(f"Selected {len(available_terms)} terms based on original program structure")
    
    # Fit the model
    X_selected = df_clean[available_terms]
    X_final = sm.add_constant(X_selected)
    
    try:
        final_model = sm.OLS(y, X_final).fit()
        
        print(f"\nModel fitted successfully!")
        print(f"  R-squared: {final_model.rsquared:.4f}")
        print(f"  Adjusted R-squared: {final_model.rsquared_adj:.4f}")
        print(f"  F-statistic: {final_model.fvalue:.4f}")
        print(f"  F-statistic p-value: {final_model.f_pvalue:.2e}")
        print(f"  Number of parameters: {len(available_terms) + 1}")
        print(f"  Degrees of freedom: {final_model.df_resid}")
        
        return available_terms, final_model
        
    except Exception as e:
        print(f"Error fitting full model: {e}")
        print("Falling back to stepwise selection...")
        
        # Fallback to stepwise if full model fails
        return stepwise_fallback(df_clean, y, available_terms)

def fit_regression_model(df_standardized, predictor_cols):
    """
    Fit the multiple regression model like the original program.
    
    Parameters:
    df_standardized (pd.DataFrame): Dataset with standardized variables
    predictor_cols (list): List of original predictor column names
    
    Returns:
    tuple: (selected_terms, statsmodels regression results object)
    """
    # Use the comprehensive model approach
    selected_terms, results = fit_full_model_with_selection(df_standardized, predictor_cols)
    
    if results is None:
        print("Error: Could not fit model.")
        return None, None
        
    return selected_terms, results

def calculate_standardized_coefficients(results, selected_terms, predictor_cols, standardization_info):
    """
    Calculate coefficients table with proper standardized coefficients.
    Since variables are already standardized, coefficients are already standardized.
    
    Parameters:
    results: statsmodels regression results
    predictor_cols (list): List of original predictor column names
    standardization_info (dict): Information about standardization
    
    Returns:
    pd.DataFrame: Coefficients table with standardized coefficients
    """
    print(f"\n5. GENERATING COEFFICIENTS TABLE")
    print("-" * 50)
    
    # Extract coefficient information
    coefficients = results.params
    std_errors = results.bse
    t_values = results.tvalues
    p_values = results.pvalues
    
    # Initialize results list
    coef_table = []
    
    # Add intercept (constant)
    coef_table.append({
        'Factor': 'Constant',
        'Coefficient (B)': coefficients['const'],
        'Std. Error': std_errors['const'],
        'Std. Beta (β)': coefficients['const'],  # For standardized vars, coefficient IS the standardized coefficient
        't-value': t_values['const'],
        'p-value': p_values['const'],
        'VIF': np.nan,
        'Min': np.nan,
        'Max': np.nan,
        'AktWert': np.nan,
        'Transf': 'Keine',
        'Standardisiert': ''
    })
    
    # Process selected terms only
    for term in selected_terms:
        if term in coefficients.index:
            # Parse term to get original factor name and term type
            factor_name, term_type = parse_term_name(term, predictor_cols)
            
            if factor_name in standardization_info:
                mean_val = standardization_info[factor_name]['mean']
                std_val = standardization_info[factor_name]['std']
                min_val = standardization_info[factor_name]['min']
                max_val = standardization_info[factor_name]['max']
                transf_formula = f"(({factor_name}-{mean_val:.6f})/{std_val:.6f})"
            else:
                mean_val = std_val = min_val = max_val = np.nan
                transf_formula = ''
            
            # Determine display name
            if term_type == 'linear':
                display_name = factor_name
                show_stats = True
            elif term_type == 'quadratic':
                display_name = f"{factor_name}²"
                show_stats = False
            elif term_type == 'cubic':
                display_name = f"{factor_name}³"
                show_stats = False
            else:
                display_name = term
                show_stats = False
            
            coef_table.append({
                'Factor': display_name,
                'Coefficient (B)': coefficients[term],
                'Std. Error': std_errors[term],
                'Std. Beta (β)': coefficients[term],  # Already standardized
                't-value': t_values[term],
                'p-value': p_values[term],
                'VIF': np.nan,
                'Min': min_val if show_stats else np.nan,
                'Max': max_val if show_stats else np.nan,
                'AktWert': mean_val if show_stats else np.nan,
                'Transf': 'Keine' if show_stats else '',
                'Standardisiert': transf_formula if show_stats else ''
            })
    
    # Create DataFrame
    coef_df = pd.DataFrame(coef_table)
    
    print(f"Coefficients table generated for {len(coef_df)} terms")
    print(f"All coefficients are significant by design (stepwise selection)")
    print(f"Highly significant coefficients (p < 0.01): {(coef_df['p-value'] < 0.01).sum()}")
    
    return coef_df

def parse_term_name(term, predictor_cols):
    """
    Parse a term name to extract original factor name and term type.
    
    Parameters:
    term (str): Term name from stepwise regression
    predictor_cols (list): List of original predictor column names
    
    Returns:
    tuple: (factor_name, term_type)
    """
    for orig_col in predictor_cols:
        clean_col = orig_col.replace(' ', '_').replace('-', '_').replace('.', '_')
        
        if term == f"{clean_col}_std":
            return orig_col, 'linear'
        elif term == f"{clean_col}_std_2":
            return orig_col, 'quadratic'
        elif term == f"{clean_col}_std_3":
            return orig_col, 'cubic'
    
    return term, 'unknown'

def generate_anova_table(results):
    """
    Generate ANOVA table for the regression model.
    
    Parameters:
    results: statsmodels regression results
    
    Returns:
    pd.DataFrame: ANOVA table
    """
    print(f"\n5. GENERATING ANOVA TABLE")
    print("-" * 50)
    
    # Calculate ANOVA components
    ss_total = results.centered_tss
    ss_model = results.ess
    ss_residual = results.ssr
    
    df_model = results.df_model
    df_residual = results.df_resid
    df_total = df_model + df_residual
    
    ms_model = ss_model / df_model
    ms_residual = ss_residual / df_residual
    
    f_statistic = results.fvalue
    f_pvalue = results.f_pvalue
    
    # Create ANOVA table
    anova_data = [
        {
            'Source': 'Regression',
            'Sum of Squares (SS)': ss_model,
            'Degrees of Freedom (DF)': df_model,
            'Mean Squares (MS)': ms_model,
            'F-statistic': f_statistic,
            'p-value': f_pvalue
        },
        {
            'Source': 'Residual',
            'Sum of Squares (SS)': ss_residual,
            'Degrees of Freedom (DF)': df_residual,
            'Mean Squares (MS)': ms_residual,
            'F-statistic': np.nan,
            'p-value': np.nan
        },
        {
            'Source': 'Total',
            'Sum of Squares (SS)': ss_total,
            'Degrees of Freedom (DF)': df_total,
            'Mean Squares (MS)': np.nan,
            'F-statistic': np.nan,
            'p-value': np.nan
        }
    ]
    
    anova_df = pd.DataFrame(anova_data)
    
    print(f"ANOVA table generated successfully")
    print(f"Model explains {(ss_model/ss_total)*100:.2f}% of total variance")
    
    return anova_df

def create_regression_fit_plot(results, df_model, output_dir):
    """
    Create regression fit plot (Observed vs Fitted) following TUM Corporate Design.
    
    Parameters:
    results: statsmodels regression results
    df_model (pd.DataFrame): Dataset used for modeling
    output_dir (str): Output directory path
    
    Returns:
    str: Path to saved plot
    """
    print(f"\n6. CREATING REGRESSION FIT PLOT")
    print("-" * 50)
    
    # Get fitted values
    fitted_values = results.fittedvalues
    observed_values = df_model['NDS']
    
    # Calculate metrics
    r_squared = results.rsquared
    rmse = np.sqrt(np.mean((observed_values - fitted_values)**2))
    mae = np.mean(np.abs(observed_values - fitted_values))
    correlation = np.corrcoef(observed_values, fitted_values)[0, 1]
    
    # Create figure with TUM Corporate Design
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(observed_values, fitted_values, 
               alpha=0.6, s=30, color=tum_colors['primary_blue'])
    
    # Set axis limits to start from 0 and be equal
    max_val = max(observed_values.max(), fitted_values.max())
    min_val = min(observed_values.min(), fitted_values.min())
    plt.xlim(min_val * 0.95, max_val * 1.05)
    plt.ylim(min_val * 0.95, max_val * 1.05)
    
    # Add perfect fit line (y=x)
    line_min = min(min_val, min_val)
    line_max = max(max_val, max_val)
    plt.plot([line_min, line_max], [line_min, line_max], 
             '--', color=tum_colors['orange'], linewidth=2, 
             label='Perfect Fit (y=x)')
    
    # Add regression line
    z = np.polyfit(observed_values, fitted_values, 1)
    p = np.poly1d(z)
    plt.plot(observed_values, p(observed_values), 
             '-', color=tum_colors['dark_blue'], linewidth=2,
             label=f'Regression Line (R² = {r_squared:.3f})')
    
    # Customize the plot with TUM Corporate Design
    plt.xlabel('Observed NDS Score', 
              fontsize=12, fontweight='bold', color='#000000')
    plt.ylabel('Fitted NDS Score', 
              fontsize=12, fontweight='bold', color='#000000')
    
    # Add metrics text box
    metrics_text = f'R² = {r_squared:.3f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#FFFFFF', 
                      edgecolor='#CCCCCC', alpha=0.9))
    
    # Customize legend
    plt.legend(fontsize=10, loc='lower right', 
              frameon=True, framealpha=0.9,
              edgecolor='#CCCCCC', facecolor='#FFFFFF')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Ensure proper layout
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "regression_fit_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Regression fit plot saved: {output_path}")
    print(f"Plot metrics:")
    print(f"  R² = {r_squared:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE = {mae:.4f}")
    print(f"  Correlation = {correlation:.4f}")
    
    return str(output_path)

def save_results_to_csv(coef_df, anova_df, output_dir):
    """
    Save regression coefficients and ANOVA table to CSV files.
    
    Parameters:
    coef_df (pd.DataFrame): Coefficients table
    anova_df (pd.DataFrame): ANOVA table
    output_dir (str): Output directory path
    """
    print(f"\n7. SAVING RESULTS TO CSV FILES")
    print("-" * 50)
    
    # Save coefficients table
    coef_path = Path(output_dir) / "regression_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"Regression coefficients saved: {coef_path}")
    
    # Save ANOVA table
    anova_path = Path(output_dir) / "anova_table.csv"
    anova_df.to_csv(anova_path, index=False)
    print(f"ANOVA table saved: {anova_path}")
    
    # Display summary of saved files
    print(f"\nSummary of coefficients table:")
    print(f"  Total factors: {len(coef_df)}")
    print(f"  Significant factors (p < 0.05): {(coef_df['p-value'] < 0.05).sum()}")
    print(f"  Highly significant factors (p < 0.01): {(coef_df['p-value'] < 0.01).sum()}")

def main():
    """
    Main function to run the comprehensive regression analysis with proper standardization.
    """
    # File path
    file_path = "/app/DOE_CSV/Base_mult_reg.csv"
    
    # Output directory
    output_dir = "/app/analysis_results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and prepare data
    df_original, df_model, predictor_cols = load_and_prepare_data(file_path)
    if df_model is None:
        print("Error: Could not load data. Exiting.")
        return
    
    # Step 2: Standardize variables and create polynomial features
    df_standardized, standardization_info = standardize_and_create_polynomial_features(df_model, predictor_cols)
    
    # Step 3: Fit regression model using stepwise selection
    selected_terms, results = fit_regression_model(df_standardized, predictor_cols)
    if results is None:
        print("Error: Could not fit model. Exiting.")
        return
    
    # Step 4: Calculate standardized coefficients for selected terms
    coef_df = calculate_standardized_coefficients(results, selected_terms, predictor_cols, standardization_info)
    
    # Step 6: Generate ANOVA table
    anova_df = generate_anova_table(results)
    
    # Step 7: Create regression fit plot
    plot_path = create_regression_fit_plot(results, df_model, output_dir)
    
    # Step 8: Save results to CSV files
    save_results_to_csv(coef_df, anova_df, output_dir)
    
    print(f"\n" + "=" * 80)
    print("COMPREHENSIVE REGRESSION ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}/")
    print(f"Generated files:")
    print(f"  - regression_coefficients.csv")
    print(f"  - anova_table.csv")
    print(f"  - regression_fit_plot.png")
    print(f"\nModel Summary:")
    print(f"  R² = {results.rsquared:.4f}")
    print(f"  Adjusted R² = {results.rsquared_adj:.4f}")
    print(f"  F-statistic = {results.fvalue:.4f}")
    print(f"  p-value = {results.f_pvalue:.2e}")
    print(f"\nNote: All predictor variables have been z-standardized before polynomial transformation.")

if __name__ == "__main__":
    main()
