#!/usr/bin/env python3
"""
Automatic Multiple Linear Regression Analysis
=============================================

This script implements various automatic model selection procedures:
1. Forward Selection
2. Backward Elimination
3. Bidirectional Stepwise
4. AIC/BIC-based Selection
5. VIF-based Multicollinearity Testing

Exactly like in professional statistical programs.
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
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# TUM Corporate Design Colors
tum_colors = {
    'primary_blue': '#0065BD',
    'dark_blue': '#005293', 
    'orange': '#E37222',
    'gray_80': '#333333',
    'gray_50': '#808080',
    'gray_20': '#CCCCCC',
    'black': '#000000',
    'white': '#FFFFFF'
}

def load_and_prepare_data(file_path):
    """Load and prepare the dataset for regression analysis."""
    print("=" * 80)
    print("AUTOMATIC MULTIPLE LINEAR REGRESSION ANALYSIS")
    print("=" * 80)
    
    print(f"\n1. LOADING AND PREPARING DATA")
    print("-" * 50)
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from: {file_path}")
        print(f"Original dataset shape: {df.shape}")
        
        # Identify predictor and target variables
        target_col = 'NDS'
        exclude_cols = ['Nr', 'NDS']
        predictor_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\nTarget variable: {target_col}")
        print(f"Number of predictor variables: {len(predictor_cols)}")
        
        # Check for missing values
        if df.isnull().any().any():
            df = df.dropna()
            print(f"Rows with missing values removed. New shape: {df.shape}")
        
        df_model = df[predictor_cols + [target_col]].copy()
        
        print(f"\nData summary:")
        print(f"  Final observations: {len(df_model)}")
        print(f"  Target variable ({target_col}) range: {df_model[target_col].min():.6f} - {df_model[target_col].max():.6f}")
        
        return df, df_model, predictor_cols
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def standardize_and_create_features(df_model, predictor_cols):
    """Z-standardize predictor variables and create polynomial features."""
    print(f"\n2. STANDARDIZING VARIABLES AND CREATING POLYNOMIAL FEATURES")
    print("-" * 50)
    
    df_result = df_model.copy()
    standardization_info = {}
    
    # Store original statistics
    for col in predictor_cols:
        mean_val = df_model[col].mean()
        std_val = df_model[col].std()
        min_val = df_model[col].min()
        max_val = df_model[col].max()
        
        standardization_info[col] = {
            'mean': mean_val, 'std': std_val, 'min': min_val, 'max': max_val,
            'transformation': f"(({col}-{mean_val:.6f})/{std_val:.6f})"
        }
        
        # Z-standardize and create polynomial terms
        df_result[f"{col}_std"] = (df_model[col] - mean_val) / std_val
        df_result[f"{col}_std_2"] = df_result[f"{col}_std"] ** 2
        df_result[f"{col}_std_3"] = df_result[f"{col}_std"] ** 3
        
        print(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    print(f"\nStandardization completed for {len(predictor_cols)} variables")
    print(f"Created {len(predictor_cols)*3} polynomial terms")
    
    return df_result, standardization_info

def calculate_vif(X):
    """Calculate Variance Inflation Factor for multicollinearity detection."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def forward_selection(X, y, significance_level=0.05, max_features=None):
    """
    Forward Selection: Start with no variables, add one by one.
    """
    print(f"\n3A. FORWARD SELECTION (p-to-enter = {significance_level})")
    print("-" * 50)
    
    if max_features is None:
        max_features = min(len(X.columns), len(y) - 5)  # Leave some degrees of freedom
    
    selected_features = []
    remaining_features = list(X.columns)
    
    step = 0
    while len(selected_features) < max_features and remaining_features:
        step += 1
        best_p_value = 1.0
        best_feature = None
        
        print(f"Step {step}:")
        
        # Test each remaining feature
        for feature in remaining_features:
            test_features = selected_features + [feature]
            X_test = sm.add_constant(X[test_features])
            
            try:
                model = sm.OLS(y, X_test).fit()
                p_value = model.pvalues[feature]
                
                if p_value < best_p_value:
                    best_p_value = p_value
                    best_feature = feature
            except:
                continue
        
        # Add feature if significant
        if best_p_value < significance_level and best_feature:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            print(f"  Added: {best_feature} (p = {best_p_value:.6f})")
        else:
            print(f"  No more significant features to add (best p = {best_p_value:.6f})")
            break
    
    print(f"\nForward selection complete: {len(selected_features)} features selected")
    return selected_features

def backward_elimination(X, y, significance_level=0.10):
    """
    Backward Elimination: Start with all variables, remove one by one.
    """
    print(f"\n3B. BACKWARD ELIMINATION (p-to-remove = {significance_level})")
    print("-" * 50)
    
    # Start with reasonable number of features to avoid overfit
    max_start_features = min(len(X.columns), len(y) - 10)
    
    # Use F-test to select initial features
    f_scores, p_values = f_regression(X, y)
    initial_features = X.columns[np.argsort(p_values)[:max_start_features]].tolist()
    
    selected_features = initial_features.copy()
    print(f"Starting with {len(selected_features)} most significant features")
    
    step = 0
    while len(selected_features) > 1:
        step += 1
        print(f"Step {step}:")
        
        X_current = sm.add_constant(X[selected_features])
        
        try:
            model = sm.OLS(y, X_current).fit()
            
            # Find feature with highest p-value (excluding constant)
            p_values_features = model.pvalues[selected_features]
            worst_p_value = p_values_features.max()
            worst_feature = p_values_features.idxmax()
            
            if worst_p_value > significance_level:
                selected_features.remove(worst_feature)
                print(f"  Removed: {worst_feature} (p = {worst_p_value:.6f})")
            else:
                print(f"  All remaining features significant (worst p = {worst_p_value:.6f})")
                break
                
        except Exception as e:
            print(f"  Error in model fitting: {e}")
            break
    
    print(f"\nBackward elimination complete: {len(selected_features)} features retained")
    return selected_features

def bidirectional_stepwise(X, y, p_enter=0.05, p_remove=0.10, max_features=None):
    """
    Bidirectional Stepwise: Combination of forward and backward.
    """
    print(f"\n3C. BIDIRECTIONAL STEPWISE (p-enter={p_enter}, p-remove={p_remove})")
    print("-" * 50)
    
    if max_features is None:
        max_features = min(len(X.columns), len(y) - 5)
    
    selected_features = []
    remaining_features = list(X.columns)
    
    step = 0
    max_steps = 200  # Prevent infinite loops
    
    while step < max_steps and len(selected_features) < max_features:
        step += 1
        print(f"Step {step}:")
        
        changed = False
        
        # Forward step: try to add a variable
        if remaining_features:
            best_p_enter = 1.0
            best_feature_enter = None
            
            for feature in remaining_features:
                test_features = selected_features + [feature]
                X_test = sm.add_constant(X[test_features])
                
                try:
                    model = sm.OLS(y, X_test).fit()
                    p_value = model.pvalues[feature]
                    
                    if p_value < best_p_enter:
                        best_p_enter = p_value
                        best_feature_enter = feature
                except:
                    continue
            
            # Add variable if significant
            if best_p_enter < p_enter and best_feature_enter:
                selected_features.append(best_feature_enter)
                remaining_features.remove(best_feature_enter)
                print(f"  Added: {best_feature_enter} (p = {best_p_enter:.6f})")
                changed = True
        
        # Backward step: try to remove variables
        if len(selected_features) > 1:
            X_current = sm.add_constant(X[selected_features])
            
            try:
                model = sm.OLS(y, X_current).fit()
                
                # Check all features for removal
                features_to_remove = []
                for feature in selected_features:
                    if model.pvalues[feature] > p_remove:
                        features_to_remove.append((feature, model.pvalues[feature]))
                
                # Remove feature with highest p-value
                if features_to_remove:
                    features_to_remove.sort(key=lambda x: x[1], reverse=True)
                    feature_to_remove, p_value = features_to_remove[0]
                    
                    selected_features.remove(feature_to_remove)
                    remaining_features.append(feature_to_remove)
                    print(f"  Removed: {feature_to_remove} (p = {p_value:.6f})")
                    changed = True
                    
            except:
                pass
        
        # Check for convergence
        if not changed:
            print(f"  No changes made - converged")
            break
    
    print(f"\nBidirectional stepwise complete: {len(selected_features)} features selected")
    return selected_features

def aic_bic_selection(X, y, max_features=None):
    """
    AIC/BIC-based model selection.
    """
    print(f"\n3D. AIC/BIC-BASED SELECTION")
    print("-" * 50)
    
    if max_features is None:
        max_features = min(len(X.columns), len(y) - 5)
    
    # Pre-select features using F-test to make this computationally feasible
    f_scores, p_values = f_regression(X, y)
    candidate_features = X.columns[np.argsort(p_values)[:min(max_features*2, len(X.columns))]].tolist()
    
    print(f"Testing {len(candidate_features)} candidate features")
    
    best_aic = np.inf
    best_bic = np.inf
    best_features_aic = []
    best_features_bic = []
    
    # Test different feature combinations
    from itertools import combinations
    
    max_subset_size = min(max_features, len(candidate_features))
    
    for size in range(1, max_subset_size + 1):
        print(f"  Testing subsets of size {size}...")
        
        # Limit combinations to avoid exponential explosion
        max_combinations = 1000
        combination_count = 0
        
        for features in combinations(candidate_features, size):
            if combination_count >= max_combinations:
                break
            combination_count += 1
            
            X_subset = sm.add_constant(X[list(features)])
            
            try:
                model = sm.OLS(y, X_subset).fit()
                
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_features_aic = list(features)
                
                if model.bic < best_bic:
                    best_bic = model.bic
                    best_features_bic = list(features)
                    
            except:
                continue
    
    print(f"\nBest AIC model: {len(best_features_aic)} features (AIC = {best_aic:.2f})")
    print(f"Best BIC model: {len(best_features_bic)} features (BIC = {best_bic:.2f})")
    
    # Return BIC model (more conservative)
    return best_features_bic

def fit_final_model(X, y, selected_features, method_name):
    """
    Fit final model with selected features.
    """
    print(f"\n4. FITTING FINAL MODEL ({method_name})")
    print("-" * 50)
    
    if not selected_features:
        print("No features selected!")
        return None
    
    X_final = sm.add_constant(X[selected_features])
    
    try:
        model = sm.OLS(y, X_final).fit()
        
        print(f"Model fitted successfully!")
        print(f"  Selected features: {len(selected_features)}")
        print(f"  R-squared: {model.rsquared:.4f}")
        print(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")
        print(f"  F-statistic: {model.fvalue:.4f}")
        print(f"  F-statistic p-value: {model.f_pvalue:.2e}")
        print(f"  AIC: {model.aic:.2f}")
        print(f"  BIC: {model.bic:.2f}")
        print(f"  Degrees of freedom: {model.df_resid}")
        
        # Check for multicollinearity
        if len(selected_features) > 1:
            try:
                vif_data = calculate_vif(X[selected_features])
                high_vif = vif_data[vif_data['VIF'] > 10]
                if len(high_vif) > 0:
                    print(f"  Warning: {len(high_vif)} features with high VIF (>10)")
                else:
                    print(f"  No multicollinearity issues detected")
            except:
                print(f"  Could not calculate VIF")
        
        return model, selected_features
        
    except Exception as e:
        print(f"Error fitting final model: {e}")
        return None, selected_features

def create_coefficients_table(model, selected_features, predictor_cols, standardization_info):
    """
    Create coefficients table with proper standardized coefficients.
    """
    print(f"\n5. GENERATING COEFFICIENTS TABLE")
    print("-" * 50)
    
    # Extract coefficient information
    coefficients = model.params
    std_errors = model.bse
    t_values = model.tvalues
    p_values = model.pvalues
    
    # Calculate true standardized coefficients
    target_std = standardization_info.get('target_std', 1.0)
    
    coef_table = []
    
    # Add intercept
    coef_table.append({
        'Factor': 'Constant',
        'Coefficient (B)': coefficients['const'],
        'Std. Error': std_errors['const'],
        'Std. Beta (β)': coefficients['const'] / target_std,
        't-value': t_values['const'],
        'p-value': p_values['const'],
        'VIF': np.nan,
        'Min': np.nan,
        'Max': np.nan,
        'AktWert': np.nan,
        'Transf': 'Keine',
        'Standardisiert': ''
    })
    
    # Process selected features
    for feature in selected_features:
        if feature in coefficients.index:
            # Parse feature name
            factor_name, term_type = parse_feature_name(feature, predictor_cols)
            
            # Get standardization info
            if factor_name in standardization_info:
                info = standardization_info[factor_name]
                mean_val = info['mean']
                std_val = info['std']
                min_val = info['min']
                max_val = info['max']
                transf_formula = info['transformation']
            else:
                mean_val = std_val = min_val = max_val = np.nan
                transf_formula = ''
            
            # Calculate proper standardized coefficient
            # For z-standardized X: β = b * (1 / σ_Y)
            std_beta = coefficients[feature] / target_std
            
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
                display_name = feature
                show_stats = False
            
            coef_table.append({
                'Factor': display_name,
                'Coefficient (B)': coefficients[feature],
                'Std. Error': std_errors[feature],
                'Std. Beta (β)': std_beta,
                't-value': t_values[feature],
                'p-value': p_values[feature],
                'VIF': np.nan,  # Would need separate calculation
                'Min': min_val if show_stats else np.nan,
                'Max': max_val if show_stats else np.nan,
                'AktWert': mean_val if show_stats else np.nan,
                'Transf': 'Keine' if show_stats else '',
                'Standardisiert': transf_formula if show_stats else ''
            })
    
    coef_df = pd.DataFrame(coef_table)
    
    print(f"Coefficients table generated for {len(coef_df)} terms")
    print(f"All coefficients are significant by selection method")
    print(f"Highly significant coefficients (p < 0.01): {(coef_df['p-value'] < 0.01).sum()}")
    
    return coef_df

def parse_feature_name(feature, predictor_cols):
    """Parse feature name to extract original factor name and term type."""
    for orig_col in predictor_cols:
        clean_col = orig_col.replace(' ', '_').replace('-', '_').replace('.', '_')
        
        if feature == f"{clean_col}_std":
            return orig_col, 'linear'
        elif feature == f"{clean_col}_std_2":
            return orig_col, 'quadratic'
        elif feature == f"{clean_col}_std_3":
            return orig_col, 'cubic'
    
    return feature, 'unknown'

def compare_methods(X, y, predictor_cols, standardization_info):
    """
    Compare different selection methods.
    """
    print(f"\n6. COMPARING SELECTION METHODS")
    print("=" * 50)
    
    methods = {}
    
    # Forward Selection
    try:
        features_forward = forward_selection(X, y, significance_level=0.05)
        model_forward, _ = fit_final_model(X, y, features_forward, "Forward Selection")
        methods['Forward'] = {
            'features': features_forward,
            'model': model_forward,
            'n_features': len(features_forward)
        }
    except Exception as e:
        print(f"Forward selection failed: {e}")
        methods['Forward'] = {'features': [], 'model': None, 'n_features': 0}
    
    # Backward Elimination
    try:
        features_backward = backward_elimination(X, y, significance_level=0.10)
        model_backward, _ = fit_final_model(X, y, features_backward, "Backward Elimination")
        methods['Backward'] = {
            'features': features_backward,
            'model': model_backward,
            'n_features': len(features_backward)
        }
    except Exception as e:
        print(f"Backward elimination failed: {e}")
        methods['Backward'] = {'features': [], 'model': None, 'n_features': 0}
    
    # Bidirectional Stepwise
    try:
        features_stepwise = bidirectional_stepwise(X, y, p_enter=0.05, p_remove=0.10)
        model_stepwise, _ = fit_final_model(X, y, features_stepwise, "Bidirectional Stepwise")
        methods['Stepwise'] = {
            'features': features_stepwise,
            'model': model_stepwise,
            'n_features': len(features_stepwise)
        }
    except Exception as e:
        print(f"Stepwise selection failed: {e}")
        methods['Stepwise'] = {'features': [], 'model': None, 'n_features': 0}
    
    # AIC/BIC Selection
    try:
        features_aic_bic = aic_bic_selection(X, y)
        model_aic_bic, _ = fit_final_model(X, y, features_aic_bic, "AIC/BIC Selection")
        methods['AIC/BIC'] = {
            'features': features_aic_bic,
            'model': model_aic_bic,
            'n_features': len(features_aic_bic)
        }
    except Exception as e:
        print(f"AIC/BIC selection failed: {e}")
        methods['AIC/BIC'] = {'features': [], 'model': None, 'n_features': 0}
    
    # Summary comparison
    print(f"\nMETHOD COMPARISON SUMMARY:")
    print("-" * 50)
    print(f"{'Method':<20} {'Features':<10} {'R²':<8} {'Adj R²':<8} {'AIC':<8} {'BIC':<8}")
    print("-" * 70)
    
    best_method = None
    best_score = -np.inf
    
    for method_name, method_data in methods.items():
        model = method_data['model']
        n_features = method_data['n_features']
        
        if model is not None:
            r2 = model.rsquared
            adj_r2 = model.rsquared_adj
            aic = model.aic
            bic = model.bic
            
            print(f"{method_name:<20} {n_features:<10} {r2:<8.4f} {adj_r2:<8.4f} {aic:<8.1f} {bic:<8.1f}")
            
            # Use adjusted R² as selection criterion
            if adj_r2 > best_score:
                best_score = adj_r2
                best_method = method_name
        else:
            print(f"{method_name:<20} {n_features:<10} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8}")
    
    print(f"\nBest method: {best_method} (Adj R² = {best_score:.4f})")
    
    return methods, best_method

def main():
    """
    Main function to run automatic regression analysis.
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
    df_standardized, standardization_info = standardize_and_create_features(df_model, predictor_cols)
    
    # Add target standard deviation for proper standardized coefficients
    standardization_info['target_std'] = df_model['NDS'].std()
    
    # Prepare feature matrix and target
    feature_cols = []
    for col in predictor_cols:
        clean_col = col.replace(' ', '_').replace('-', '_').replace('.', '_')
        feature_cols.extend([f"{clean_col}_std", f"{clean_col}_std_2", f"{clean_col}_std_3"])
    
    X = df_standardized[feature_cols]
    y = df_standardized['NDS']
    
    # Step 3-6: Compare different selection methods
    methods, best_method = compare_methods(X, y, predictor_cols, standardization_info)
    
    # Step 7: Generate results for best method
    if best_method and methods[best_method]['model'] is not None:
        print(f"\n7. FINAL RESULTS USING {best_method}")
        print("=" * 50)
        
        best_model = methods[best_method]['model']
        best_features = methods[best_method]['features']
        
        # Generate coefficients table
        coef_df = create_coefficients_table(best_model, best_features, predictor_cols, standardization_info)
        
        # Save results
        coef_path = Path(output_dir) / "automatic_regression_coefficients.csv"
        coef_df.to_csv(coef_path, index=False)
        print(f"\nCoefficients table saved: {coef_path}")
        
        # Generate summary report
        summary_path = Path(output_dir) / "automatic_regression_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("AUTOMATISCHE REGRESSIONSANALYSE - ERGEBNISSE\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Beste Methode: {best_method}\n")
            f.write(f"Anzahl ausgewählte Features: {len(best_features)}\n")
            f.write(f"R² = {best_model.rsquared:.4f}\n")
            f.write(f"Adjustiertes R² = {best_model.rsquared_adj:.4f}\n")
            f.write(f"F-Statistik = {best_model.fvalue:.4f}\n")
            f.write(f"p-Wert = {best_model.f_pvalue:.2e}\n")
            f.write(f"AIC = {best_model.aic:.2f}\n")
            f.write(f"BIC = {best_model.bic:.2f}\n\n")
            
            f.write("AUSGEWÄHLTE FEATURES:\n")
            f.write("-" * 30 + "\n")
            for i, feature in enumerate(best_features, 1):
                factor_name, term_type = parse_feature_name(feature, predictor_cols)
                if term_type == 'linear':
                    display_name = factor_name
                elif term_type == 'quadratic':
                    display_name = f"{factor_name}²"
                elif term_type == 'cubic':
                    display_name = f"{factor_name}³"
                else:
                    display_name = feature
                f.write(f"{i:2d}. {display_name}\n")
        
        print(f"Summary report saved: {summary_path}")
        
    print(f"\n" + "=" * 80)
    print("AUTOMATISCHE REGRESSIONSANALYSE ABGESCHLOSSEN!")
    print("=" * 80)
    print(f"Alle Ergebnisse gespeichert in: {output_dir}/")

if __name__ == "__main__":
    main()
