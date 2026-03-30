#!/usr/bin/env python3
"""
Comparative Interaction Plots Analysis for D-Optimal Experiment

This script creates four comparative interaction plots to visualize how the robustness 
of 7 different models varies in response to four critical disturbance factors.
Uses polynomial regression modeling to generate clean main effect curves.

Author: Data Science Analysis
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import os
from pathlib import Path
import warnings
import pickle

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

# Set TUM color palette with additional colors for better distinction
tum_palette = [tum_colors['primary_blue'], tum_colors['orange'], 
               tum_colors['dark_blue'], tum_colors['green'],
               tum_colors['very_dark_blue'], tum_colors['gray_80'], tum_colors['gray_50']]

sns.set_palette(tum_palette)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9  # 9 Punkt requirement
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

def setup_environment():
    """
    Setup the analysis environment and create output directory.
    
    Returns:
    str: Path to the output directory
    """
    print("=" * 80)
    print("COMPARATIVE INTERACTION PLOTS ANALYSIS")
    print("D-Optimal Experiment: Model Robustness Visualization")
    print("=" * 80)
    
    # Define output directory
    output_dir = "/app/analysis_plots"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n1. SETUP COMPLETE")
    print("-" * 40)
    print(f"Output directory: {output_dir}")
    print(f"Directory created/verified: ✓")
    
    return output_dir

def load_experimental_data(file_path):
    """
    Load the complete experimental data from CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Complete experimental dataset
    """
    print(f"\n2. LOADING EXPERIMENTAL DATA")
    print("-" * 40)
    
    try:
        # Load the complete dataset
        df = pd.read_csv(file_path)
        
        # Remove any completely empty columns
        df = df.dropna(axis=1, how='all')
        
        print(f"Data loaded successfully from: {file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Display basic information about the experiment
        print(f"\nExperimental design information:")
        print(f"  Number of models: {df['Model_Name'].nunique()}")
        print(f"  Models: {sorted(df['Model_Name'].unique())}")
        print(f"  Number of experimental runs: {df['Nr'].nunique()}")
        print(f"  Total observations: {len(df)}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        print(f"  Missing values: {missing_values}")
        
        if missing_values > 0:
            print("  Warning: Missing values detected in dataset")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_factor_means(df, key_factors):
    """
    Calculate mean values for all predictor columns for the "what-if" analysis.
    
    Parameters:
    df (pd.DataFrame): Complete experimental dataset
    key_factors (list): List of key factors to analyze
    
    Returns:
    dict: Dictionary of mean values for all factors
    """
    print(f"\n3. CALCULATING FACTOR MEANS")
    print("-" * 40)
    
    # Get all numeric columns except NDS, Model_Name, and Nr
    exclude_cols = ['NDS', 'Model_Name', 'Nr']
    predictor_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    # Calculate means for all predictor variables
    factor_means = {}
    for col in predictor_cols:
        factor_means[col] = df[col].mean()
    
    print(f"Calculated means for {len(factor_means)} predictor variables")
    print(f"Key factors to analyze: {key_factors}")
    
    # Display means for key factors
    print(f"\nMean values for key factors:")
    for factor in key_factors:
        if factor in factor_means:
            print(f"  {factor}: {factor_means[factor]:.4f}")
        else:
            print(f"  Warning: {factor} not found in dataset")
    
    return factor_means, predictor_cols

def fit_polynomial_model(model_data, formula):
    """
    Fit a polynomial regression model for a specific model's data.
    
    Parameters:
    model_data (pd.DataFrame): Data for a specific model
    formula (str): Regression formula
    
    Returns:
    statsmodels regression result object
    """
    try:
        # Fit the polynomial regression model
        model = smf.ols(formula, data=model_data)
        fitted_model = model.fit()
        
        return fitted_model
        
    except Exception as e:
        print(f"    Error fitting model: {e}")
        return None

def generate_prediction_curves(df, key_factors, factor_means, formula):
    """
    Generate prediction curves for each model and each key factor.
    
    Parameters:
    df (pd.DataFrame): Complete experimental dataset
    key_factors (list): List of key factors to analyze
    factor_means (dict): Mean values for all factors
    formula (str): Regression formula
    
    Returns:
    pd.DataFrame: Combined prediction results
    """
    print(f"\n4. GENERATING PREDICTION CURVES")
    print("-" * 40)
    
    # List to store all prediction results
    all_predictions = []
    
    # Get unique models
    models = sorted(df['Model_Name'].unique())
    print(f"Processing {len(models)} models: {models}")
    
    # Number of points for smooth curves
    n_points = 50
    
    # Loop through each model
    for model_name in models:
        print(f"\n  Processing model: {model_name}")
        
        # Filter data for current model
        model_data = df[df['Model_Name'] == model_name].copy()
        print(f"    Data points: {len(model_data)}")
        
        # Fit polynomial regression model
        fitted_model = fit_polynomial_model(model_data, formula)
        
        if fitted_model is None:
            print(f"    Skipping {model_name} due to modeling error")
            continue
            
        print(f"    Model R²: {fitted_model.rsquared:.4f}")
        
        # Generate prediction curves for each key factor
        for factor in key_factors:
            if factor not in df.columns:
                print(f"    Warning: Factor {factor} not found in dataset")
                continue
                
            # Get factor range from the data
            factor_min = df[factor].min()
            factor_max = df[factor].max()
            factor_range = np.linspace(factor_min, factor_max, n_points)
            
            # Create prediction dataset
            pred_data = pd.DataFrame()
            
            # Set all factors to their mean values
            for col, mean_val in factor_means.items():
                pred_data[col] = [mean_val] * n_points
            
            # Vary the current key factor
            pred_data[factor] = factor_range
            
            try:
                # Generate predictions
                predictions = fitted_model.predict(pred_data)
                
                # Store results
                for i, (factor_val, pred_nds) in enumerate(zip(factor_range, predictions)):
                    all_predictions.append({
                        'Model_Name': model_name,
                        'Factor': factor,
                        'Factor_Value': factor_val,
                        'Predicted_NDS': pred_nds
                    })
                
                print(f"    Generated {len(factor_range)} predictions for {factor}")
                
            except Exception as e:
                print(f"    Error generating predictions for {factor}: {e}")
                continue
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    print(f"\nTotal predictions generated: {len(predictions_df)}")
    
    return predictions_df

def create_interaction_plots(predictions_df, key_factors, output_dir, factor_names):
    """
    Create four comparative interaction plots.
    
    Parameters:
    predictions_df (pd.DataFrame): Prediction results
    key_factors (list): List of key factors
    output_dir (str): Output directory path
    """
    print(f"\n5. CREATING INTERACTION PLOTS")
    print("-" * 40)
    
    # Set up the plotting parameters - convert cm to inches (1 cm = 0.393701 inches)
    plt.rcParams['figure.figsize'] = (12 * 0.393701, 7 * 0.393701)  # 12cm x 7cm for plot + legend space
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 8
    
    # Create plots for each key factor
    for factor in key_factors:
        print(f"  Creating plot for factor: {factor}")
        
        # Filter predictions for current factor
        factor_data = predictions_df[predictions_df['Factor'] == factor]
        
        if len(factor_data) == 0:
            print(f"    Warning: No data found for factor {factor}")
            continue
        
        # Create the plot with specified dimensions
        plt.figure(figsize=(12 * 0.393701, 7 * 0.393701))  # 12cm x 7cm for plot + legend space
        
        # Create line plot with TUM Corporate Design - clean lines without markers
        # Define line styles for better differentiation when colors are similar
        line_styles = {
            'M_Base': '-',      # solid
            'M_C_DO': '-',      # solid  
            'M_C_UP': '-',      # solid
            'M_DRO': '-',       # solid
            'M_GT': '--',       # dashed (to differentiate from similar colors)
            'M_K_DO': ':',      # dotted (to differentiate from M_GT)
            'M_K_UP': '-.'      # dash-dot
        }
        
        for i, model in enumerate(factor_data['Model_Name'].unique()):
            model_data = factor_data[factor_data['Model_Name'] == model]
            linestyle = line_styles.get(model, '-')  # default to solid
            plt.plot(model_data['Factor_Value'], model_data['Predicted_NDS'],
                    linestyle=linestyle,
                    linewidth=1.2,
                    label=model)
        
        # Customize the plot with TUM Corporate Design - NO TITLE as requested
        scientific_name = factor_names.get(factor, factor.replace("_", " ").title())
        # NO TITLE - removed as requested
        plt.xlabel(f'{scientific_name} Value', 
                  fontsize=9, fontweight='bold', color='#000000')
        plt.ylabel('Predicted NDS Score', 
                  fontsize=9, fontweight='bold', color='#000000')
        
        # Customize legend with TUM Corporate Design - more elegant and compact
        plt.legend(title='Model', title_fontsize=8, fontsize=7, 
                  loc='center left', bbox_to_anchor=(1, 0.5), 
                  frameon=True, framealpha=0.95,
                  edgecolor='#CCCCCC', facecolor='#FFFFFF',
                  borderpad=0.3, handlelength=1.5, handletextpad=0.4)
        
        # Add grid for better readability with TUM Corporate Design
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#CCCCCC')
        
        # Set background to white
        plt.gca().set_facecolor('#FFFFFF')
        plt.gcf().set_facecolor('#FFFFFF')
        
        # Tight layout for better appearance with legend outside
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # Make space for legend on the right
        
        # Save the plot as PNG
        filename = f"vergleich_plot_{factor}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Save as SVG (vector format - editable in Illustrator/Inkscape)
        svg_filename = f"vergleich_plot_{factor}.svg"
        svg_filepath = os.path.join(output_dir, svg_filename)
        plt.savefig(svg_filepath, format='svg', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Save as PDF (vector format - good for LaTeX)
        pdf_filename = f"vergleich_plot_{factor}.pdf"
        pdf_filepath = os.path.join(output_dir, pdf_filename)
        plt.savefig(pdf_filepath, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Save the matplotlib figure object as pickle file (for Python)
        fig_filename = f"vergleich_plot_{factor}.pkl"
        fig_filepath = os.path.join(output_dir, fig_filename)
        with open(fig_filepath, 'wb') as f:
            pickle.dump(plt.gcf(), f)
        
        print(f"    Plot saved: {filename}")
        print(f"    SVG (vector) saved: {svg_filename}")
        print(f"    PDF saved: {pdf_filename}")
        print(f"    Matplotlib pickle saved: {fig_filename}")
        
        # Close the figure to free memory
        plt.close()
    
    print(f"\nAll plots saved to: {output_dir}")

def generate_summary_report(predictions_df, key_factors, output_dir):
    """
    Generate a summary report of the analysis.
    
    Parameters:
    predictions_df (pd.DataFrame): Prediction results
    key_factors (list): List of key factors
    output_dir (str): Output directory path
    """
    print(f"\n6. GENERATING SUMMARY REPORT")
    print("-" * 40)
    
    report_file = os.path.join(output_dir, "analysis_summary.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPARATIVE INTERACTION PLOTS ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ANALYSIS OVERVIEW:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Analysis type: Polynomial regression-based interaction plots\n")
        f.write(f"Number of models analyzed: {predictions_df['Model_Name'].nunique()}\n")
        f.write(f"Models: {', '.join(sorted(predictions_df['Model_Name'].unique()))}\n")
        f.write(f"Key factors analyzed: {', '.join(key_factors)}\n")
        f.write(f"Total predictions generated: {len(predictions_df)}\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Loaded D-optimal experimental data (73 runs × 7 models = 511 observations)\n")
        f.write("2. Fitted polynomial regression models for each model separately\n")
        f.write("3. Generated prediction curves by varying each key factor while holding others constant\n")
        f.write("4. Created comparative visualization plots for model robustness analysis\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 40 + "\n")
        
        # Calculate some basic statistics for each factor
        for factor in key_factors:
            factor_data = predictions_df[predictions_df['Factor'] == factor]
            if len(factor_data) > 0:
                f.write(f"\n{factor.upper()}:\n")
                
                # Calculate range of predictions for each model
                model_ranges = factor_data.groupby('Model_Name')['Predicted_NDS'].agg(['min', 'max', 'mean'])
                model_ranges['range'] = model_ranges['max'] - model_ranges['min']
                
                f.write(f"  Model sensitivity (NDS range across factor variation):\n")
                for model in model_ranges.index:
                    range_val = model_ranges.loc[model, 'range']
                    mean_val = model_ranges.loc[model, 'mean']
                    f.write(f"    {model}: Range = {range_val:.4f}, Mean = {mean_val:.4f}\n")
                
                # Most/least sensitive model
                most_sensitive = model_ranges['range'].idxmax()
                least_sensitive = model_ranges['range'].idxmin()
                f.write(f"  Most sensitive model: {most_sensitive}\n")
                f.write(f"  Least sensitive model: {least_sensitive}\n")
        
        f.write(f"\nOUTPUT FILES:\n")
        f.write("-" * 40 + "\n")
        for factor in key_factors:
            f.write(f"  vergleich_plot_{factor}.png\n")
        f.write(f"  analysis_summary.txt (this file)\n")
        
        f.write(f"\nAnalysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Summary report saved: analysis_summary.txt")

def main():
    """
    Main function to run the complete interaction plots analysis.
    """
    # Define key factors to analyze with scientific names
    key_factors = ['lid_pos', 'lid_drop', 'cam_at', 'ra_clas']
    
    # Scientific factor names for plot labels
    factor_names = {
        'lid_pos': 'LiDAR Position Noise',
        'lid_drop': 'LiDAR Dropout Rate', 
        'cam_at': 'Camera Attribute Accuracy',
        'ra_clas': 'Radar Classification Accuracy'
    }
    
    # Define the polynomial regression formula
    formula = ("NDS ~ lid_pos + lid_drop + lid_yaw + lid_dim + lid_vel + cam_at + lid_at + "
              "cam_dim + ra_at + cam_drop + ra_yaw + ra_pos + ra_clas + cam_yaw + ra_vel + "
              "cam_pos + lid_clas + FP + ra_drop + cam_clas + I(lid_vel**2) + I(cam_at**2) + "
              "I(lid_at**2) + I(cam_dim**2) + I(cam_yaw**2) + I(ra_vel**2) + I(cam_pos**2) + "
              "I(lid_clas**2) + I(FP**2) + I(ra_drop**2) + I(lid_pos**3) + I(lid_yaw**3) + "
              "I(lid_dim**3) + I(cam_at**3) + I(cam_dim**3) + I(ra_at**3) + I(cam_drop**3) + "
              "I(ra_yaw**3) + I(ra_pos**3) + I(ra_clas**3) + I(cam_yaw**3)")
    
    # File path
    file_path = "/app/DOE_CSV/Master_Table_NDS_full.csv"
    
    # Step 1: Setup environment
    output_dir = setup_environment()
    
    # Step 2: Load experimental data
    df = load_experimental_data(file_path)
    if df is None:
        print("Error: Could not load data. Exiting.")
        return
    
    # Step 3: Calculate factor means
    factor_means, predictor_cols = calculate_factor_means(df, key_factors)
    
    # Step 4: Generate prediction curves
    predictions_df = generate_prediction_curves(df, key_factors, factor_means, formula)
    
    if len(predictions_df) == 0:
        print("Error: No predictions generated. Exiting.")
        return
    
    # Step 5: Create interaction plots
    create_interaction_plots(predictions_df, key_factors, output_dir, factor_names)
    
    # Step 6: Generate summary report
    generate_summary_report(predictions_df, key_factors, output_dir)
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}/")
    print(f"Generated files:")
    for factor in key_factors:
        print(f"  - vergleich_plot_{factor}.png")
    print(f"  - analysis_summary.txt")

if __name__ == "__main__":
    main()
