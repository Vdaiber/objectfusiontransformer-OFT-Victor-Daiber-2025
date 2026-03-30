#!/usr/bin/env python3
"""
Regression Fit Plot - Observed vs Fitted NDS Scores

This script creates an Observed vs Fitted plot to validate the regression models
used in the multi-model robustness analysis. The plot shows how well the fitted
values match the observed NDS scores.

TUM Corporate Design compliant visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

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

def load_regression_data(file_path):
    """
    Load the regression fit data from CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Dataset with Observed and Fitted columns
    """
    print("=" * 60)
    print("REGRESSION FIT PLOT - OBSERVED VS FITTED")
    print("=" * 60)
    
    print(f"\n1. LOADING REGRESSION DATA")
    print("-" * 40)
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Remove any completely empty columns
        df = df.dropna(axis=1, how='all')
        
        print(f"Data loaded successfully from: {file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if we have at least 2 columns for Observed and Fitted data
        if len(df.columns) < 2:
            print(f"Error: Need at least 2 columns, found {len(df.columns)}")
            return None
        
        # Use first two columns as Observed and Fitted
        df_clean = df.iloc[:, :2].copy()
        df_clean.columns = ['Observed', 'Fitted']
        
        # Remove rows with missing values
        df_clean = df_clean.dropna()
        
        print(f"\nData summary:")
        print(f"  Total observations: {len(df_clean)}")
        print(f"  Observed NDS range: {df_clean['Observed'].min():.4f} - {df_clean['Observed'].max():.4f}")
        print(f"  Fitted NDS range: {df_clean['Fitted'].min():.4f} - {df_clean['Fitted'].max():.4f}")
        print(f"  Missing values removed: {len(df) - len(df_clean)}")
        
        return df_clean
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_regression_metrics(df):
    """
    Calculate regression fit metrics.
    
    Parameters:
    df (pd.DataFrame): Dataset with Observed and Fitted columns
    
    Returns:
    dict: Dictionary with R², RMSE, and other metrics
    """
    print(f"\n2. CALCULATING REGRESSION METRICS")
    print("-" * 40)
    
    observed = df['Observed']
    fitted = df['Fitted']
    
    # Calculate R-squared
    ss_res = np.sum((observed - fitted) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((observed - fitted) ** 2))
    
    # Calculate MAE
    mae = np.mean(np.abs(observed - fitted))
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(observed, fitted)[0, 1]
    
    metrics = {
        'R²': r_squared,
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation
    }
    
    print(f"Regression Fit Metrics:")
    print(f"  R² (Coefficient of Determination): {r_squared:.4f}")
    print(f"  RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  Correlation Coefficient: {correlation:.4f}")
    
    return metrics

def create_regression_fit_plot(df, metrics, output_dir):
    """
    Create the Observed vs Fitted plot with TUM Corporate Design.
    
    Parameters:
    df (pd.DataFrame): Dataset with Observed and Fitted columns
    metrics (dict): Regression metrics
    output_dir (str): Output directory path
    """
    print(f"\n3. CREATING REGRESSION FIT PLOT")
    print("-" * 40)
    
    # Create figure with TUM Corporate Design
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(df['Observed'], df['Fitted'], 
               alpha=0.6, s=30, color=tum_colors['primary_blue'])
    
    # Set axis limits to start from 0 and be equal
    max_val = max(df['Observed'].max(), df['Fitted'].max())
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    # Add perfect fit line (y=x) from 0 to max_val
    plt.plot([0, max_val], [0, max_val], 
             '--', color=tum_colors['orange'], linewidth=2, 
             label='Perfect Fit (y=x)')
    
    # Add regression line
    z = np.polyfit(df['Observed'], df['Fitted'], 1)
    p = np.poly1d(z)
    plt.plot(df['Observed'], p(df['Observed']), 
             '-', color=tum_colors['dark_blue'], linewidth=2,
             label=f'Regression Line (R² = {metrics["R²"]:.3f})')
    
    # Customize the plot with TUM Corporate Design (no title)
    plt.xlabel('Observed NDS Score', 
              fontsize=12, fontweight='bold', color='#000000')
    plt.ylabel('Fitted NDS Score', 
              fontsize=12, fontweight='bold', color='#000000')
    
    # Add R² text box only
    metrics_text = f'R² = {metrics["R²"]:.3f}'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#FFFFFF', 
                      edgecolor='#CCCCCC', alpha=0.9))
    
    # Customize legend
    plt.legend(fontsize=10, loc='lower right', 
              frameon=True, framealpha=0.9,
              edgecolor='#CCCCCC', facecolor='#FFFFFF')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#CCCCCC')
    
    # Set background to white
    plt.gca().set_facecolor('#FFFFFF')
    plt.gcf().set_facecolor('#FFFFFF')
    
    # Set equal aspect ratio for better visualization (45° line)
    plt.axis('equal')
    plt.axis('square')
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir) / 'regression_fit_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='#FFFFFF', edgecolor='none')
    plt.close()
    
    print(f"  Plot saved: {output_path}")
    
    return output_path

def generate_summary_report(df, metrics, output_dir):
    """
    Generate a summary report of the regression fit analysis.
    
    Parameters:
    df (pd.DataFrame): Dataset with Observed and Fitted columns
    metrics (dict): Regression metrics
    output_dir (str): Output directory path
    """
    print(f"\n4. GENERATING SUMMARY REPORT")
    print("-" * 40)
    
    output_path = Path(output_dir) / 'regression_fit_summary.txt'
    
    with open(output_path, 'w') as f:
        f.write("REGRESSION FIT ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("PURPOSE:\n")
        f.write("-" * 20 + "\n")
        f.write("This analysis validates the quality of the polynomial regression models\n")
        f.write("used in the multi-model robustness comparison. The Observed vs Fitted\n")
        f.write("plot shows how well the fitted values match the actual NDS scores.\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Total observations: {len(df)}\n")
        f.write(f"Observed NDS range: {df['Observed'].min():.4f} - {df['Observed'].max():.4f}\n")
        f.write(f"Fitted NDS range: {df['Fitted'].min():.4f} - {df['Fitted'].max():.4f}\n\n")
        
        f.write("REGRESSION METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"R² (Coefficient of Determination): {metrics['R²']:.4f}\n")
        f.write(f"RMSE (Root Mean Square Error): {metrics['RMSE']:.4f}\n")
        f.write(f"MAE (Mean Absolute Error): {metrics['MAE']:.4f}\n")
        f.write(f"Correlation Coefficient: {metrics['Correlation']:.4f}\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-" * 15 + "\n")
        if metrics['R²'] > 0.9:
            f.write("Excellent fit: R² > 0.9 indicates very good model performance.\n")
        elif metrics['R²'] > 0.8:
            f.write("Good fit: R² > 0.8 indicates good model performance.\n")
        elif metrics['R²'] > 0.7:
            f.write("Acceptable fit: R² > 0.7 indicates acceptable model performance.\n")
        else:
            f.write("Poor fit: R² < 0.7 indicates poor model performance.\n")
        
        f.write(f"The regression models explain {metrics['R²']*100:.1f}% of the variance in NDS scores.\n")
        f.write(f"Average prediction error: {metrics['MAE']:.4f} NDS units.\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write("-" * 15 + "\n")
        f.write("  regression_fit_plot.png\n")
        f.write("  regression_fit_summary.txt (this file)\n\n")
        
        f.write(f"Analysis completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Summary report saved: {output_path}")

def main():
    """
    Main function to run the complete regression fit analysis.
    """
    # File path
    file_path = "/app/DOE_CSV/Regression Fit Plot.csv"
    
    # Output directory
    output_dir = "../analysis_plots"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load regression data
    df = load_regression_data(file_path)
    if df is None:
        print("Error: Could not load data. Exiting.")
        return
    
    # Step 2: Calculate regression metrics
    metrics = calculate_regression_metrics(df)
    
    # Step 3: Create regression fit plot
    plot_path = create_regression_fit_plot(df, metrics, output_dir)
    
    # Step 4: Generate summary report
    generate_summary_report(df, metrics, output_dir)
    
    print(f"\n" + "=" * 60)
    print("REGRESSION FIT ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"All outputs saved to: {output_dir}/")
    print(f"Generated files:")
    print(f"  - regression_fit_plot.png")
    print(f"  - regression_fit_summary.txt")

if __name__ == "__main__":
    main()
