#!/usr/bin/env python3
"""
Tukey-HSD Post-Hoc Analysis for NDS Model Comparisons

This script performs pairwise comparisons between different Model_Name groups
using Tukey's Honestly Significant Difference (HSD) test to identify which
specific pairs of models show statistically significant differences.

"""

import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """
    Load the CSV file and prepare data for Tukey-HSD analysis.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Prepared dataset
    """
    print("=" * 70)
    print("TUKEY-HSD POST-HOC ANALYSIS")
    print("Pairwise Comparisons of Model_Name Groups")
    print("=" * 70)
    
    print("\n1. LOADING AND PREPARING DATA")
    print("-" * 40)
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Remove any completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Keep only the first 3 columns (Nr, Model_Name, NDS)
        df = df.iloc[:, :3]
        
        print(f"Data loaded successfully from: {file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display basic information
        print(f"\nModel groups and sample sizes:")
        group_counts = df['Model_Name'].value_counts().sort_index()
        for model, count in group_counts.items():
            mean_nds = df[df['Model_Name'] == model]['NDS'].mean()
            print(f"  {model}: n={count}, mean NDS={mean_nds:.6f}")
        
        print(f"\nTotal observations: {len(df)}")
        print(f"Number of model groups: {df['Model_Name'].nunique()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def perform_tukey_hsd(df):
    """
    Perform Tukey-HSD post-hoc test for pairwise comparisons.
    
    Parameters:
    df (pd.DataFrame): Dataset with NDS and Model_Name columns
    
    Returns:
    TukeyHSDResults: Results object from the Tukey test
    """
    print("\n2. PERFORMING TUKEY-HSD POST-HOC TEST")
    print("-" * 40)
    
    print("Test specification:")
    print("- Dependent variable: NDS (continuous)")
    print("- Grouping variable: Model_Name (7 groups)")
    print("- Method: Tukey's Honestly Significant Difference")
    print("- Alpha level: 0.05")
    print("- Multiple comparison correction: Built into Tukey-HSD")
    
    try:
        # Perform Tukey-HSD test
        tukey_results = pairwise_tukeyhsd(
            endog=df['NDS'],           # Dependent variable
            groups=df['Model_Name'],   # Grouping variable
            alpha=0.05                 # Significance level
        )
        
        print("\nTukey-HSD test completed successfully!")
        return tukey_results
        
    except Exception as e:
        print(f"Error performing Tukey-HSD test: {e}")
        return None

def display_tukey_results(tukey_results):
    """
    Display the complete Tukey-HSD results table.
    
    Parameters:
    tukey_results: TukeyHSDResults object
    """
    print("\n3. COMPLETE TUKEY-HSD RESULTS TABLE")
    print("=" * 70)
    
    print(tukey_results)
    
    return tukey_results

def extract_significant_pairs(tukey_results):
    """
    Extract and display only the statistically significant pairwise comparisons.
    
    Parameters:
    tukey_results: TukeyHSDResults object
    
    Returns:
    list: List of significant comparisons
    """
    print("\n4. SIGNIFICANT PAIRWISE COMPARISONS")
    print("=" * 70)
    
    # Get the results as a DataFrame for easier manipulation
    # Access the summary data directly from the tukey_results object
    summary_data = tukey_results.summary().data[1:]  # Skip header row
    
    results_df = pd.DataFrame(summary_data, columns=[
        'group1', 'group2', 'meandiff', 'p_adj', 'lower', 'upper', 'reject'
    ])
    
    # Convert numeric columns to appropriate types
    results_df['meandiff'] = pd.to_numeric(results_df['meandiff'])
    results_df['p_adj'] = pd.to_numeric(results_df['p_adj'])
    results_df['lower'] = pd.to_numeric(results_df['lower'])
    results_df['upper'] = pd.to_numeric(results_df['upper'])
    results_df['reject'] = results_df['reject'].astype(bool)
    
    # Filter for significant comparisons (p_adj < 0.05)
    significant_pairs = results_df[results_df['reject'] == True].copy()
    
    if len(significant_pairs) > 0:
        print("Statistically significant pairs (p-adj < 0.05):")
        print("-" * 50)
        
        significant_list = []
        
        for idx, row in significant_pairs.iterrows():
            group1, group2 = row['group1'], row['group2']
            meandiff = row['meandiff']
            p_adj = row['p_adj']
            lower_ci = row['lower']
            upper_ci = row['upper']
            
            # Determine direction of difference
            if meandiff > 0:
                direction = f"{group1} > {group2}"
                interpretation = f"{group1} has significantly higher NDS than {group2}"
            else:
                direction = f"{group1} < {group2}"
                interpretation = f"{group1} has significantly lower NDS than {group2}"
            
            print(f"\n{group1} vs {group2}:")
            print(f"  Mean difference: {meandiff:.6f}")
            print(f"  P-value (adjusted): {p_adj:.6f}")
            print(f"  95% CI: [{lower_ci:.6f}, {upper_ci:.6f}]")
            print(f"  Direction: {direction}")
            print(f"  Interpretation: {interpretation}")
            
            significant_list.append({
                'pair': f"{group1} vs {group2}",
                'direction': direction,
                'p_adj': p_adj,
                'meandiff': meandiff,
                'interpretation': interpretation
            })
        
        print(f"\nTotal significant comparisons: {len(significant_pairs)} out of {len(results_df)}")
        
    else:
        print("No statistically significant pairwise differences found (p-adj < 0.05)")
        significant_list = []
    
    return significant_list, results_df

def summarize_findings(significant_list, df):
    """
    Provide a comprehensive summary of the post-hoc test findings.
    
    Parameters:
    significant_list (list): List of significant pairwise comparisons
    df (pd.DataFrame): Original dataset for context
    """
    print("\n5. SUMMARY OF FINDINGS")
    print("=" * 70)
    
    if len(significant_list) == 0:
        print("CONCLUSION:")
        print("No statistically significant pairwise differences were found between any")
        print("of the Model_Name groups after Tukey-HSD correction for multiple comparisons.")
        return
    
    # Calculate group means for context
    group_means = df.groupby('Model_Name')['NDS'].mean().sort_values(ascending=False)
    
    print("Model performance ranking (highest to lowest mean NDS):")
    print("-" * 50)
    for i, (model, mean_nds) in enumerate(group_means.items(), 1):
        print(f"{i}. {model}: {mean_nds:.6f}")
    
    print(f"\nSignificant pairwise differences found:")
    print("-" * 50)
    
    # Group findings by better/worse performance
    better_performers = {}
    worse_performers = {}
    
    for comp in significant_list:
        pair_parts = comp['pair'].split(' vs ')
        group1, group2 = pair_parts[0], pair_parts[1]
        
        if comp['meandiff'] > 0:  # group1 > group2
            if group1 not in better_performers:
                better_performers[group1] = []
            better_performers[group1].append(group2)
            
            if group2 not in worse_performers:
                worse_performers[group2] = []
            worse_performers[group2].append(group1)
        else:  # group1 < group2
            if group2 not in better_performers:
                better_performers[group2] = []
            better_performers[group2].append(group1)
            
            if group1 not in worse_performers:
                worse_performers[group1] = []
            worse_performers[group1].append(group2)
    
    # Summary by performance
    if better_performers:
        print("\nModels with significantly BETTER performance:")
        for model, beaten_models in better_performers.items():
            beaten_str = ", ".join(beaten_models)
            print(f"  • {model} significantly outperforms: {beaten_str}")
    
    if worse_performers:
        print("\nModels with significantly WORSE performance:")
        for model, beaten_by in worse_performers.items():
            beaten_by_str = ", ".join(beaten_by)
            print(f"  • {model} significantly underperforms compared to: {beaten_by_str}")
    
    # Generate conclusion sentence
    print(f"\nCONCLUSION:")
    
    best_model = group_means.index[0]  # Highest mean
    worst_model = group_means.index[-1]  # Lowest mean
    
    # Find if best model significantly beats others
    best_beats = better_performers.get(best_model, [])
    
    if len(best_beats) > 0:
        if len(best_beats) == 1:
            conclusion = f"The post-hoc test reveals that the {best_model} model performed significantly better than the {best_beats[0]} model."
        elif len(best_beats) == 2:
            conclusion = f"The post-hoc test reveals that the {best_model} model performed significantly better than the {' and '.join(best_beats)} models."
        else:
            conclusion = f"The post-hoc test reveals that the {best_model} model performed significantly better than {len(best_beats)} other models ({', '.join(best_beats)})."
    else:
        conclusion = f"The post-hoc test reveals significant differences between {len(significant_list)} pairs of models, with {best_model} showing the highest mean performance."
    
    print(conclusion)

def write_results_to_file(tukey_results, significant_list, results_df, df):
    """
    Write complete post-hoc analysis results to a file.
    
    Parameters:
    tukey_results: TukeyHSDResults object
    significant_list (list): List of significant comparisons
    results_df (pd.DataFrame): Complete results table
    df (pd.DataFrame): Original dataset
    """
    output_file = "/app/Tukey_PostHoc_Ergebnisse.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TUKEY-HSD POST-HOC ANALYSE ERGEBNISSE\n")
        f.write("Paarweise Vergleiche der Model_Name Gruppen\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset info
        f.write("DATENSATZ INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Gesamtanzahl Beobachtungen: {len(df)}\n")
        f.write(f"Anzahl Modellgruppen: {df['Model_Name'].nunique()}\n")
        f.write(f"Anzahl möglicher Paarvergleiche: {len(results_df)}\n\n")
        
        # Group means
        f.write("GRUPPENMITTELWERTE:\n")
        f.write("-" * 40 + "\n")
        group_means = df.groupby('Model_Name')['NDS'].mean().sort_values(ascending=False)
        for model, mean_nds in group_means.items():
            f.write(f"{model}: {mean_nds:.6f}\n")
        f.write("\n")
        
        # Complete Tukey results
        f.write("VOLLSTÄNDIGE TUKEY-HSD ERGEBNISSE:\n")
        f.write("-" * 40 + "\n")
        f.write(str(tukey_results) + "\n\n")
        
        # Significant pairs
        f.write("SIGNIFIKANTE PAARVERGLEICHE (p-adj < 0.05):\n")
        f.write("-" * 40 + "\n")
        
        if len(significant_list) > 0:
            for comp in significant_list:
                f.write(f"{comp['pair']}: p-adj = {comp['p_adj']:.6f}\n")
                f.write(f"  {comp['interpretation']}\n\n")
        else:
            f.write("Keine signifikanten Paarvergleiche gefunden.\n\n")
        
        # Summary
        f.write("ZUSAMMENFASSUNG:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Anzahl signifikanter Vergleiche: {len(significant_list)} von {len(results_df)}\n")
        
        f.write(f"\nAnalyse durchgeführt am: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nPost-Hoc Ergebnisse wurden in Datei gespeichert: {output_file}")
    return output_file

def main():
    """
    Main function to run the complete Tukey-HSD post-hoc analysis.
    """
    # File path
    file_path = "/app/DOE_CSV/Master_Table_NDS.csv"
    
    # Step 1: Load and prepare data
    df = load_and_prepare_data(file_path)
    if df is None:
        return
    
    # Step 2: Perform Tukey-HSD test
    tukey_results = perform_tukey_hsd(df)
    if tukey_results is None:
        return
    
    # Step 3: Display complete results
    display_tukey_results(tukey_results)
    
    # Step 4: Extract significant pairs
    significant_list, results_df = extract_significant_pairs(tukey_results)
    
    # Step 5: Summarize findings
    summarize_findings(significant_list, df)
    
    # Step 6: Write results to file
    output_file = write_results_to_file(tukey_results, significant_list, results_df, df)
    
    return output_file

if __name__ == "__main__":
    main()
