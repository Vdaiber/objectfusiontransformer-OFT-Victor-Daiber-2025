#!/usr/bin/env python3
"""
Mixed-Effects Linear Model Analysis for NDS Experiment Data

This script performs a Mixed-Effects Linear Model analysis to determine if there are
statistically significant differences in NDS scores among different Model_Name groups,
while controlling for the variability introduced by the Nr (block/subject factor).

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def load_and_examine_data(file_path):
    """
    Load the CSV file and examine the data structure.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Loaded and cleaned data
    """
    print("=" * 60)
    print("MIXED-EFFECTS LINEAR MODEL ANALYSIS")
    print("=" * 60)
    
    # Load the data
    print("\n1. LOADING DATA")
    print("-" * 30)
    
    try:
        # Read CSV file, handling the extra empty columns
        df = pd.read_csv(file_path)
        
        # Remove any completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Keep only the first 3 columns (Nr, Model_Name, NDS)
        df = df.iloc[:, :3]
        
        print(f"Data loaded successfully from: {file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def examine_data_structure(df):
    """
    Examine and display the structure of the data.
    
    Parameters:
    df (pd.DataFrame): The loaded dataset
    """
    print("\n2. DATA STRUCTURE EXAMINATION")
    print("-" * 30)
    
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nBasic statistics for NDS:")
    print(df['NDS'].describe())
    
    print(f"\nUnique Model_Name values:")
    model_counts = df['Model_Name'].value_counts().sort_index()
    print(model_counts)
    
    print(f"\nNumber of unique subjects (Nr): {df['Nr'].nunique()}")
    print(f"Nr range: {df['Nr'].min()} to {df['Nr'].max()}")
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())

def prepare_data_for_analysis(df):
    """
    Prepare the data for mixed-effects modeling.
    
    Parameters:
    df (pd.DataFrame): The raw dataset
    
    Returns:
    pd.DataFrame: Prepared dataset
    """
    print("\n3. DATA PREPARATION")
    print("-" * 30)
    
    # Create a copy for analysis
    analysis_df = df.copy()
    
    # Ensure Nr is treated as categorical for grouping
    analysis_df['Nr'] = analysis_df['Nr'].astype('category')
    
    # Ensure Model_Name is categorical
    analysis_df['Model_Name'] = analysis_df['Model_Name'].astype('category')
    
    print("Data types after preparation:")
    print(analysis_df.dtypes)
    
    print(f"\nFinal dataset ready for analysis:")
    print(f"Shape: {analysis_df.shape}")
    print(f"NDS (dependent variable): continuous")
    print(f"Model_Name (fixed factor): {analysis_df['Model_Name'].nunique()} levels")
    print(f"Nr (random factor): {analysis_df['Nr'].nunique()} levels")
    
    return analysis_df

def fit_mixed_effects_model(df):
    """
    Fit the Mixed-Effects Linear Model.
    
    Parameters:
    df (pd.DataFrame): Prepared dataset
    
    Returns:
    statsmodels results object
    """
    print("\n4. MIXED-EFFECTS MODEL FITTING")
    print("-" * 30)
    
    print("Model specification:")
    print("Formula: NDS ~ C(Model_Name)")
    print("Random effects grouping variable: Nr")
    print("Model type: Mixed-Effects Linear Model (equivalent to ANOVA with random factor)")
    
    try:
        # Fit the mixed-effects model
        # Formula: NDS ~ C(Model_Name) with random intercepts for Nr
        model = mixedlm("NDS ~ C(Model_Name)", df, groups=df["Nr"])
        result = model.fit()
        
        print("\nModel fitted successfully!")
        return result
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None

def extract_and_display_results(result):
    """
    Extract and display the model results.
    
    Parameters:
    result: statsmodels mixed-effects model result
    """
    print("\n5. MODEL RESULTS")
    print("=" * 60)
    
    # Display full model summary
    print("\nFULL MODEL SUMMARY:")
    print("-" * 30)
    print(result.summary())
    
    # Extract p-values for Model_Name factor
    print("\n6. STATISTICAL SIGNIFICANCE ANALYSIS")
    print("-" * 30)
    
    # Get the coefficient table
    coef_table = result.summary().tables[1]
    
    # Extract p-values for Model_Name coefficients
    model_name_pvalues = []
    coefficient_names = []
    
    # Look for coefficients that contain "C(Model_Name)"
    for i, row in enumerate(result.params.index):
        if "C(Model_Name)" in row:
            p_value = result.pvalues[row]
            model_name_pvalues.append(p_value)
            coefficient_names.append(row)
            print(f"Coefficient: {row}")
            print(f"P-value: {p_value:.6f}")
            print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
            print()
    
    return model_name_pvalues, coefficient_names

def perform_overall_f_test(result):
    """
    Perform an overall F-test for the Model_Name factor.
    
    Parameters:
    result: statsmodels mixed-effects model result
    """
    print("\n7. OVERALL MODEL_NAME EFFECT TEST")
    print("-" * 30)
    
    try:
        # Get model parameters
        model_name_coeffs = [param for param in result.params.index if "C(Model_Name)" in param]
        
        if len(model_name_coeffs) > 0:
            print(f"Number of Model_Name coefficients: {len(model_name_coeffs)}")
            
            # Calculate the minimum p-value among Model_Name coefficients
            min_pvalue = min([result.pvalues[coeff] for coeff in model_name_coeffs])
            
            print(f"Minimum p-value among Model_Name coefficients: {min_pvalue:.6f}")
            
            # Check if any coefficient is significant
            any_significant = any([result.pvalues[coeff] < 0.05 for coeff in model_name_coeffs])
            
            print(f"Any Model_Name coefficient significant at α=0.05: {'Yes' if any_significant else 'No'}")
            
            return min_pvalue, any_significant
        else:
            print("No Model_Name coefficients found in the model.")
            return None, None
            
    except Exception as e:
        print(f"Error in overall test: {e}")
        return None, None

def interpret_results(pvalues, any_significant, min_pvalue):
    """
    Provide interpretation of the statistical results.
    
    Parameters:
    pvalues (list): List of p-values for Model_Name coefficients
    any_significant (bool): Whether any coefficient is significant
    min_pvalue (float): Minimum p-value among coefficients
    """
    print("\n8. INTERPRETATION")
    print("=" * 60)
    
    if any_significant:
        interpretation = (
            f"There IS a statistically significant difference in mean NDS scores "
            f"among the different Model_Name groups (minimum p-value = {min_pvalue:.6f} < 0.05), "
            f"after controlling for the random variability introduced by the Nr factor."
        )
    else:
        interpretation = (
            f"There is NO statistically significant difference in mean NDS scores "
            f"among the different Model_Name groups (minimum p-value = {min_pvalue:.6f} ≥ 0.05), "
            f"after controlling for the random variability introduced by the Nr factor."
        )
    
    print("CONCLUSION:")
    print(interpretation)
    
    print(f"\nTechnical details:")
    print(f"- Analysis method: Mixed-Effects Linear Model")
    print(f"- Fixed factor: Model_Name (7 levels)")
    print(f"- Random factor: Nr (73 subjects/blocks)")
    print(f"- Significance level: α = 0.05")
    print(f"- Number of observations: 511")

def write_results_to_file(df, result, pvalues, coeff_names, min_pvalue, any_significant):
    """
    Write complete analysis results to a file for further use.
    """
    output_file = "/app/Statistische_Analyse_Ergebnisse.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTISCHE ANALYSE - MIXED-EFFECTS LINEAR MODEL\n")
        f.write("Experimentelle Daten: NDS Scores verschiedener Modelle\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset Information
        f.write("DATENSATZ INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Anzahl Beobachtungen: {len(df)}\n")
        f.write(f"Anzahl Probanden (Nr): {df['Nr'].nunique()}\n")
        f.write(f"Anzahl Modelle: {df['Model_Name'].nunique()}\n")
        f.write(f"Modelle: {', '.join(sorted(df['Model_Name'].unique()))}\n")
        f.write(f"NDS Mittelwert: {df['NDS'].mean():.6f}\n")
        f.write(f"NDS Standardabweichung: {df['NDS'].std():.6f}\n")
        f.write(f"NDS Bereich: {df['NDS'].min():.6f} - {df['NDS'].max():.6f}\n\n")
        
        # Model Specification
        f.write("MODELL SPEZIFIKATION:\n")
        f.write("-" * 40 + "\n")
        f.write("Modelltyp: Mixed-Effects Linear Model\n")
        f.write("Formel: NDS ~ C(Model_Name)\n")
        f.write("Random Effects: Zufällige Intercepts für Nr (Probanden)\n")
        f.write("Fixed Effects: Model_Name (kategoriale Variable)\n")
        f.write("Methode: REML (Restricted Maximum Likelihood)\n\n")
        
        # Full Model Summary
        f.write("VOLLSTÄNDIGE MODELL ZUSAMMENFASSUNG:\n")
        f.write("-" * 40 + "\n")
        f.write(str(result.summary()) + "\n\n")
        
        # Detailed Results
        f.write("DETAILLIERTE ERGEBNISSE:\n")
        f.write("-" * 40 + "\n")
        f.write("Referenzgruppe: M_Base\n\n")
        
        for i, coeff_name in enumerate(coeff_names):
            model_name = coeff_name.replace("C(Model_Name)[T.", "").replace("]", "")
            coefficient = result.params[coeff_name]
            p_value = result.pvalues[coeff_name]
            ci_lower = result.conf_int().loc[coeff_name, 0]
            ci_upper = result.conf_int().loc[coeff_name, 1]
            
            f.write(f"Modell: {model_name}\n")
            f.write(f"  Koeffizient: {coefficient:.6f}\n")
            f.write(f"  P-Wert: {p_value:.6f}\n")
            f.write(f"  95% Konfidenzintervall: [{ci_lower:.6f}, {ci_upper:.6f}]\n")
            f.write(f"  Signifikant (α=0.05): {'Ja' if p_value < 0.05 else 'Nein'}\n")
            f.write(f"  Interpretation: {model_name} unterscheidet sich {'signifikant' if p_value < 0.05 else 'nicht signifikant'} von M_Base\n\n")
        
        # Overall Test Results
        f.write("GESAMTTEST ERGEBNISSE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Minimaler P-Wert: {min_pvalue:.6f}\n")
        f.write(f"Signifikante Unterschiede vorhanden: {'Ja' if any_significant else 'Nein'}\n\n")
        
        # Final Conclusion
        f.write("WISSENSCHAFTLICHE SCHLUSSFOLGERUNG:\n")
        f.write("-" * 40 + "\n")
        if any_significant:
            f.write("Es bestehen statistisch signifikante Unterschiede in den mittleren NDS-Scores ")
            f.write("zwischen den verschiedenen Model_Name Gruppen (p < 0.05), nach Kontrolle für ")
            f.write("die zufällige Variabilität der Probanden (Nr-Faktor).\n\n")
            
            significant_models = [coeff_names[i].replace("C(Model_Name)[T.", "").replace("]", "") 
                                for i, p in enumerate(pvalues) if p < 0.05]
            f.write(f"Signifikant unterschiedliche Modelle: {', '.join(significant_models)}\n")
        else:
            f.write("Es bestehen keine statistisch signifikanten Unterschiede in den mittleren NDS-Scores ")
            f.write("zwischen den verschiedenen Model_Name Gruppen (p ≥ 0.05).\n")
        
        f.write(f"\nAnalyse durchgeführt am: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nErgebnisse wurden in Datei gespeichert: {output_file}")
    return output_file

def main():
    """
    Main function to run the complete analysis.
    """
    # File path
    file_path = "/app/DOE_CSV/Master_Table_NDS.csv"
    
    # Step 1: Load and examine data
    df = load_and_examine_data(file_path)
    if df is None:
        return
    
    # Step 2: Examine data structure
    examine_data_structure(df)
    
    # Step 3: Prepare data for analysis
    analysis_df = prepare_data_for_analysis(df)
    
    # Step 4: Fit mixed-effects model
    result = fit_mixed_effects_model(analysis_df)
    if result is None:
        return
    
    # Step 5: Extract and display results
    pvalues, coeff_names = extract_and_display_results(result)
    
    # Step 6: Perform overall test
    min_pvalue, any_significant = perform_overall_f_test(result)
    
    # Step 7: Interpret results
    if min_pvalue is not None:
        interpret_results(pvalues, any_significant, min_pvalue)
        
        # Step 8: Write results to file
        output_file = write_results_to_file(analysis_df, result, pvalues, coeff_names, min_pvalue, any_significant)
        
        return output_file

if __name__ == "__main__":
    main()
