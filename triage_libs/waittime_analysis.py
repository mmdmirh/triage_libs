import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


class WaitTimeAnalyzer:
    """
    Analyzes patient wait times to identify breaches against target times.
    Focus is on Business Days for accurate comparison with 'Auth to Appt'.
    """

    def __init__(self):
        pass

    def parse_target_days(self, df, priority_col='Priority Type (RFL)', urgent_default=10, semi_default=22):
        """
        Parses the 'Priority Type (RFL)' column to extract target Business Days.
        
        Logic:
        - "Urgent ... X days" -> X
        - "Urgent ... X weeks" -> X * 5 (Business Days)
        - "Urgent ... X months" -> X * 22 (Business Days)
        - "Urgent" (plain) -> urgent_default (default 10 business days ~ 2 weeks)
        - "Semi-urgent" (plain) -> semi_default (default 22 business days ~ 1 month)
        
        Args:
            df (pd.DataFrame): Input dataframe.
            priority_col (str): Column containing priority strings.
            urgent_default (int): Default business days for plain "Urgent" (default 10).
            semi_default (int): Default business days for plain "Semi-urgent" (default 22).
            
        Returns:
            pd.DataFrame: DataFrame with new 'Target_Days' column.
        """
        # Work on a copy
        df = df.copy()
        
        def extract_days(val):
            s_val = str(val).lower()
            
            # 1. Regex for explicit timeframes
            # Match number followed by unit (days, weeks, months)
            # e.g. "urgent 05 - 5 days", "semi-urgent 03 - 3 months"
            match = re.search(r'(\d+)\s*(days?|weeks?|months?)', s_val)
            
            if match:
                number = int(match.group(1))
                unit = match.group(2)
                
                if 'day' in unit:
                    return number
                elif 'week' in unit:
                    return number * 5 # Business Days
                elif 'month' in unit:
                    return number * 22 # Business Days
            
            # 2. Defaults for plain types
            if 'semi' in s_val:
                return semi_default
            elif 'urgent' in s_val:
                return urgent_default
            
            return np.nan

        df['Target_Days'] = df[priority_col].apply(extract_days)
        return df

    def identify_breaches(self, df, actual_wait_col='Auth to Appt', target_col='Target_Days'):
        """
        Compares actual wait time vs target to identify breaches.
        
        Args:
            df (pd.DataFrame): Dataframe with target days (from parse_target_days).
            actual_wait_col (str): Column with actual wait times.
            target_col (str): Column with target wait times.
            
        Returns:
            pd.DataFrame: DataFrame with 'Is_Breach' (bool) and 'Breach_Margin' (float) columns.
        """
        df = df.copy()
        
        # Ensure numeric types
        df[actual_wait_col] = pd.to_numeric(df[actual_wait_col], errors='coerce')
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        
        # Drop rows where either value is NaN (invalid/missing)
        original_len = len(df)
        df = df.dropna(subset=[actual_wait_col, target_col])
        dropped_len = original_len - len(df)
        
        if dropped_len > 0:
            print(f"Dropped {dropped_len} rows due to missing/invalid wait time or target.")
        
        # Identify Breach: Actual > Target
        df['Is_Breach'] = df[actual_wait_col] > df[target_col]
        
        # Breach Margin: Positive means late, Negative means early
        df['Breach_Margin'] = df[actual_wait_col] - df[target_col]
        
        return df

    def generate_summary_report(self, df, group_by_col='Priority Type (RFL)'):
        """
        Aggregates data to report breaches and on-time counts per Patient Type.
        
        Args:
            df (pd.DataFrame): Processed dataframe with breach info.
            group_by_col (str): Column to group by (default 'Priority Type (RFL)').
            
        Returns:
            pd.DataFrame: Summary table with Breach_Count and On_Time_Count.
        """
        if group_by_col not in df.columns:
            # Fallback if column missing (shouldn't happen for Priority Type)
            return pd.DataFrame()

        # Group by Patient Type
        # Is_Breach is boolean: sum() = count of True (Breaches)
        # count() = Total
        agg_funcs = {
            'Is_Breach': ['count', 'sum']
        }
        
        summary = df.groupby(group_by_col).agg(agg_funcs)
        
        # Flatten columns
        summary.columns = ['Total', 'Breach_Count']
        
        # Calculate "Unbreached" (Within Target)
        summary['On_Time_Count'] = summary['Total'] - summary['Breach_Count']
        
        # Calculate Grand Total for Global Percentages
        grand_total = summary['Total'].sum()
        
        # 1. Type-based Percentages (Relative to that specific Priority Type)
        summary['Breach_%_Type'] = (summary['Breach_Count'] / summary['Total']) * 100
        summary['OnTime_%_Type'] = (summary['On_Time_Count'] / summary['Total']) * 100
        
        # 2. Global Percentages (Relative to Grand Total of all patients in file)
        summary['Breach_%_Global'] = (summary['Breach_Count'] / grand_total) * 100
        summary['OnTime_%_Global'] = (summary['On_Time_Count'] / grand_total) * 100
        
        # Round all component columns
        summary['Breach_%_Type'] = summary['Breach_%_Type'].round(1)
        summary['OnTime_%_Type'] = summary['OnTime_%_Type'].round(1)
        summary['Breach_%_Global'] = summary['Breach_%_Global'].round(1)
        summary['OnTime_%_Global'] = summary['OnTime_%_Global'].round(1)
        
        # Select and reorder desired columns
        cols = [
            'Breach_Count', 'On_Time_Count', 'Total',
            'Breach_%_Type', 'OnTime_%_Type',
            'Breach_%_Global', 'OnTime_%_Global'
        ]
        summary = summary[cols]
        
        # Sort by most breaches
        summary = summary.sort_values('Breach_Count', ascending=False)
        
        return summary

    def run_analysis_pipeline(self, df, priority_col='Priority Type (RFL)', actual_wait_col='Auth to Appt'):
        """
        Orchestrates the full analysis pipeline.
        Refactored to analyze single file without grouping by clinic.
        """
        print("--- Starting Wait Time Analysis ---")
        
        # Work on a copy
        df = df.copy()

        # 1. Parse Targets
        print(f"Parsing targets from '{priority_col}'...")
        df_processed = self.parse_target_days(df, priority_col=priority_col)
        
        # 2. Identify Breaches
        print(f"Identifying breaches using actual wait '{actual_wait_col}'...")
        df_processed = self.identify_breaches(df_processed, actual_wait_col=actual_wait_col)
        
        # 3. Generate Summary
        print("Generating summary report...")
        summary = self.generate_summary_report(df_processed, group_by_col=priority_col)
        
        print("\n--- Summary Report ---")
        print(summary)
        
        return df_processed, summary

    def plot_breach_margin_histogram(self, df, priority_type='all', bins=20):
        """
        Plots a histogram of Breach Margins (+ve is late, -ve is early).
        
        Args:
            df (pd.DataFrame): Processed dataframe.
            priority_type (str): Specific 'Priority Type (RFL)' to filter by. 
                                 Default 'all' plots for everyone.
            bins (int): Number of histogram bins.
        """
        plt.figure(figsize=(10, 6))
        
        # Filter Data
        if priority_type.lower() != 'all':
            # mask
            mask = df['Priority Type (RFL)'] == priority_type
            plot_data = df[mask].copy() # Explicit copy
            title_suffix = f"for '{priority_type}'"
        else:
            plot_data = df.copy() # Explicit copy
            title_suffix = "for All Priority Types"
            
        if len(plot_data) == 0:
            print(f"No data found for Priority Type: {priority_type}")
            return
            
        # Ensure Breach_Margin is numeric just in case
        plot_data['Breach_Margin'] = pd.to_numeric(plot_data['Breach_Margin'], errors='coerce')
        plot_data = plot_data.dropna(subset=['Breach_Margin'])
        
        # Reset index to avoid any alignment issues with seaborn
        plot_data = plot_data.reset_index(drop=True)
            
        # Plot
        # We plot 'Breach_Margin'
        sns.histplot(data=plot_data, x='Breach_Margin', bins=bins, kde=True, color='skyblue', edgecolor='black')
        
        # Add a vertical line at 0 (Target)
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Target Date (0)')
        
        plt.title(f"Distribution of Breach Margins {title_suffix}")
        plt.xlabel("Days Over/Under Target\n(Positive = Late/Breach, Negative = Early/On-Time)")
        plt.ylabel("Number of Patients")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
