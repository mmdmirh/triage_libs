import pandas as pd
import numpy as np
import re


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
        
        # Identify Breach: Actual > Target
        mask_valid = df[actual_wait_col].notna() & df[target_col].notna()
        
        df['Is_Breach'] = False
        df.loc[mask_valid, 'Is_Breach'] = df.loc[mask_valid, actual_wait_col] > df.loc[mask_valid, target_col]
        
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
        
        # Select and reorder desired columns
        summary = summary[['Breach_Count', 'On_Time_Count']]
        
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
