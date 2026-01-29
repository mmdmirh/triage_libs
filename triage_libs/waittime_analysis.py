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

    def parse_target_days(self, df, priority_col='Priority Type (RFL)', urgent_default=14, semi_default=30):
        """
        Parses the 'Priority Type (RFL)' column to extract target Days (Calendar Days).
        
        Logic:
        - "Urgent ... X days" -> X
        - "Urgent ... X weeks" -> X * 7 (Days)
        - "Urgent ... X months" -> X * 30 (Days)
        - "Urgent" (plain) -> urgent_default (default 14 days ~ 2 weeks)
        - "Semi-urgent" (plain) -> semi_default (default 30 days ~ 1 month)
        
        Args:
            df (pd.DataFrame): Input dataframe.
            priority_col (str): Column containing priority strings.
            urgent_default (int): Default days for plain "Urgent" (default 14).
            semi_default (int): Default days for plain "Semi-urgent" (default 30).
            
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
                    return number * 7 # Calendar Days
                elif 'month' in unit:
                    return number * 30 # Calendar Days (approx)
            
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
        
        # 1. Type-based Percentages (Relative to that specific Priority Type)
        summary['Breach_%_Type'] = (summary['Breach_Count'] / summary['Total']) * 100
        
        # Round 
        summary['Breach_%_Type'] = summary['Breach_%_Type'].round(1)
        
        # Select and reorder desired columns
        cols = [
            'Breach_Count', 'Total', 'Breach_%_Type'
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

    def plot_breach_margin_histogram(self, df, priority_type='all', bins=20, save_path=None):
        """
        Plots a histogram of Breach Margins (+ve is late, -ve is early).
        
        Args:
            df (pd.DataFrame): Processed dataframe.
            priority_type (str): Specific 'Priority Type (RFL)' to filter by. 
                                 Default 'all' plots for everyone.
            bins (int): Number of histogram bins.
            save_path (str): Optional path to save the plot image.
        """
        plt.figure(figsize=(10, 6))
        
        # Filter Data
        if priority_type.lower() != 'all':
            # mask: Case-insensitive partial match
            # e.g. "urgent" will match "Urgent", "Urgent 05", etc.
            mask = df['Priority Type (RFL)'].astype(str).str.lower().str.contains(priority_type.lower(), na=False)
            plot_data = df[mask].copy() # Explicit copy
            title_suffix = f"for types containing '{priority_type}'"
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
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
        plt.show()

    def generate_yearly_breach_report(self, df, date_col='Created', priority_col='Priority Type (RFL)', start_month=1):
        """
        Generates a summary report grouped by Year and Priority Type.
        Supports custom start month for Fiscal/Academic Years.
        
        Args:
            df (pd.DataFrame): Processed dataframe.
            date_col (str): Column containing the date to extract year from (default 'Created').
            priority_col (str): Priority column.
            start_month (int): Month number (1-12) to start the year. 
                               Default 1 (Calendar Year).
                               If > 1, returns range formatted years (e.g. "2023-2024").
            
        Returns:
            pd.DataFrame: Summary table grouped by Year and Priority.
        """
        df = df.copy()
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Drop rows with no date
        df = df.dropna(subset=[date_col])
        
        # Calculate Year
        if start_month == 1:
            # Standard Calendar Year
            df['Year'] = df[date_col].dt.year.astype(str)
        else:
            # Custom Fiscal/Academic Year
            # If month < start_month, it belongs to the *previous* year's cycle start
            # Example: start_month=9 (Sep). Date=Aug 2023. Month(8) < 9. Effective Year = 2022. Label "2022-2023"
            #                                Date=Sep 2023. Month(9) >= 9. Effective Year = 2023. Label "2023-2024"
            
            def get_fiscal_year(d):
                eff_year = d.year if d.month >= start_month else d.year - 1
                return f"{eff_year}-{eff_year + 1}"
                
            df['Year'] = df[date_col].apply(get_fiscal_year)
        
        # Group by Year AND Priority
        agg_funcs = {
            'Is_Breach': ['count', 'sum']
        }
        
        summary = df.groupby(['Year', priority_col]).agg(agg_funcs)
        
        # Flatten columns
        summary.columns = ['Total', 'Breach_Count']
        
        # Calculate Percentages (Relative to that Year+Type group)
        summary['Breach_%'] = (summary['Breach_Count'] / summary['Total']) * 100
        
        # Round
        summary['Breach_%'] = summary['Breach_%'].round(1)
        
        # Reorder
        summary = summary[['Breach_Count', 'Total', 'Breach_%']]
        
        return summary.reset_index()

    def plot_yearly_breach_trend(self, summary_df, priority_col='Priority Type (RFL)', y_col='Breach_%', clinic_name=None, save_path=None):
        """
        Plots the trend of a metric over years for each priority type using a Grouped Bar Chart.
        
        Args:
            summary_df (pd.DataFrame): Output from generate_yearly_breach_report.
            priority_col (str): Column name for priority type (used for color/hue).
            y_col (str): Column to plot on Y-axis (e.g. 'Breach_%', 'Total', 'Breach_Count').
            clinic_name (str): Optional. Filename or clinic name to display in title. 
                               If a path is provided, it handles cleaning the name.
            save_path (str): Optional path to save the plot image.
        """
        plt.figure(figsize=(12, 6))
        
        # Ensure data is safe
        plot_data = summary_df.copy()
        
        # Check if cols exist
        if priority_col not in plot_data.columns:
            print(f"Error: Column '{priority_col}' not found in summary dataframe.")
            return
        if y_col not in plot_data.columns:
            print(f"Error: Column '{y_col}' not found. Available: {list(plot_data.columns)}")
            return

        # Plot: Grouped Bar Chart
        sns.barplot(data=plot_data, x='Year', y=y_col, hue=priority_col, palette='viridis', edgecolor='black')
        
        # Construct Title
        title_text = f"Yearly {y_col} by Priority Type"
        if clinic_name:
            title_text += f" - {clinic_name}"
            
        plt.title(title_text)
        plt.xlabel("Year")
        plt.ylabel(y_col)
        
        # Move legend outside if there are many types
        plt.legend(title="Priority Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
        plt.show()

    def save_to_excel(self, df, filepath, sheet_name='Sheet1', index=False):
        """
        Saves a DataFrame to an Excel file.
        
        Args:
            df (pd.DataFrame): Dataframe to save.
            filepath (str): Target file path (should end in .xlsx).
            sheet_name (str): Name of the Excel sheet (default 'Sheet1').
            index (bool): Whether to include the index (default False).
        """
        try:
            df.to_excel(filepath, sheet_name=sheet_name, index=index)
            print(f"Successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")
