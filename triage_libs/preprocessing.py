import pandas as pd
import holidays
from dateutil.easter import easter
from datetime import timedelta

class Preprocessing:
    def __init__(self):
        self.patient_types = []


    def standardize_patient_types(self, df, column_name='Priority Type (RFL)'):
        """
        Standardizes the patient types in the specified column.
        Logic:
        - If 'semi' is in the value (case-insensitive) -> 'semiurgent'
        - Else if 'urgent' is in the value (case-insensitive) -> 'urgent'
        """
        def standardize(val):
            s_val = str(val).lower()
            if 'semi' in s_val:
                return 'semiurgent'
            elif 'urgent' in s_val:
                return 'urgent'
            return val

        if column_name in df.columns:
            # Use .loc to avoid SettingWithCopyWarning if df is a slice
            df = df.copy() 
            df[column_name] = df[column_name].apply(standardize)
        return df

    def count_arrivals(self, df, date_column='Authorized On', frequency='daily'):
        """
        Counts arrivals per date, week, month, or year for each patient type.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            date_column (str): Name of the date column.
            frequency (str): 'daily', 'weekly', 'monthly', 'yearly'.
                             Weekly starts Monday and ends Sunday.
        
        Returns:
            pd.DataFrame: DataFrame with index as date and columns as patient types (urgent, semiurgent).
        """
        # Ensure working on a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Map frequency to pandas alias
        freq_map = {
            'daily': 'D',
            'weekly': 'W-SUN', # Ends on Sunday, so covers Mon-Sun
            'monthly': 'M',    # Month end
            'yearly': 'Y'      # Year end
        }
        
        if frequency not in freq_map:
            raise ValueError(f"Invalid frequency: {frequency}. Choose from daily, weekly, monthly, yearly.")
            
        freq = freq_map[frequency]
        
        # Group by date and type
        type_col = 'Priority Type (RFL)'
        
        # Using groupby + unstack to get types as columns
        counts = df.groupby([pd.Grouper(key=date_column, freq=freq), type_col]).size().unstack(fill_value=0)
        
        # Fill missing dates using the new method
        counts = self.fill_missing_dates(counts, frequency)

        # Add sum column
        counts['sum'] = counts.sum(axis=1)
        
        counts.index.name = date_column
        return counts

    def fill_missing_dates(self, df, frequency='daily'):
        """
        Fills missing dates in a DataFrame with a DatetimeIndex.
        """
        freq_map = {
            'daily': 'D',
            'weekly': 'W-SUN',
            'monthly': 'M',
            'yearly': 'Y'
        }
        
        if frequency not in freq_map:
             raise ValueError(f"Invalid frequency: {frequency}")
             
        freq = freq_map[frequency]
        
        if not df.empty:
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
            df = df.reindex(full_range, fill_value=0)
            
        return df

    def remove_off_dates(self, df, country='CA', prov='ON'):
        """
        Removes rows where:
        1. Total arrivals (sum) is 0.
        2. AND the date is:
           - Weekend
           - Holiday (defined by country/prov, observed)
           - Civic Holiday (if CA/ON logic applies)
           - Easter Monday
        
        Args:
            df (pd.DataFrame): Input dataframe.
            country (str): Country code for holidays (default 'CA').
            prov (str): Subdivision code for holidays (default 'ON').
        
        Preserves:
        - Any day with arrivals > 0.
        - Weekdays/Business days even if arrivals == 0.
        """
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
             # Try to convert if it's not
             try:
                 df.index = pd.to_datetime(df.index)
             except:
                 raise ValueError("DataFrame index must be DateTime to check off-dates.")

        # load holidays with OBSERVED=True
        # distinct naming to avoid confusion
        local_holidays = holidays.country_holidays(country, subdiv=prov, observed=True)
        
        def is_off_day(date):
            # Weekend: Saturday=5, Sunday=6
            if date.weekday() >= 5:
                return True
            
            # Holiday (Stat + Observed)
            if date in local_holidays:
                return True
            
            # Additional Custom Logic
            # Civic Holiday (First Monday of August) - Common in ON, Canada
            if date.month == 8 and date.weekday() == 0 and 1 <= date.day <= 7:
                return True
            
            # Easter Monday
            # Easter Sunday is calculated by easter(date.year)
            easter_sunday = easter(date.year)
            easter_monday = easter_sunday + timedelta(days=1)
            # compare date.date() with easter_monday
            if date.date() == easter_monday:
                return True
                
            return False

        # Identify rows to remove
        # Condition: sum == 0 AND is_off_day
        # We assume 'sum' column exists or we calculate it
        if 'sum' not in df.columns:
            total_arrivals = df.sum(axis=1) # summation of all columns
        else:
            total_arrivals = df['sum']

        mask_keep = []
        for date, total in zip(df.index, total_arrivals):
             if total == 0 and is_off_day(date):
                 mask_keep.append(False)
             else:
                 mask_keep.append(True)
                 
        return df[mask_keep]
