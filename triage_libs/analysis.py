import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
import scipy.stats as stats
import warnings

class TimeSeriesAnalyzer:
    def __init__(self):
        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore")

    def decompose_stl(self, data, pationt_type, period=7, seasonal=13):
        """
        Performs STL Decomposition on the time series.
        
        Args:
            data (pd.Series or pd.DataFrame): Time series data.
            period (int, optional): Periodicity of the sequence (default 7).
            seasonal (int): Seasonal smoother length (must be odd).
            pationt_type (str, mandatory): Specific column to analyze if data is a DataFrame.
        """
        print("\n--- Performing STL Decomposition ---")
        
        # Handle DataFrame input
        if isinstance(data, pd.DataFrame):
            if pationt_type.lower() in data.columns:
                print(f"Using specified column: '{pationt_type}'")
                data = data[pationt_type]
            else:
                raise ValueError(f"Pationt type '{pationt_type}' not found in DataFrame. Available columns: {data.columns.tolist()}")
        
        print(f"Using Period: {period}, Seasonal: {seasonal}")
        
        stl = STL(data, period=period, seasonal=seasonal)
        result = stl.fit()
        return result

    def plot_decomposition(self, result):
        """
        Plots the STL decomposition result (Trend, Seasonality, Residuals).
        """
        fig = result.plot()
        plt.suptitle("STL Decomposition (Trend, Seasonality, Residuals)", fontsize=16)
        plt.tight_layout()
        plt.show()

    def analyze_residuals(self, residuals):
        """
        Fits various distributions to residuals and returns goodness-of-fit stats.
        """
        print("\n--- Analyzing Residuals Distribution ---")
        
        distributions = {
            "Normal": stats.norm,
            "Laplace": stats.laplace,
            "Student's t": stats.t,
            "Logistic": stats.logistic,
            "Cauchy": stats.cauchy,
            "Negative Binomial": stats.nbinom
        }
        
        results = {}
        print("\nGoodness of Fit (KS Test):")
        for name, dist in distributions.items():
            try:
                # Special handling for Negative Binomial (requires non-negative integers)
                if name == "Negative Binomial":
                    # Shift to make positive and round to nearest int
                    offset = abs(min(residuals)) if min(residuals) < 0 else 0
                    data_for_fit = np.round(residuals + offset).astype(int)
                    
                    # Manual Fit via Method of Moments (scipy nbinom.fit is unreliable/missing)
                    # Mean = n(1-p)/p, Var = n(1-p)/p^2
                    # p = Mean / Var
                    # n = Mean * p / (1-p)
                    
                    mean_val = np.mean(data_for_fit)
                    var_val = np.var(data_for_fit)
                    
                    if var_val > mean_val:
                        p_est = mean_val / var_val
                        n_est = (mean_val * p_est) / (1 - p_est)
                        params = (n_est, p_est) # n, p
                    else:
                        # Fallback if under-dispersed (Mean > Var), treat as Poisson-ish (p near 1)
                        # or just force a fit that works mathematically
                        p_est = 0.99 
                        n_est = mean_val
                        params = (n_est, p_est)
                        print("  (Warning: Data is under-dispersed, Negative Binomial may be poor fit)")

                    # For KS test / AIC, use the transformed data
                    curr_residuals = data_for_fit
                    print(f"  (Note: Negative Binomial fitted to shifted/rounded data, offset={offset:.2f})")
                else:
                    curr_residuals = residuals
                    params = dist.fit(curr_residuals)
                
                # KS Test
                # Pass the cdf method directly instead of the name string
                D, p_value = stats.kstest(curr_residuals, dist.cdf, args=params)
                
                # Calculate AIC
                # AIC = 2k - 2ln(L)
                # Use logpmf for discrete (Negative Binomial), logpdf for continuous
                if name == "Negative Binomial":
                     log_likelihood = np.sum(dist.logpmf(curr_residuals, *params))
                else:
                     log_likelihood = np.sum(dist.logpdf(curr_residuals, *params))
                     
                k = len(params)
                aic = 2 * k - 2 * log_likelihood
                
                # Store offset (default 0)
                offset_val = offset if name == "Negative Binomial" else 0
                
                results[name] = {'D': D, 'p_value': p_value, 'AIC': aic, 'params': params, 'dist': dist, 'offset': offset_val}
                print(f"{name}: D={D:.4f}, p-value={p_value:.4f}, AIC={aic:.4f}")
            except Exception as e:
                print(f"{name}: Failed to fit - {str(e)}")
            
        return results

    def plot_residuals_distribution(self, residuals, fit_results):
        """
        Plots the histogram of residuals and the fitted PDF curves.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, stat="density", linewidth=0, label='Residuals')
        plt.title("Distribution of Residuals")
        plt.xlabel("Residual Value")
        plt.ylabel("Density")
        
        x = np.linspace(residuals.min(), residuals.max(), 100)
        
        for name, res in fit_results.items():
            dist = res['dist']
            params = res['params']
            offset = res.get('offset', 0)
            
            if name == "Negative Binomial":
                # Use PMF for discrete, shifted by offset
                # We round x+offset to nearest integers for PMF evaluation
                # but plotting it as a smooth curve for visual comparison
                pdf = dist.pmf(x + offset, *params)
            else:
                pdf = dist.pdf(x, *params)
                
            plt.plot(x, pdf, label=f'{name} fit', linewidth=2)
            
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def get_best_distribution(self, fit_results):
        """
        Selects the best distribution based on AIC and KS test p-value.
        Priority:
        1. Valid fit (p-value > 0.05) with lowest AIC.
        2. If no valid fit, lowest AIC among all (with warning).
        
        Returns:
            tuple: (best_name, best_stats_dict)
        """
        valid_fits = {k: v for k, v in fit_results.items() if v['p_value'] > 0.05}
        
        if valid_fits:
            # Select lowest AIC among valid fits
            best_name = min(valid_fits, key=lambda k: valid_fits[k]['AIC'])
            print(f"\n--- Best Fit Selection ---")
            print(f"Selected '{best_name}' (Lowest AIC among valid fits).")
            return best_name, fit_results[best_name]
        else:
            # Fallback to lowest AIC among all
            best_name = min(fit_results, key=lambda k: fit_results[k]['AIC'])
            print(f"\n--- Best Fit Selection ---")
            print(f"WARNING: No distribution passed the KS test (p > 0.05).")
            print(f"Selected '{best_name}' based on lowest AIC only.")
            return best_name, fit_results[best_name]

    def run_full_analysis(self, data, pationt_type, period=7, seasonal=13):
        """
        Runs the complete analysis pipeline sequentially:
        1. Load Data (if file path provided)
        2. Decompose STL
        3. Plot Decomposition
        4. Analyze Residuals (fit distributions)
        5. Plot Residuals Distribution
        6. Select Best Distribution
        
        Args:
            data: DataFrame, Series, or string (file path to Excel/CSV).
            pationt_type: Column name to analyze.
            period: Periodicity (default 7).
            seasonal: Seasonal smoother (default 13).
            
        Returns:
            dict: The stats of the best fitting distribution.
        """
        # 0. Handle File Path Input
        if isinstance(data, str):
            print(f"File path detected: {data}")
            from .preprocessing import Preprocessing
            pre = Preprocessing()
            # Auto-load with defaults, drop_cancelled=True is safer for analysis
            data = pre.preprocess_data(data, drop_cancelled=True)
            print("Data loaded and preprocessed successfully.")

        # 1. Decompose
        result = self.decompose_stl(data, pationt_type, period, seasonal)
        
        # 2. Plot Decomposition
        self.plot_decomposition(result)
        
        # 3. Analyze Residuals
        fit_results = self.analyze_residuals(result.resid)
        
        # 4. Plot Residuals Distribution
        self.plot_residuals_distribution(result.resid, fit_results)
        
        # 5. Get Best Distribution
        best_name, best_stats = self.get_best_distribution(fit_results)
        
        print(f"\n--- Analysis Complete ---")
        print(f"Best fitting distribution: {best_name}")
        return best_stats
