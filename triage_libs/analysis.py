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
            "Cauchy": stats.cauchy
        }
        
        results = {}
        print("\nGoodness of Fit (KS Test):")
        for name, dist in distributions.items():
            # Fit parameters
            params = dist.fit(residuals)
            
            # KS Test
            # Pass the cdf method directly instead of the name string
            D, p_value = stats.kstest(residuals, dist.cdf, args=params)
            
            # Calculate AIC
            # AIC = 2k - 2ln(L)
            log_likelihood = np.sum(dist.logpdf(residuals, *params))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            
            results[name] = {'D': D, 'p_value': p_value, 'AIC': aic, 'params': params, 'dist': dist}
            print(f"{name}: D={D:.4f}, p-value={p_value:.4f}, AIC={aic:.4f}")
            
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
        1. Decompose STL
        2. Plot Decomposition
        3. Analyze Residuals (fit distributions)
        4. Plot Residuals Distribution
        5. Select Best Distribution
        
        Args:
            data: DataFrame or Series.
            pationt_type: Column name to analyze.
            period: Periodicity (default 7).
            seasonal: Seasonal smoother (default 13).
            
        Returns:
            dict: The stats of the best fitting distribution.
        """
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
