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
