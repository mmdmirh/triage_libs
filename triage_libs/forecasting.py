import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats

class ARIMAModeler:
    def __init__(self, data):
        """
        Initialize the ARIMAModeler.
        
        Args:
            data (pd.Series or pd.DataFrame): Time series data. 
                                              If DataFrame, specific column checks should be done prior.
        """
        self.data = data
        self.model_result = None
        self.best_order = None
        self.best_seasonal_order = None
        
        # Suppress convergence warnings during grid search
        warnings.filterwarnings("ignore")

    def find_best_model(self, max_p=3, max_d=2, max_q=3, seasonal=False, m=12):
        """
        Performs a grid search to find the best ARIMA/SARIMA model based on AIC.
        
        Args:
            max_p (int): Max AR terms.
            max_d (int): Max differencing terms.
            max_q (int): Max MA terms.
            seasonal (bool): Whether to include seasonal components.
            m (int): Seasonality period (e.g. 12 for monthly, 7 for daily).
            
        Returns:
            tuple: (best_order, best_seasonal_order, best_aic)
        """
        print(f"\n--- Starting Grid Search (Seasonal={seasonal}, m={m}) ---")
        
        # Define parameter ranges
        p = range(0, max_p + 1)
        d = range(0, max_d + 1)
        q = range(0, max_q + 1)
        
        pdq = list(product(p, d, q))
        
        seasonal_pdq = [(0, 0, 0, 0)]
        if seasonal:
            # Simplified seasonal grid to avoid explosion
            # P, D, Q usually small (0 or 1)
            seasonal_pdq = list(product(range(0, 2), range(0, 2), range(0, 2), [m]))
            
        best_aic = float("inf")
        best_param = None
        best_seasonal_param = None
        
        total_combinations = len(pdq) * len(seasonal_pdq)
        print(f"Testing {total_combinations} combinations...")
        
        counter = 0
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                counter += 1
                try:
                    model = SARIMAX(self.data,
                                    order=param,
                                    seasonal_order=param_seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    
                    results = model.fit(disp=False)
                    
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_param = param
                        best_seasonal_param = param_seasonal
                        print(f"New Best AIC: {best_aic:.2f} | Order: {param} | Seasonal: {param_seasonal}")
                        
                except Exception as e:
                    continue
                    
        print("\n--- Grid Search Complete ---")
        print(f"Best AIC: {best_aic:.2f}")
        print(f"Best Order: {best_param}")
        print(f"Best Seasonal Order: {best_seasonal_param}")
        
        self.best_order = best_param
        self.best_seasonal_order = best_seasonal_param
        
        return best_param, best_seasonal_param, best_aic

    def fit_model(self, order, seasonal_order=(0, 0, 0, 0)):
        """
        Fits a SARIMAX model with the specified parameters.
        """
        print(f"\nFitting model with order={order}, seasonal_order={seasonal_order}...")
        
        try:
            model = SARIMAX(self.data,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            
            self.model_result = model.fit(disp=False)
            print(self.model_result.summary())
            return self.model_result.summary()
        except Exception as e:
            print(f"Error fitting model: {e}")
            return None

    def plot_diagnostics(self):
        """
        Plots diagnostic charts for the fitted model residuals.
        """
        if self.model_result is None:
            print("No model fitted yet. Call fit_model() first.")
            return

        print("\n--- Plotting Diagnostics ---")
        # Standard residuals
        residuals = self.model_result.resid
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Standardized Residuals
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Standardized Residuals')
        axes[0, 0].axhline(0, color='black', linestyle='--')
        
        # 2. Histogram + KDE vs Normal
        sns.histplot(residuals, kde=True, ax=axes[0, 1], stat="density", label='Residuals')
        # Plot standard normal for comparison
        xmin, xmax = axes[0, 1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, residuals.mean(), residuals.std())
        axes[0, 1].plot(x, p, 'r', linewidth=2, label='Normal Distribution')
        axes[0, 1].set_title('Histogram plus Estimated Density')
        axes[0, 1].legend()
        
        # 3. Normal Q-Q
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Normal Q-Q')
        
        # 4. Correlogram (ACF)
        plot_acf(residuals, ax=axes[1, 1], lags=None, title='Correlogram') # lags=None uses default logic
        
        plt.tight_layout()
        plt.show()
        
        # Ljung-Box Test for Whiteness
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        p_val = lb_test['lb_pvalue'].iloc[0]
        print(f"\nLjung-Box Test (lag=10): p-value = {p_val:.4f}")
        if p_val > 0.05:
            print("SUCCESS: Residuals allow us to accept the null hypothesis (White Noise). Model is good.")
        else:
            print("WARNING: Residuals are not independent (Autocorrelation exists). Try different parameters.")

    def forecast(self, steps=12):
        """
        Generates forecasts for fiture steps.
        
        Args:
            steps (int): Number of steps to forecast.
            
        Returns:
            pd.DataFrame: Forecast frame with 'mean', 'mean_ci_lower', 'mean_ci_upper'.
        """
        if self.model_result is None:
            print("No model fitted yet.")
            return None

        print(f"\nGenerating forecast for {steps} steps...")
        
        # Get forecast
        pred_uc = self.model_result.get_forecast(steps=steps)
        pred_ci = pred_uc.conf_int()
        
        forecast_df = pd.DataFrame({
            'forecast': pred_uc.predicted_mean,
            'lower_ci': pred_ci.iloc[:, 0],
            'upper_ci': pred_ci.iloc[:, 1]
        })
        
        # Plotting Forecast
        plt.figure(figsize=(12, 6))
        
        # Plot observed data
        plt.plot(self.data.index, self.data, label='Observed')
        
        # Plot Forecast
        plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='red')
        
        # Plot Confidence Interval
        plt.fill_between(forecast_df.index,
                         forecast_df['lower_ci'],
                         forecast_df['upper_ci'], color='pink', alpha=0.3)
        
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('ARIMA Forecast')
        plt.legend()
        plt.show()
        
        return forecast_df
