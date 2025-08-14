import numpy as np
import scipy.stats as stats
from scipy.stats import norm
# from scipy.fft import fft
from scipy.integrate import quad
# from scipy.optimize import newton

class OptionPricing:
    def __init__(self, S0=100, r=0.05, sigma=0.2, T=1, K=100, M=50):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.M = M
        
        # Initialize all models
        self.crr_model = CRRModel()
        self.bs_model = BlackScholesModel()
        self.mc_model = MonteCarloModel()
        self.ni_model = NumericalIntegrationModel()

class CRRModel:
    def CRR_Option(self, S0, r, sigma, T, M, K, option_type='call', option_style='European'):
        delta_t = T / M
        beta = (np.exp(-r*delta_t)+np.exp((r+(sigma)**2)*delta_t))/2
        u = beta + np.sqrt(beta**2 - 1)
        d = 1 / u
        q = (np.exp(r * delta_t) - d) / (u - d)
        
        S = np.zeros((M + 1, M + 1))
        S[0, 0] = S0
        
        for i in range(1, M + 1):
            for j in range(1, i + 1):
                S[j, i] = S0 * (u**j) * (d**(i-j))
        
        V = np.zeros((M + 1, M + 1))
        for j in range(M + 1):
            if option_type == 'call':
                V[j, M] = max(0, S[j, M] - K)
            else:
                V[j, M] = max(0, K - S[j, M])
        
        # Backward induction
        for i in range(M - 1, -1, -1):
            for j in range(i + 1):
                if option_style == 'European':
                    V[j, i] = np.exp(-r * delta_t) * (q * V[j+1, i+1] + (1-q) * V[j, i+1])
                else:  # American
                    if option_type == 'call':
                        exercise = max(0, S[j, i] - K)
                    else:
                        exercise = max(0, K - S[j, i])
                    hold = np.exp(-r * delta_t) * (q * V[j+1, i+1] + (1-q) * V[j, i+1])
                    V[j, i] = max(exercise, hold)
        
        return V[0, 0]

class BlackScholesModel:
    def BlackScholes_Option(self, t, St, K, T, r, sigma, option='call'):
        
        d1 = ((np.log(St / K) + (r + 0.5 * sigma**2) * (T - t))) / (sigma * np.sqrt(T - t))
        d2 = d1 - sigma * np.sqrt(T - t)
        
        if option == 'call':
            price = St * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * (T - t)) * norm.cdf(-d2) - St * norm.cdf(-d1)
        
        return price

class MonteCarloModel:
    def MonteCarlo_Option(self, t, St, K, T, r, sigma, M, option ='call'):
        np.random.seed(42)  # For reproducibility
        Z = np.random.normal(0, 1, M)
        ST = St * np.exp((r - 0.5 * sigma ** 2) * (T-t) + sigma * np.sqrt(T-t) * Z)
        
        if option == 'call':
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)
        
        option_price = np.exp(-r * T) * np.mean(payoff)
        stderr = np.exp(-r * T) * np.std(payoff) / np.sqrt(M)
        conf_interval = [option_price - 1.96 * stderr, option_price + 1.96 * stderr]

        
        return option_price, stderr, conf_interval

class NumericalIntegrationModel:
    def BS_Price_Int(self, t, St, K, T, r, sigma, option='call'):
        # Define the payoff function based on option type
        if option == 'call':
            payoff = lambda S: max(S - K, 0)
        elif option == 'put':
            payoff = lambda S: max(K - S, 0)
        else:
            raise ValueError("Invalid option_type. Use 'call' or 'put'.")

        # Define the integrand for risk-neutral expectation
        def integrand(x):
            ST = St * np.exp((r - 0.5 * sigma**2) * (T - t) + sigma * np.sqrt(T - t) * x)
            return payoff(ST) * np.exp(-0.5 * x**2)

        price, _ = quad(integrand, -np.inf, np.inf)
        normalization = 1 / np.sqrt(2 * np.pi)
        return np.exp(-r * (T - t)) * normalization * price



# Example usage
if __name__ == "__main__":
    # Initialize the OptionPricing class with parameters
    option_pricing = OptionPricing(S0=100, r=0.05, sigma=0.2, T=1, K=100, M=50)

    # Test parameters
    S0 = 100  # Initial stock price
    r = 0.05   # Risk-free interest rate
    sigma = 0.2  # Volatility of the underlying asset
    T = 1      # Time to expiration (in years)
    K = 105    # Strike price
    M = 100    # Number of time steps for CRR model

    # Test commands for different methods and scenarios
    print("European Call Option Price (CRR model):", 
          option_pricing.crr_model.CRR_Option(S0, r, sigma, T, M, K, 'call', 'European'))
    
    print("American Put Option Price (CRR model):", 
          option_pricing.crr_model.CRR_Option(S0, r, sigma, T, M, K, 'put', 'American'))
    
    print("European Put Option Price (Black-Scholes model):", 
           option_pricing.bs_model.BlackScholes_Option(t=0, St=S0, K=K, T=T, r=r, sigma=sigma, option='call'))
    
    print("European Put Option Price (Black-Scholes model):", 
          option_pricing.bs_model.BlackScholes_Option(t=0, St=S0, K=K, T=T, r=r, sigma=sigma, option='put'))
    
    print("European Call Option Price (Monte Carlo simulation):", 
         option_pricing.mc_model.MonteCarlo_Option(t=0, St=S0, K=K, T=T, r=r, sigma=sigma, M=10000, option='call')[0])

    
    print("European Call Option Price (Numerical Integration):", 
          option_pricing.ni_model.BS_Price_Int(t=0, St=S0, K=K, T=T, r=r, sigma=sigma, option='call'))
