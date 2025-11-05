import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from yfinance import download
import pandas as pd
import yfinance as yf


#Call and Put option pricing using Black-Scholes formula
def black_scholes_option_pricer(S, K, T=1.0, sigma=0.2, r=0.05, in_days = True):
    # Convert days to years
    if in_days:
        T = T / 365

    #Catch all possible errors that could bring mathematic problems or unrealistic problems    
    if S <= 0 or K <= 0:  
        raise ValueError("Asset and strike prices must be positive.")
    if T <= 0:
        raise ValueError("Time to maturity must be greater than 0.")
    if sigma <= 0:
        raise ValueError("Volatility must be greater than 0.")
    
    #Calculation of the d1 and d2 components
    d1 = (np.log(S/K) + (r + (1/2) * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    #Calculation of the call and put price
    call = S * norm.cdf(d1) - K * np.exp(-r * T) *norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return call, put

#Computes the greeks of an option for both call and put prices
#Useful to get insight on option behaviour when we change variables like volatility, etc.
def greeks_calculator(S, K, T=1.0, sigma=0.2, r=0.05, in_days = True):
    # Convert days to years
    if in_days:
        T = T / 365

    #Catch all possible errors that could bring mathematic problems or unrealistic problems    
    if S <= 0 or K <= 0:  
        raise ValueError("Asset and strike prices must be positive.")
    if T <= 0:
        raise ValueError("Time to maturity must be greater than 0.")
    if sigma <= 0:
        raise ValueError("Volatility must be greater than 0.")
    
    #Calculation of the d1 and d2 components
    d1 = (np.log(S/K) + (r + (1/2) * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    #Calculation of deltas -> risk linked to the asset price
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1

    #Calculation of gamma -> risk linked to the asset price
    gamma = norm.pdf(d1)/(S * sigma * np.sqrt(T))

    #Claculation of rhos -> risk linked to the interest rate
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2)
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    #Calculation of thetas -> risk linked to the time to maturity
    theta_call = -((S * norm.pdf(d1) * sigma)/(2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_put = -((S * norm.pdf(d1) * sigma)/(2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)

    #Claculation of vega -> risk linked to the volatility
    vega = S * norm.pdf(d1) * np.sqrt(T)

    #We store everything into a nice dataframe for display
    data = {
        'Index' : ['Delta', 'Gamma', 'Rho', 'Theta', 'Vega'],
        'Call values': [delta_call, gamma, rho_call, theta_call, vega],
        'Put values': [delta_put, gamma, rho_put, theta_put, vega]
    }

    greeks_data = pd.DataFrame(data)
    greeks_data.set_index('Index', inplace=True)

    return greeks_data

 #Computes two matrices (one for call and one for put) and gives alongside index for a heatmap
 #The spot price of the stock rattached to the option is the x-axis, and the volatility the y-axis
 #The resolution controls how much tiles will appear in our heatmap
def generate_option_heatmap_data(S_min, S_max, K, T, r, min_vol, max_vol, resolution=10):
    #we generate the ranges of each values depending on the user inputs
    S_range = np.linspace(S_min, S_max, resolution)
    vol_range = np.linspace(min_vol, max_vol, resolution)

    #We create two 2d-matrices of zeros, of size our heatmap
    call_matrix = np.zeros((len(S_range), len(vol_range)))
    put_matrix = np.zeros((len(S_range), len(vol_range)))

    #we calculate each call and put price depending on the x and y values (i.e. the spot and volatiliy)
    #we then replace the zeroes by the actual values and return everything needed to compute that graph
    for i in range(len(S_range)):
        for j in range(len(vol_range)):
            call_matrix[i][j], put_matrix[i][j] = black_scholes_option_pricer(S_range[i], K, T, vol_range[j], r, in_days = True)

    return call_matrix, put_matrix, S_range, vol_range


# Function to plot call and put heatmaps using pcolormesh
# This part was arguably the hardest for me so please be kind I sweated blood and tears on matplotlib
def plot_option_heatmap(matrix, x_vals, y_vals, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_vals, y_vals)
    # pcolormesh expects shape of Z = (len(y), len(x))
    pcm = ax.pcolormesh(X, Y, matrix, cmap='RdYlGn', shading='auto')
    
    # Add text values centered in each cell, if not it's unreadable
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            ax.text(x, y, f"{matrix[j][i]:.1f}", ha='center', va='center', fontsize=8, color='white')
    
    #We must set ticks to the exact spot and volatility values to get useful info from the heatmap
    ax.set_xticks(x_vals)
    ax.set_yticks(y_vals)
    ax.set_xticklabels([f"{s:.2f}" for s in x_vals])
    ax.set_yticklabels([f"{v:.2f}" for v in y_vals])

    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax)
    return fig


def stock_option_calculator(symbol, start_date, end_date, T, sigma, r ):
    
    data = yf.download(symbol, start= start_date, end= end_date, auto_adjust=True, multi_level_index=False)[['Close']]
    data = data.reset_index()

    option_data = []


    #We average the stock value in order to use it as our strike price
    K = float(np.mean(data['Close']))

    for i in range(len(data['Close'])):
        S = data['Close'].iloc[i]
        Date = data['Date'].iloc[i]
        call_price, put_price = black_scholes_option_pricer(S, K, T, sigma, r, in_days = True)
        option_data.append({'Date': Date, 'Spot Price': S, 'Call Price': call_price, 'Put Price': put_price})

    return data, pd.DataFrame(option_data)


    