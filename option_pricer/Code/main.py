from utils import *
import streamlit as st
import matplotlib.pyplot as plt

st.title("Black-Scholes Option Pricer")
st.subheader("By Nathan Chollet")

with st.sidebar:
    st.subheader("Input Parameters")
    #We get the 5 parameters for the the calculations of the call and put
    S = st.number_input("Current price of the asset:", value=100.0, min_value=0.0)
    K = st.number_input("Strike price of the asset:", value=80.0, min_value=0.0)
    T = st.number_input("Time to maturity of the asset (days):", value=30.0, min_value=0.0)
    sigma = st.number_input("Volatilty of the asset:", value=0.2, min_value=0.0)
    r = st.number_input("Risk-free interest rate of the asset:", value=0.05, min_value=0.0)

    #Outputs for the pricer and the heatmap
    calculate = st.button("Calculate", key="calculate-btn")

    #Inputs for the heatmap 
    st.divider()
    st.subheader("Heatmap Parameters")
    S_min = st.number_input("Min Spot Price:", value=10.0, min_value=0.0)
    S_max = st.number_input("Max Spot Price:", value=20.0, min_value=0.0)
    min_vol = st.slider("Minimum Volatility:", 0.01, 1.0, 0.2)
    max_vol = st.slider("Maximum Volatility:", 0.01, 1.0, 0.4)

    #Inputs for the stock options visualizer
    st.divider()
    st.subheader('Stock Option Visualizer Parameters')
    stock_symbol = st.selectbox("Select the stock symbol:", ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDIA', 'META'))
    start_date = st.date_input("Start Date:", value=pd.to_datetime('2022-01-01'))
    end_date = st.date_input("End Date:", value=pd.to_datetime('2022-12-12'))
    calculate2 = st.button('Visualize', key='stock-btn')

# Placeholders for Call and Put with default values
pricer_col_1, pricer_col_2 = st.columns(2)

# We compute the heatmaps for both call and put prices depending on spot and volatility 
# We keep constant (you can change them, you get the idea) the rest as spot price and volatility are the most sensitive parameters for pricing and option trading
call_matrix, put_matrix, S_range, vol_range = generate_option_heatmap_data(S_min, S_max, K, T, r, min_vol, max_vol, resolution=10)

with pricer_col_1:
    call_placeholder = st.empty()
    call_placeholder.metric("Call Price", "0.00")

#We plot the call option heatmap 
    st.divider()
    fig1 = plot_option_heatmap(call_matrix, S_range, vol_range, "Call Option Heatmap")
    st.pyplot(fig1)

with pricer_col_2:
    put_placeholder = st.empty()
    put_placeholder.metric("Put Price", "0.00")

#We plot the put option heatmap 
    st.divider()
    fig2 = plot_option_heatmap(put_matrix, S_range, vol_range, "Put Option Heatmap")
    st.pyplot(fig2)
    st.caption("The heatmaps above show the theoretical prices of call and put options based on the Black-Scholes model, varying spot prices and volatilities while keeping other parameters constant.")

if calculate:
    #We calculate and display the call and put prices for the given parameters
    call_price, put_price = black_scholes_option_pricer(S, K, T, sigma, r)
    call_placeholder.metric("Call Price", f"{call_price:.2f}")
    put_placeholder.metric("Put Price", f"{put_price:.2f}")

    

#We calculate and display the greeks 
st.subheader("Option Greeks")
greeks_data = greeks_calculator(S, K, T, sigma, r, in_days = True)
st.table(greeks_data)

#We display the value of a chosen stock symbol 
st.subheader("Option visualizer")

if calculate2:
    
    stock_data, option_data = stock_option_calculator(stock_symbol, start_date, end_date, T, sigma, r)

    # Plot stock price
    fig3, ax = plt.subplots(figsize=(10,5))
    ax.plot(stock_data['Date'], stock_data['Close'], label='Spot Price', color='black')

    # Plot option prices
    ax.plot(option_data['Date'], option_data['Call Price'], label='Call Option', color='green')
    ax.plot(option_data['Date'], option_data['Put Price'], label='Put Option', color='red')

    # Labels and formatting
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig3)
    st.caption("Theoretical option value evolution over time (based on Black–Scholes)")
