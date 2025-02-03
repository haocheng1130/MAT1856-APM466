# forward_curve.py
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

# Reuse the function to calculate days since last coupon payment.
def calculate_days_since_last_coupon_payment(date):
    if date.month < 7:
        start_date = datetime(date.year, 1, 1)
    else:
        start_date = datetime(date.year, 6, 30)
    return (date - start_date).days

# Reuse the function to calculate dirty prices.
def calculate_dirty_prices(price, days_since_last_coupon_payment, coupon_rate):
    if price == "-":
        return "-"
    else:
        return price + (days_since_last_coupon_payment / 365 * coupon_rate * 100)

# Linear interpolation for missing rates.
def interpolate_rate(rate_dict, target_maturity):
    sorted_maturities = sorted(rate_dict.keys())
    lower_maturity, upper_maturity = None, None
    lower_rate, upper_rate = None, None
    for m in sorted_maturities:
        if m <= target_maturity:
            lower_maturity = m
            lower_rate = rate_dict[m]
        elif m > target_maturity:
            upper_maturity = m
            upper_rate = rate_dict[m]
            break
    if upper_maturity is None:
        return lower_rate
    return lower_rate + (target_maturity - lower_maturity) * (upper_rate - lower_rate) / (upper_maturity - lower_maturity)

# Read bonds data from Excel and perform preprocessing.
def create_sorted_bonds_by_maturity(file_name):
    dfs = pd.read_excel(file_name, sheet_name=None, usecols=['Coupon', 'Maturity Date', 'Price', 'Years until Maturity'])
    for sheet_name, df in dfs.items():
        df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
        df['Days since last coupon payment'] = df['Maturity Date'].apply(calculate_days_since_last_coupon_payment)
        df['Dirty price'] = df.apply(lambda row: calculate_dirty_prices(row['Price'], row['Days since last coupon payment'], row['Coupon']), axis=1)
        df['Years until Maturity'] = df['Years until Maturity'].round(4)
    return dfs

# Compute spot rates (required for forward rate calculation) using bootstrapping.
def calcualte_spot_rates(existing_spot_rates, dirty_price, coupon_rate, years_until_maturity):
    if years_until_maturity < 0.5:
        notional = 100 + coupon_rate / 2 * 100
        return - np.log(dirty_price / notional) / years_until_maturity

    number_of_payments = int(np.floor(years_until_maturity * 2))
    residual_price = dirty_price
    for i in range(number_of_payments):
        t = round(years_until_maturity - (number_of_payments - i) * 0.5, 4)
        coupon = 100 * coupon_rate / 2
        if t in existing_spot_rates:
            rate = existing_spot_rates[t]
        else:
            rate = interpolate_rate(existing_spot_rates, t)
        residual_price -= coupon * np.exp(- rate * t)
    final_cashflow = 100 + 100 * coupon_rate / 2
    return - np.log(residual_price / final_cashflow) / years_until_maturity

# Compute the spot curve for each day (used in forward rate calculation).
def calculate_all_spot_rates(bonds):
    all_spot_rates = {}
    for date, bond in bonds.items():
        bond = bond.sort_values(by='Years until Maturity', ascending=True)
        spot_rates = {}
        for index, row in bond.iterrows():
            T = row['Years until Maturity']
            if row['Dirty price'] == "-":
                continue
            spot_rates[T] = calcualte_spot_rates(spot_rates, row['Dirty price'], row['Coupon'], T)
        all_spot_rates[date] = spot_rates
    return all_spot_rates

# Calculate the future price (at time t=1) of a zero-coupon bond maturing at T.
def calculate_future_price_zero_bonds(future_time, years_until_maturity, spot_rates):
    # Retrieve the spot rate for the given maturity, or interpolate if needed.
    if years_until_maturity in spot_rates:
        r = spot_rates[years_until_maturity]
    else:
        r = interpolate_rate(spot_rates, years_until_maturity)
    # Compute the future price (price at time t=future_time).
    return np.exp(- r * (years_until_maturity - future_time))

# Derive the forward rate curve from the spot rates.
def calculate_forward_rate(all_spot_rates, all_bonds):
    all_forward_rates = {}
    for date in all_spot_rates:
        spot_rates = all_spot_rates[date]
        bonds = all_bonds[date]
        future_prices = []
        maturities = []
        
        # Sort bonds by increasing maturity.
        bonds = bonds.sort_values(by='Years until Maturity', ascending=True)
        # For forward rate calculation, only consider bonds with maturity >= 1 year.
        for index, row in bonds.iterrows():
            T = row['Years until Maturity']
            if row['Dirty price'] == "-" or T < 1 or T in maturities:
                continue
            maturities.append(T)
            # Compute the logarithm of the future price of a zero-coupon bond at t=1.
            price_at_1 = calculate_future_price_zero_bonds(1, T, spot_rates)
            future_prices.append(np.log(price_at_1))
        
        forward_rates = {}
        num_points = len(future_prices)
        # Use finite differences to approximate the derivative of ln(price) with respect to T.
        for idx in range(num_points):
            if idx == 0:
                # Forward difference for the lower boundary.
                derivative = (future_prices[idx + 1] - future_prices[idx]) / (maturities[idx + 1] - maturities[idx])
            elif idx == num_points - 1:
                # Backward difference for the upper boundary.
                derivative = (future_prices[idx] - future_prices[idx - 1]) / (maturities[idx] - maturities[idx - 1])
            else:
                # Central difference for interior points.
                derivative = 0.5 * ((future_prices[idx + 1] - future_prices[idx]) / (maturities[idx + 1] - maturities[idx]) +
                                    (future_prices[idx] - future_prices[idx - 1]) / (maturities[idx] - maturities[idx - 1]))
            # The forward rate is the negative derivative.
            forward_rates[maturities[idx]] = - derivative
        all_forward_rates[date] = forward_rates
    return all_forward_rates

# Plot the 1-year forward curve for each day.
def plot_forward_rates_curve(forward_rates):
    cmap = plt.get_cmap('tab10')
    for i, date in enumerate(forward_rates):
        maturities = list(forward_rates[date].keys())
        fwd_rates = list(forward_rates[date].values())
        label = date
        plt.plot(maturities, fwd_rates, label=label, marker='o', linestyle='-', color=cmap(i / len(forward_rates)))
    
    plt.xlabel('Years until Maturity', fontsize=14)
    plt.ylabel('1-Year Forward Rate', fontsize=14)
    plt.title('1-Year Forward Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)

# Main execution block for forward curve analysis.
if __name__ == "__main__":
    # Replace with your Excel file name.
    sorted_bonds = create_sorted_bonds_by_maturity('selected_10_bonds.xlsx')
    # Compute the spot rates first (required for forward rate calculation).
    spot_rates = calculate_all_spot_rates(sorted_bonds)
    # Compute the forward rates using the spot rates and bond data.
    forward_rates = calculate_forward_rate(spot_rates, sorted_bonds)
    
    plt.figure()
    plot_forward_rates_curve(forward_rates)
    plt.show()

    # Store the computed forward_rates dictionary into a file using pickle.
    with open('forward_rates.pkl', 'wb') as f:
        pickle.dump(forward_rates, f)
    print("Forward rates have been stored in 'forward_rates.pkl'.")
