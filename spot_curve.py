# spot_curve.py
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

# Calculate the number of days since the last coupon payment.
def calculate_days_since_last_coupon_payment(date):
    if date.month < 7:
        start_date = datetime(date.year, 1, 1)
    else:
        start_date = datetime(date.year, 6, 30)
    return (date - start_date).days

# Calculate the dirty price by adding accrued interest to the clean price.
def calculate_dirty_prices(price, days_since_last_coupon_payment, coupon_rate):
    if price == "-":
        return "-"
    else:
        return price + (days_since_last_coupon_payment / 365 * coupon_rate * 100)

# Linear interpolation function to estimate a rate at a missing maturity.
def interpolate_rate(rate_dict, target_maturity):
    sorted_maturities = sorted(rate_dict.keys())
    lower_maturity, upper_maturity = None, None
    lower_rate, upper_rate = None, None

    # Find the two adjacent maturities around target_maturity.
    for m in sorted_maturities:
        if m <= target_maturity:
            lower_maturity = m
            lower_rate = rate_dict[m]
        elif m > target_maturity:
            upper_maturity = m
            upper_rate = rate_dict[m]
            break

    # If no upper bound is found, return the lower rate.
    if upper_maturity is None:
        return lower_rate
    # Linear interpolation formula.
    return lower_rate + (target_maturity - lower_maturity) * (upper_rate - lower_rate) / (upper_maturity - lower_maturity)

# Read bonds data from Excel and perform initial processing.
def create_sorted_bonds_by_maturity(file_name):
    dfs = pd.read_excel(file_name, sheet_name=None, usecols=['Coupon', 'Maturity Date', 'Price', 'Years until Maturity'])
    
    for sheet_name, df in dfs.items():
        df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
        df['Days since last coupon payment'] = df['Maturity Date'].apply(calculate_days_since_last_coupon_payment)
        df['Dirty price'] = df.apply(lambda row: calculate_dirty_prices(row['Price'], row['Days since last coupon payment'], row['Coupon']), axis=1)
        df['Years until Maturity'] = df['Years until Maturity'].round(4)
    return dfs

# Bootstrap spot rates from bonds data.
def calcualte_spot_rates(existing_spot_rates, dirty_price, coupon_rate, years_until_maturity):
    # For very short maturities, compute the rate directly.
    if years_until_maturity < 0.5:
        notional = 100 + coupon_rate / 2 * 100
        return - np.log(dirty_price / notional) / years_until_maturity

    number_of_payments = int(np.floor(years_until_maturity * 2))
    residual_price = dirty_price

    # Subtract the present value of coupon payments already priced by the existing spot rates.
    for i in range(number_of_payments):
        t = round(years_until_maturity - (number_of_payments - i) * 0.5, 4)
        coupon = 100 * coupon_rate / 2
        # Retrieve the spot rate for t, or interpolate if not available.
        if t in existing_spot_rates:
            rate = existing_spot_rates[t]
        else:
            rate = interpolate_rate(existing_spot_rates, t)
        residual_price -= coupon * np.exp(- rate * t)
    
    # Calculate the spot rate at maturity.
    final_cashflow = 100 + 100 * coupon_rate / 2
    return - np.log(residual_price / final_cashflow) / years_until_maturity

# Compute the spot curve for each day.
def calculate_all_spot_rates(bonds):
    all_spot_rates = {}
    for date, bond in bonds.items():
        bond = bond.sort_values(by='Years until Maturity', ascending=True)
        spot_rates = {}
        for index, row in bond.iterrows():
            T = row['Years until Maturity']
            if row['Dirty price'] == "-":
                continue
            # Bootstrap the spot rate using already computed rates.
            spot_rates[T] = calcualte_spot_rates(spot_rates, row['Dirty price'], row['Coupon'], T)
        all_spot_rates[date] = spot_rates
    return all_spot_rates

# Plot the 5-year spot curve for each day.
def plot_spot_curve(spot_rates):
    cmap = plt.get_cmap('tab10')
    for i, date in enumerate(spot_rates):
        maturities = list(spot_rates[date].keys())
        rates = list(spot_rates[date].values())
        label = date  # Use the sheet name or date as the label.
        plt.plot(maturities, rates, label=label, marker='o', linestyle='-', color=cmap(i / len(spot_rates)))
    
    plt.xlabel('Years until Maturity', fontsize=14)
    plt.ylabel('Spot Rate', fontsize=14)
    plt.title('5-Year Spot Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)

# Main execution block for spot curve analysis.
if __name__ == "__main__":
    # Replace with the name of your Excel file.
    sorted_bonds = create_sorted_bonds_by_maturity('selected_10_bonds.xlsx')
    spot_rates = calculate_all_spot_rates(sorted_bonds)
    
    plt.figure()
    plot_spot_curve(spot_rates)
    plt.show()

    # Store the computed spot_rates dictionary into a file using pickle.
    with open('spot_rates.pkl', 'wb') as f:
        pickle.dump(spot_rates, f)
    print("Spot rates have been stored in 'spot_rates.pkl'.")
