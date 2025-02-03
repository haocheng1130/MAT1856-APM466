# yield_curve.py
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pickle

# Calculate the number of days since the last coupon payment.
def calculate_days_since_last_coupon_payment(date):
    # Using a simple convention: if the month is before July, assume the coupon was paid on January 1;
    # otherwise, assume it was paid on June 30.
    if date.month < 7:
        start_date = datetime(date.year, 1, 1)
    else:
        start_date = datetime(date.year, 6, 30)
    return (date - start_date).days

# Calculate the dirty price by adding accrued interest to the clean price.
def calculate_dirty_prices(price, days_since_last_coupon_payment, coupon_rate):
    # If price is missing (represented by "-"), return "-" to indicate an invalid value.
    if price == "-":
        return "-"
    else:
        # Assuming a face value of $100.
        accrued_interest = days_since_last_coupon_payment / 365 * coupon_rate * 100
        return price + accrued_interest

# Read the bonds data from an Excel file and preprocess it.
def create_sorted_bonds_by_maturity(file_name):
    # Read all sheets from the Excel file. Each sheet represents a day.
    dfs = pd.read_excel(file_name, sheet_name=None, usecols=['Coupon', 'Maturity Date', 'Price', 'Years until Maturity'])
    
    for sheet_name, df in dfs.items():
        # Convert the 'Maturity Date' to a datetime object.
        df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
        # Compute days since last coupon using the defined function.
        df['Days since last coupon payment'] = df['Maturity Date'].apply(calculate_days_since_last_coupon_payment)
        # Calculate the dirty price for each bond.
        df['Dirty price'] = df.apply(lambda row: calculate_dirty_prices(row['Price'], row['Days since last coupon payment'], row['Coupon']), axis=1)
        # Round the 'Years until Maturity' to four decimals.
        df['Years until Maturity'] = df['Years until Maturity'].round(4)
    return dfs

# Compute the Yield-to-Maturity (YTM) for a given bond.
def calculate_ytm(dirty_price, coupon_rate, years_until_maturity):
    coupon_payment = 100 * coupon_rate / 2  # Assume semiannual coupon payments.
    # For bonds with maturity less than 0.5 years (only one coupon payment):
    if years_until_maturity < 0.5:
        notional = 100 + coupon_payment
        return - np.log(dirty_price / notional) / years_until_maturity

    number_of_coupon_payments = int(np.floor(years_until_maturity * 2))
    
    # Define the pricing equation based on continuous compounding.
    def pricing_equation(y):
        present_value = 0
        # Sum the present value of all coupon payments except the final one.
        for i in range(number_of_coupon_payments - 1):
            t = years_until_maturity - (number_of_coupon_payments - i - 1) * 0.5
            present_value += coupon_payment * np.exp(-y * t)
        # Add the final cash flow (face value + coupon payment).
        present_value += (100 + coupon_payment) * np.exp(-y * years_until_maturity)
        return present_value - dirty_price

    # Solve the equation numerically (using fsolve) to find y.
    ytm = fsolve(pricing_equation, 0.05)  # Initial guess: 5%
    return ytm[0]

# Compute the YTM curve for each day.
def calculate_ytm_curve(bonds):
    all_ytm_vals = {}
    for date, bond in bonds.items():
        # Sort the bonds by increasing 'Years until Maturity'
        bond = bond.sort_values(by='Years until Maturity', ascending=True)
        ytm_vals = {}
        for index, row in bond.iterrows():
            T = row['Years until Maturity']
            if row['Dirty price'] == "-":
                continue
            ytm_vals[T] = calculate_ytm(row['Dirty price'], row['Coupon'], T)
        all_ytm_vals[date] = ytm_vals
    return all_ytm_vals

# Plot the yield curve for each day.
def plot_ytm_curve(ytm_rates):
    cmap = plt.get_cmap('tab10')
    for i, date in enumerate(ytm_rates):
        maturities = list(ytm_rates[date].keys())
        yields = list(ytm_rates[date].values())
        # Extract a label from the sheet name (assuming it contains a date).
        label = date
        plt.plot(maturities, yields, label=label, marker='o', linestyle='-', color=cmap(i / len(ytm_rates)))
    
    plt.xlabel('Years until Maturity', fontsize=14)
    plt.ylabel('Yield-to-Maturity', fontsize=14)
    plt.title('5-Year Yield Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)

# Main execution block for yield curve analysis.
if __name__ == "__main__":
    # Replace 'bond_prices_chosen_bonds.xlsx' with your Excel file name.
    sorted_bonds = create_sorted_bonds_by_maturity('selected_10_bonds.xlsx')
    yield_rates = calculate_ytm_curve(sorted_bonds)
    
    plt.figure()
    plot_ytm_curve(yield_rates)
    plt.show()

    # Store the computed yield_rates dictionary into a file using pickle.
    with open('yield_rates.pkl', 'wb') as f:
        pickle.dump(yield_rates, f)
    print("Spot rates have been stored in 'yield_rates.pkl'.")
