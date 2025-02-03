# covariance_eigen.py
import numpy as np
import pickle

def interpolate_rate(rate_dict, target_maturity):
    """
    Interpolate linearly to estimate the rate for a target maturity.
    
    Parameters:
        rate_dict (dict): Dictionary with keys as maturities and values as rates.
        target_maturity (float): The maturity for which to interpolate the rate.
    
    Returns:
        float: The interpolated rate.
    """
    sorted_maturities = sorted(rate_dict.keys())
    lower_maturity = None
    upper_maturity = None
    lower_rate = None
    upper_rate = None

    # Find the two adjacent maturities around target_maturity.
    for m in sorted_maturities:
        if m <= target_maturity:
            lower_maturity = m
            lower_rate = rate_dict[m]
        elif m > target_maturity:
            upper_maturity = m
            upper_rate = rate_dict[m]
            break
    
    # If no upper maturity is found, return the lower rate.
    if upper_maturity is None:
        return lower_rate
    
    # Linear interpolation formula.
    return lower_rate + (target_maturity - lower_maturity) * (upper_rate - lower_rate) / (upper_maturity - lower_maturity)


def calculate_covariance_matricies_for_yields(all_rates, num_variables):
    """
    Calculate the covariance matrix of the daily log-returns of yield (YTM) rates and
    compute its eigenvalues and eigenvectors.
    
    Parameters:
        all_rates (dict): A dictionary where each key is a date and each value is another
                          dictionary mapping maturities (e.g., 1,2,3,4,5 years) to yield rates.
        num_variables (int): Number of yield rate variables (e.g., 5 for 1-yr, 2-yr, ..., 5-yr).
    
    Returns:
        tuple: (covariance_matrix, eigenvalues, eigenvectors)
    """
    # Get the dates; assume rates are available for successive days.
    dates = list(all_rates.keys())
    num_rows = len(dates) - 1  # Log-returns are computed from day j to day j+1.
    
    # Initialize a list to hold the log-return vectors (each as a column vector).
    vectors = [np.zeros((num_rows, 1)) for _ in range(num_variables)]
    
    # Compute the log-returns for each maturity.
    for j in range(num_rows):
        today_rates = all_rates[dates[j]]
        tomorrow_rates = all_rates[dates[j + 1]]
        for i in range(num_variables):
            maturity = float(i + 1)  # Maturities: 1, 2, 3, 4, 5 years.
            # Retrieve today's rate; if not directly available, interpolate.
            if maturity in today_rates:
                today_rate = today_rates[maturity]
            else:
                today_rate = interpolate_rate(today_rates, maturity)
            # Retrieve tomorrow's rate; if not directly available, interpolate.
            if maturity in tomorrow_rates:
                tomorrow_rate = tomorrow_rates[maturity]
            else:
                tomorrow_rate = interpolate_rate(tomorrow_rates, maturity)
            # Compute the log-return.
            vectors[i][j] = np.log(tomorrow_rate / today_rate)
    
    # Stack the vectors horizontally to form a data matrix.
    data = np.hstack(vectors)
    covariance_matrix = np.cov(data, rowvar=False)
    
    # Compute the eigenvalues and eigenvectors.
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Print outputs.
    print("Covariance Matrix for Yields:")
    print(covariance_matrix)
    print("Eigenvalues:")
    print(eigenvalues)
    print("Eigenvectors:")
    print(eigenvectors)
    
    return covariance_matrix, eigenvalues, eigenvectors


def calculate_covariance_matricies_for_forward_rates(all_rates, num_variables):
    """
    Calculate the covariance matrix of the daily log-returns of forward rates and
    compute its eigenvalues and eigenvectors.
    
    Parameters:
        all_rates (dict): A dictionary where each key is a date and each value is another
                          dictionary mapping maturities (e.g., forward rates for 1-yr, 2-yr, etc.)
                          to forward rate values.
        num_variables (int): Number of forward rate variables (e.g., 5 for 1-yr to 5-yr forward rates).
    
    Returns:
        tuple: (covariance_matrix, eigenvalues, eigenvectors)
    """
    dates = list(all_rates.keys())
    num_rows = len(dates) - 1  # Log-returns computed from successive days.
    
    vectors = [np.zeros((num_rows, 1)) for _ in range(num_variables)]
    
    # Compute the log-returns for forward rates.
    for j in range(num_rows):
        today_rates = all_rates[dates[j]]
        tomorrow_rates = all_rates[dates[j + 1]]
        for i in range(num_variables):
            maturity = float(i + 1)  # For example: 1-yr, 2-yr, etc.
            # Retrieve today's forward rate.
            if maturity in today_rates:
                today_rate = today_rates[maturity]
            else:
                # For maturity 1, if missing, use the smallest available key.
                if maturity == 1:
                    smallest_key = min(today_rates)
                    today_rate = today_rates[smallest_key]
                else:
                    today_rate = interpolate_rate(today_rates, maturity)
            # Retrieve tomorrow's forward rate.
            if maturity in tomorrow_rates:
                tomorrow_rate = tomorrow_rates[maturity]
            else:
                if maturity == 1:
                    smallest_key = min(tomorrow_rates)
                    tomorrow_rate = tomorrow_rates[smallest_key]
                else:
                    tomorrow_rate = interpolate_rate(tomorrow_rates, maturity)
            # Compute the log-return.
            vectors[i][j] = np.log(tomorrow_rate / today_rate)
    
    data = np.hstack(vectors)
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    print("Covariance Matrix for Forward Rates:")
    print(covariance_matrix)
    print("Eigenvalues:")
    print(eigenvalues)
    print("Eigenvectors:")
    print(eigenvectors)
    
    return covariance_matrix, eigenvalues, eigenvectors


# Optional main block for demonstration purposes.
if __name__ == "__main__":
    # Sample data for demonstration.
    # In practice, these dictionaries would be constructed from your computed yield or forward rate curves.
    # sample_yield_rates = {
    #     'Day1': {1.0: 0.03, 2.0: 0.035, 3.0: 0.04, 4.0: 0.045, 5.0: 0.05},
    #     'Day2': {1.0: 0.031, 2.0: 0.036, 3.0: 0.041, 4.0: 0.046, 5.0: 0.051},
    #     'Day3': {1.0: 0.032, 2.0: 0.037, 3.0: 0.042, 4.0: 0.047, 5.0: 0.052},
    #     'Day4': {1.0: 0.033, 2.0: 0.038, 3.0: 0.043, 4.0: 0.048, 5.0: 0.053},
    #     'Day5': {1.0: 0.034, 2.0: 0.039, 3.0: 0.044, 4.0: 0.049, 5.0: 0.054},
    # }
    # sample_forward_rates = {
    #     'Day1': {1.0: 0.031, 2.0: 0.033, 3.0: 0.035, 4.0: 0.037, 5.0: 0.039},
    #     'Day2': {1.0: 0.032, 2.0: 0.034, 3.0: 0.036, 4.0: 0.038, 5.0: 0.04},
    #     'Day3': {1.0: 0.033, 2.0: 0.035, 3.0: 0.037, 4.0: 0.039, 5.0: 0.041},
    #     'Day4': {1.0: 0.034, 2.0: 0.036, 3.0: 0.038, 4.0: 0.04, 5.0: 0.042},
    #     'Day5': {1.0: 0.035, 2.0: 0.037, 3.0: 0.039, 4.0: 0.041, 5.0: 0.043},
    # }
    
    # print("Calculating covariance matrix for yields:")
    # calculate_covariance_matricies_for_yields(sample_yield_rates, 5)
    
    # print("\nCalculating covariance matrix for forward rates:")
    
    # calculate_covariance_matricies_for_forward_rates(sample_forward_rates, 5)
    # But for the question, we need to compute yield_rates and forward_rates using the previous result.
    # Load the stored spot_rates dictionary.
    with open('yield_rates.pkl', 'rb') as f:
        yield_rates = pickle.load(f)
    
    with open('spot_rates.pkl', 'rb') as f:
        spot_rates = pickle.load(f)

    print("Loaded spot_rates from 'spot_rates.pkl'.")
    
    with open('forward_rates.pkl', 'rb') as f:
        forward_rates = pickle.load(f)

    print("Loaded forward_rates from 'forward_rates.pkl'.")

    print("Calculating covariance matrix for yields:")
    calculate_covariance_matricies_for_yields(yield_rates, 5)
    
    print("\nCalculating covariance matrix for forward rates:")
    
    calculate_covariance_matricies_for_forward_rates(forward_rates, 5)