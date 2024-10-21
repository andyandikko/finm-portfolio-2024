import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys
import cmds.portfolio_management_helper as pmh
import seaborn as sns
from sklearn.linear_model import LinearRegression
from dask.distributed import Client, wait
# From HW1
from typing import Tuple, Dict
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.metrics import r2_score

def load_data(filename, path="./data", sheet_name="excess returns"):
    """
    Load data from an Excel file and set the 'Date' column as the index.

    Parameters:
    - filename (str): The name of the Excel file to be loaded.
    - path (str): The directory path where the file is located. Default is './data'.
    - sheet_name (str): The sheet name from which data should be read. Default is 'excess returns'.

    Returns:
    - pd.DataFrame: A DataFrame with the data from the specified Excel sheet, with 'Date' as the index.
    
    Example:
    ```
    data = load_data("financial_data.xlsx", sheet_name="Sheet1")
    print(data.tail())
    print(data.dtypes)
    
    Alternatively,
    # alternative
    
    notebook_dir = "practice"
    if os.path.basename(os.getcwd()) == notebook_dir:
        os.chdir(os.pardir)  # Change to parent directory
        sys.path.append(os.getcwd())  # Add the new current directory to sys.path

    import cmds.portfolio_management_helper as pmh
    data = pmh.read_excel_default(os.path.join("./data",filename),sheet_name="excess returns")
    data = load_data(filename)
    display(data.tail())
    print(data.dtypes)
    
    
    ```
    """
    # Construct the file path
    file_path = os.path.join(path, filename)
    
    # Load the data from the Excel file
    data = pd.read_excel(file_path, sheet_name=sheet_name, parse_dates=True)
    
    
    # Set 'Date' column as the index
    # data.set_index("Date", inplace=True)
    
    return data


def calculate_summary_statistics(
    data: pd.DataFrame, 
    annual_factor: int = 12, 
    use_pmh: bool = False, 
    provided_excess_returns: bool = True
) -> pd.DataFrame:
    """
    Calculate annualized summary statistics including mean, volatility, and Sharpe ratio.
    
    Parameters:
    - data (pd.DataFrame): The input data containing returns.
    - annual_factor (int): The factor used for annualizing the returns and volatility. Default is 12 (monthly).
    - use_pmh (bool): If True, uses the `pmh.calc_summary_statistics` method as an alternative. Default is False.
    - provided_excess_returns (bool): A parameter to be passed to `pmh.calc_summary_statistics` when using the alternative method.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the annualized mean, volatility, and Sharpe ratio or results from `pmh.calc_summary_statistics`.
    
    Example:
    ```
    summary_stats = calculate_summary_statistics(data, annual_factor=12)
    summary_stats_pmh = calculate_summary_statistics(data, use_pmh=True)
    ```
    """
    if use_pmh:
        # Use the pmh method as an alternative approach
        try:
            summary = pmh.calc_summary_statistics(
                returns=data, 
                annual_factor=annual_factor, 
                provided_excess_returns=provided_excess_returns
            )
            return summary.sort_values(by="Annualized Sharpe", ascending=False)
        except Exception as e:
            raise Exception(f"An error occurred when using pmh method: {e}")
    
    # Calculate annualized mean, volatility, and Sharpe ratio
    try:
        annualised_mean = data.mean(axis=0) * annual_factor
        annualised_vol = data.std(axis=0) * np.sqrt(annual_factor)
        sharpe_ratio = annualised_mean / annualised_vol
        
        # Combine into a single DataFrame
        summary = pd.concat(
            [annualised_mean, annualised_vol, sharpe_ratio], 
            axis=1, 
            keys=["Mean", "Vol", "Sharpe"]
        )
        return summary
    except Exception as e:
        raise Exception(f"An error occurred when calculating summary statistics: {e}")

def calculate_tangency_portfolio(
    data: pd.DataFrame, 
    annual_factor: int = 12, 
    use_pmh: bool = False
) -> pd.DataFrame:
    """
    Calculate the tangency portfolio weights, mean, volatility, and Sharpe ratio.
    
    Parameters:
    - data (pd.DataFrame): The input data containing returns.
    - annual_factor (int): The factor used for annualizing the covariance matrix and returns. Default is 12 (monthly).
    - use_pmh (bool): If True, uses the `pmh.calc_tangency_weights` method as an alternative. Default is False.
    
    Returns:
    - pd.DataFrame: A DataFrame containing tangency portfolio weights or the results from `pmh.calc_tangency_weights`.
    
    Example:
    ```
    tangency_weights = calculate_tangency_portfolio(data, annual_factor=12)
    tangency_weights_pmh = calculate_tangency_portfolio(data, use_pmh=True)
    ```
    """
    if use_pmh:
        # Use the pmh method as an alternative approach
        try:
            tangency_weights = pmh.calc_tangency_weights(data, annual_factor=annual_factor)
            return tangency_weights
        except Exception as e:
            raise Exception(f"An error occurred when using pmh method: {e}")
    
    try:
        # Calculate covariance matrix and its inverse
        cov_matrix = data.cov()
        inverse_cov_matrix = np.linalg.inv(cov_matrix) * annual_factor
        
        # Calculate annualized mean returns
        mean_data = data.mean() * annual_factor
        
        # Tangency portfolio weights calculation
        tangency_weights = pd.DataFrame(
            (inverse_cov_matrix @ mean_data) / (inverse_cov_matrix @ mean_data @ np.ones(mean_data.shape)),
            index=data.columns, 
            columns=["Tangency Weights"]
        )
        
        # Sort tangency weights
        tangency_weights = tangency_weights.sort_values(by="Tangency Weights", ascending=False)
        
        # Tangency portfolio statistics
        tangency_mean = mean_data @ tangency_weights
        tangency_portfolio_var = (tangency_weights.T @ cov_matrix @ tangency_weights) * annual_factor
        tangency_vol = np.sqrt(tangency_portfolio_var)
        tangency_sharpe = tangency_mean / tangency_vol
        
        # Prepare results
        result = pd.DataFrame({
            "Tangency Mean": [tangency_mean.item()],
            "Tangency Volatility": [tangency_vol.item()],
            "Tangency Sharpe Ratio": [tangency_sharpe.item()]
        })
        
        # Combine tangency weights with portfolio statistics
        result = pd.concat([tangency_weights, result], axis=1)
        return result
    
    except Exception as e:
        raise Exception(f"An error occurred when calculating tangency portfolio: {e}")


def find_corr(
    correlation_matrix: pd.DataFrame, 
    type: str = "min", 
    absolute: bool = True
) -> tuple:
    """
    Find the minimum or maximum correlation in a correlation matrix.

    Parameters:
    - correlation_matrix (pd.DataFrame): The input correlation matrix.
    - type (str): Specify "min" to find the minimum correlation or "max" to find the maximum correlation. Default is "min".
    - absolute (bool): If True, considers absolute values when searching for min or max correlations. Default is True.

    Returns:
    - tuple: A tuple containing the row name, column name, and the correlation value.

    Example:
    ```
    find_corr(correlation_matrix, "min", absolute=True)
    find_corr(correlation_matrix, "max", absolute=False)
    ```
    """
    if type not in ["min", "max"]:
        raise ValueError("The 'type' parameter must be either 'min' or 'max'.")

    # Prepare the correlation matrix for processing
    matrix = np.absolute(correlation_matrix.to_numpy()) if absolute else correlation_matrix.to_numpy()

    if type == "min":
        val = matrix.min()
        corr = np.where(matrix == val)
    elif type == "max":
        # Subtract identity matrix to ignore diagonal (self-correlation)
        matrix -= np.eye(correlation_matrix.shape[0])
        val = matrix.max()
        corr = np.where(matrix == val)

    # Extract row and column information
    row = corr[0][0]
    col = corr[1][0]
    row_name = correlation_matrix.index[row]
    col_name = correlation_matrix.columns[col]

    # Output the result
    print(f"{type.capitalize()} correlation is between '{row_name}' and '{col_name}' with a coefficient of {val:.7f}")
    return row_name, col_name, val


def calculate_various_portfolio_metrics(
    data: pd.DataFrame, 
    target: float = 0.12, 
    annual_factor: int = 12, 
    use_pmh: bool = False
) -> pd.DataFrame:
    """
    Calculate various portfolio metrics for Tangency, Equal Weighted, Risk Parity, and Regularized portfolios.

    Parameters:
    - data (pd.DataFrame): The input data containing asset returns.
    - target (float): The target annualized mean return. Default is 0.12.
    - annual_factor (int): The factor used for annualizing returns and volatility. Default is 12 (monthly).
    - use_pmh (bool): If True, uses the `pmh` methods to compute the portfolios. Default is False.
    
    Returns:
    - pd.DataFrame: A DataFrame with annualized mean, volatility, and Sharpe ratio for each portfolio.
    
    Example:
    ```
    metrics = calculate_various_portfolio_metrics(data, target=0.1, annual_factor=12)
    metrics_pmh = calculate_various_portfolio_metrics(data, use_pmh=True)
    ```
    """
    if use_pmh:
        try:
            n_assets = len(data.columns)
            # Equal Weights Portfolio
            portfolio_equal_weights = pmh.create_portfolio(
                data, weights=[1 / n_assets for _ in range(n_assets)], port_name="Equal Weights"
            ) * target / pmh.create_portfolio(data, weights=[1 / n_assets for _ in range(n_assets)]).mean()

            # Risk Parity Portfolio
            asset_variance_dict = data.std().map(lambda x: x ** 2).to_dict()
            asset_inv_variance_dict = {asset: 1 / variance for asset, variance in asset_variance_dict.items()}
            portfolio_risk_parity = pmh.create_portfolio(
                data, weights=asset_inv_variance_dict, port_name="Risk Parity"
            ) * target / pmh.create_portfolio(data, weights=asset_inv_variance_dict).mean()

            # Tangency Portfolio
            portfolio_tangency = pmh.calc_tangency_weights(data, return_port_ret=True) * target / pmh.calc_tangency_weights(data, return_port_ret=True).mean()

            # Regularized Portfolio
            portfolio_regularized = pmh.calc_tangency_weights(
                data, return_port_ret=True, cov_mat=0.5, name="Regularized"
            ) * target / pmh.calc_tangency_weights(data, return_port_ret=True, cov_mat=0.5).mean()

            # Compile results
            portfolios = pd.concat(
                [portfolio_equal_weights, portfolio_risk_parity, portfolio_tangency, portfolio_regularized], axis=1
            )
            return pmh.calc_summary_statistics(
                portfolios, provided_excess_returns=True, annual_factor=annual_factor, keep_columns=['Annualized Mean', 'Annualized Vol', 'Annualized Sharpe']
            )

        except Exception as e:
            raise Exception(f"An error occurred when using pmh methods: {e}")

    try:
        # Tangency Portfolio
        tangency_weights = np.linalg.inv(data.cov()) @ data.mean()
        tangency_portfolio_mean = data.mean() @ tangency_weights * annual_factor
        scaled_tangency_weights = (target / tangency_portfolio_mean) * tangency_weights
        tangency_mean = scaled_tangency_weights @ data.mean() * annual_factor
        tangency_vol = np.sqrt(scaled_tangency_weights @ data.cov() @ scaled_tangency_weights) * np.sqrt(annual_factor)
        tangency_sharpe = tangency_mean / tangency_vol
        Tangency_df = pd.DataFrame(
            {"Tangency": [tangency_mean, tangency_vol, tangency_sharpe]},
            index=["Annualised_Mean", "Annualised_Vol", "Annualised_Sharpe"]
        )

        # Equal Weighted Portfolio
        number_of_securities = data.columns.size
        equal_weights = np.repeat(1 / number_of_securities, number_of_securities)
        equal_weight_portfolio_mean = data.mean() @ equal_weights * annual_factor
        scaled_equal_weights = (target / equal_weight_portfolio_mean) * equal_weights
        ew_mean = scaled_equal_weights @ data.mean() * annual_factor
        ew_vol = np.sqrt(scaled_equal_weights @ data.cov() @ scaled_equal_weights) * np.sqrt(annual_factor)
        ew_sharpe = ew_mean / ew_vol
        EW_df = pd.DataFrame(
            {"EW": [ew_mean, ew_vol, ew_sharpe]},
            index=["Annualised_Mean", "Annualised_Vol", "Annualised_Sharpe"]
        )

        # Risk Parity Portfolio
        data_variance = data.var()
        rp_weights = 1 / data_variance
        risk_parity_portfolio_mean = data.mean() @ rp_weights * annual_factor
        scaled_risk_parity = (target / risk_parity_portfolio_mean) * rp_weights
        rp_mean = scaled_risk_parity @ data.mean() * annual_factor
        rp_vol = np.sqrt(scaled_risk_parity @ data.cov() @ scaled_risk_parity) * np.sqrt(annual_factor)
        rp_sharpe = rp_mean / rp_vol
        RP_df = pd.DataFrame(
            {"RP": [rp_mean, rp_vol, rp_sharpe]},
            index=["Annualised_Mean", "Annualised_Vol", "Annualised_Sharpe"]
        )

        # Regularized Portfolio
        reg_cov_matrix = (data.cov() + np.diag(np.diag(data.cov()))) / 2
        reg_weights = np.linalg.inv(reg_cov_matrix) @ data.mean()
        reg_portfolio_mean = data.mean() @ reg_weights * annual_factor
        scaled_reg_weights = (target / reg_portfolio_mean) * reg_weights
        reg_mean = scaled_reg_weights @ data.mean() * annual_factor
        reg_vol = np.sqrt(scaled_reg_weights @ data.cov() @ scaled_reg_weights) * np.sqrt(annual_factor)
        reg_sharpe = reg_mean / reg_vol
        Reg_df = pd.DataFrame(
            {"Reg": [reg_mean, reg_vol, reg_sharpe]},
            index=["Annualised_Mean", "Annualised_Vol", "Annualised_Sharpe"]
        )

        # Compile results and show correlations heatmap
        result_df = pd.concat([Tangency_df, EW_df, RP_df, Reg_df], axis=1).T
        
        # Calculate weighted returns and show correlation heatmap
        array_of_weights = np.array(
            [scaled_tangency_weights, scaled_equal_weights, scaled_risk_parity, scaled_reg_weights]
        )
        weighted_return_data = (data @ array_of_weights.T).rename(columns={0: "Tangency", 1: "EW", 2: "RP", 3: "Reg"})
        sns.heatmap(weighted_return_data.corr(), annot=True)
        plt.show()
        
        return result_df

    except Exception as e:
        raise Exception(f"An error occurred when calculating portfolio metrics: {e}")


def pmh_portfolio_analysis(
    in_sample_data: pd.DataFrame, 
    out_of_sample_data: pd.DataFrame, 
    target_return: float = 0.01, 
    return_returns: bool = False
) -> pd.DataFrame | pd.Series:
    """
    Perform in-sample and out-of-sample portfolio analysis using `pmh` methods.
    
    Parameters:
    ----------
    in_sample_data : pd.DataFrame
        Historical return data used for calculating in-sample portfolio weights.
        
    out_of_sample_data : pd.DataFrame
        Out-of-sample data used for evaluating portfolio performance or calculating returns.
        
    target_return : float, optional
        The desired annualized return for scaling portfolio weights. Default is 0.01 (1%).
    
    return_returns : bool, optional
        If True, return the portfolio returns for each portfolio type. Default is False.
    
    Returns:
    -------
    pd.DataFrame or pd.Series
        - If `return_returns` is False, returns a DataFrame with portfolio statistics for each portfolio type.
        - If `return_returns` is True, returns a Series of out-of-sample returns for each portfolio type.
    
    Example:
    -------
    ```
    
    # Example usage:
    in_sample_data = data.loc["2011-02-28":"2022-12-31", :]
    out_of_sample_data = data.loc["2023-01-01":, :]

    # Get OOS statistics
    pmh_results = pmh_portfolio_analysis(in_sample_data, out_of_sample_data, target_return=0.12)
    display(pmh_results)

    # Get OOS returns for further analysis
    pmh_returns = pmh_portfolio_analysis(in_sample_data, out_of_sample_data, target_return=0.12, return_returns=True)
    display(pmh_returns)
    
    ```
    """
    # Regularized Portfolio
    in_sample_weights_regularized = pmh.calc_tangency_weights(in_sample_data, cov_mat=0.5, name="Regularized")

    # Tangency Portfolio
    in_sample_weights_tangency = pmh.calc_tangency_weights(in_sample_data)

    # No TIPS Tangency Portfolio
    in_sample_weights_no_tips = pmh.calc_tangency_weights(in_sample_data, name="No TIPS Tangency")

    # Modified TIPS Tangency Portfolio
    in_sample_weights_modified_tips = pmh.calc_tangency_weights(in_sample_data, name="Mod TIPS Tangency")

    # Risk Parity Portfolio
    in_sample_asset_variance_dict = in_sample_data.std().map(lambda x: x ** 2).to_dict()
    in_sample_asset_inv_variance_dict = {asset: 1 / variance for asset, variance in in_sample_asset_variance_dict.items()}
    in_sample_weights_risk_parity = pd.DataFrame(in_sample_asset_inv_variance_dict, index=["Risk Parity Weights"]).transpose()

    # Equal Weights Portfolio
    n_assets = len(in_sample_data.columns)
    in_sample_weights_equal = pd.DataFrame(
        data=[[1 / n_assets] for _ in range(n_assets)],
        columns=["Equal Weights"],
        index=in_sample_data.columns
    )

    # Combine all in-sample weights
    in_sample_weights = (
        pd.concat([
            in_sample_weights_regularized,
            in_sample_weights_tangency,
            in_sample_weights_no_tips,
            in_sample_weights_modified_tips,
            in_sample_weights_risk_parity,
            in_sample_weights_equal
        ], axis=1)
        .fillna(0)
    )

    # Scale weights to meet target return
    in_sample_weights_scaled = (
        in_sample_weights
        .apply(lambda weights: weights * target_return / (in_sample_data @ weights).mean())
    )

    if return_returns:
        # Calculate and return OOS returns
        oos_returns = out_of_sample_data @ in_sample_weights_scaled
        return oos_returns
    
    # Calculate OOS performance statistics
    oos_stats = pmh.calc_summary_statistics(
        out_of_sample_data @ in_sample_weights_scaled,
        annual_factor=12,
        provided_excess_returns=True,
        keep_columns=['Annualized mean', 'Annualized vol', 'Annualized sharpe']
    ).sort_values('Annualized Sharpe', ascending=False)

    return oos_stats


def rolling_oos_performance(
    assets_excess_returns: pd.DataFrame,
    in_sample_end_date: str,
    out_of_sample_start_date: str,
    out_of_sample_end_date: str,
    MU_MONTH_TARGET: float = 0.01
) -> pd.DataFrame:
    """
    Calculate rolling Out-of-Sample (OOS) portfolio performance using various portfolio strategies.

    Parameters:
    ----------
    assets_excess_returns : pd.DataFrame
        Historical return data of assets, with each column representing an asset and each row representing a time period.
        
    in_sample_end_date : str
        The end date of the in-sample period (format: "YYYY-MM-DD").
        
    out_of_sample_start_date : str
        The start date of the out-of-sample period (format: "YYYY-MM-DD").
        
    out_of_sample_end_date : str
        The end date of the out-of-sample period (format: "YYYY-MM-DD").
        
    MU_MONTH_TARGET : float, optional
        Target monthly return for rescaling weights. Default is 0.01 (1%).

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the out-of-sample portfolio returns for each strategy.

    Example:
    -------
    ```
    
    oos_portfolios_performance = pd.DataFrame({})

    for in_sample_last_year in range(2015, 2024):
        oos_portfolios_yearly_performance = rolling_oos_performance(
            data,
            f"{in_sample_last_year}-12-31",
            f"{in_sample_last_year+1}-01-01",
            f"{in_sample_last_year+1}-12-31",
            MU_MONTH_TARGET=0.01
        )
        oos_portfolios_performance = pd.concat([oos_portfolios_performance, oos_portfolios_yearly_performance])

    display(oos_portfolios_performance)

    display(pmh.calc_cummulative_returns(oos_portfolios_performance))
    display(pmh.calc_summary_statistics(
        oos_portfolios_performance,
        annual_factor=12,
        provided_excess_returns=True,
        keep_columns=['Annualized Mean', 'Annualized Vol', 'Annualized Sharpe']
    ))
    ```
    """
    # Make a copy of the input data
    assets_excess_returns = assets_excess_returns.copy()
    n_assets = assets_excess_returns.shape[1]

    # Define in-sample and out-of-sample data
    in_sample_assets_excess_returns = assets_excess_returns.loc[:in_sample_end_date]
    out_of_sample_assets_excess_returns = (
        assets_excess_returns.loc[out_of_sample_start_date:out_of_sample_end_date]
    )

    # Regularized Portfolio
    in_sample_weights_regularized = pmh.calc_tangency_weights(in_sample_assets_excess_returns, cov_mat=0.5, name="Regularized")

    # Tangency Portfolio
    in_sample_weights_tangency = pmh.calc_tangency_weights(in_sample_assets_excess_returns)

    # Risk Parity Portfolio
    in_sample_asset_variance_dict = in_sample_assets_excess_returns.std().map(lambda x: x ** 2).to_dict()
    in_sample_asset_inv_variance_dict = {asset: 1 / variance for asset, variance in in_sample_asset_variance_dict.items()}
    in_sample_weights_risk_parity = pd.DataFrame(in_sample_asset_inv_variance_dict, index=["Risk Parity Weights"]).transpose()

    # Equal Weights Portfolio
    in_sample_weights_equal = pd.DataFrame(
        data=[[1 / n_assets] for _ in range(n_assets)],
        columns=["Equal Weights"],
        index=in_sample_assets_excess_returns.columns
    )

    # Combine all in-sample weights
    in_sample_weights = (
        pd.concat([
            in_sample_weights_regularized,
            in_sample_weights_tangency,
            in_sample_weights_risk_parity,
            in_sample_weights_equal
        ], axis=1)
        .fillna(0)
    )

    # Rescale Weights to Match Target Return
    in_sample_weights_scaled = (
        in_sample_weights
        .apply(lambda weights: weights * MU_MONTH_TARGET / (in_sample_assets_excess_returns @ weights).mean())
    )

    # Calculate and return OOS returns
    return out_of_sample_assets_excess_returns @ in_sample_weights_scaled


def univariate_factor_decomposition_stats(
    data: pd.DataFrame, 
    benchmark: pd.Series, 
    annual_factor: int = 12,
    intercept: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform univariate factor decomposition on multiple funds using a specified independent variable (e.g., benchmark returns).
    
    This function calculates Betas, Alphas, Annualised Treynor Ratios, Annualised Information Ratios, and Residuals for each fund 
    in the `data` DataFrame by regressing each fund's returns against the returns of an independent variable (e.g., a benchmark like SPY). 
    
    Parameters:
    ----------
    data : pd.DataFrame
        DataFrame containing the returns of multiple funds, with each column representing a different fund.
        
    benchmark : pd.Series
        Independent variable (e.g., benchmark returns such as SPY, market index, or any other factor) 
        used to regress against the funds' returns.
        
    annual_factor : int, optional (default=12)
        The factor used to annualize monthly returns and other metrics. 
        Common values:
        - 12 for monthly data (annualized to a year)
        - 252 for daily data (annualized to a year assuming 252 trading days)
    intercept : bool, optional (default=True)
        If True, includes an intercept term (Alpha) in the regression; if False, no intercept will be included.
        
        
    Returns:
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - pd.DataFrame: A DataFrame with Betas, Alphas, Annualised Treynor Ratios, and Annualised Information Ratios for each fund.
        - pd.DataFrame: A DataFrame containing the residuals of the regression for each fund.
    
    Concepts:
    ---------
    **Beta (β):**
        Beta measures the sensitivity of a fund's returns to the independent variable (e.g., market index like SPY or any benchmark). 
        A Beta of 1 indicates that the fund moves in line with the market, while a Beta greater than 1 suggests higher volatility 
        relative to the market, and a Beta less than 1 suggests lower volatility.

    **Alpha (α):**
        Alpha represents the intercept of the regression and measures the excess return of a fund beyond what can be explained 
        by its exposure to the independent variable (Beta). A positive Alpha indicates that the fund has outperformed 
        relative to its expected return, while a negative Alpha suggests underperformance.

    **Treynor Ratio:**
        The Treynor Ratio is an alternative measure of the risk-reward tradeoff, calculated as the mean return of the fund divided by its Beta. 
        It is used to evaluate the performance of an investment by measuring returns per unit of systematic risk (market risk). 
        Typically, it is used to assess a fund's performance relative to a market benchmark. 
        The Treynor Ratio is annualized using the `annual_factor` parameter.

        Formula:
        \[
        \text{Treynor Ratio} = \frac{\text{Mean Return} \times \text{Annual Factor}}{\text{Beta}}
        \]

    **Information Ratio:**
        The Information Ratio is similar to the Sharpe Ratio but focuses on the non-factor component of returns 
        (i.e., Alpha plus Epsilon, where Epsilon represents the random error or residuals). 
        It measures the risk-adjusted performance of the excess return (Alpha) relative to the non-factor volatility (Epsilon).
        Alpha is the mean of the non-factor component, and the standard deviation of the residuals represents the non-factor volatility.
        This ratio indicates how much excess return is achieved per unit of non-systematic risk, helping investors gauge a fund's 
        performance beyond the influence of the market factor. The Information Ratio is also annualized using the `annual_factor` parameter.

        Formula:
        \[
        \text{Information Ratio} = \frac{\alpha \times \sqrt{\text{Annual Factor}}}{\sigma(\epsilon)}
        \]
        Where:
        - \(\alpha\) is the intercept (excess return beyond the factor)
        - \(\sigma(\epsilon)\) is the standard deviation of the residuals (non-factor volatility)

    **Residuals:**
        Residuals represent the difference between the actual returns and the predicted returns based on the regression model.
        Residuals help in understanding the part of the fund's returns that cannot be explained by its exposure to the independent variable.
    """
    
    # Fit Linear Regression Models using Dask for parallel processing
    with Client() as client:
        futures = []
        for col in data.columns:
            model = LinearRegression(fit_intercept=intercept)
            X = benchmark.values.reshape(-1, 1)  # Independent variable
            y = data[col].values  # Dependent variable for each column
            future = client.submit(model.fit, X, y)
            futures.append((col, future))
        
        # Collect results
        results = {col_name: future.result() for col_name, future in futures}

    # Calculate Betas and Alphas
    betas = {col: results[col].coef_[0] for col in data.columns}
    alphas = {col: results[col].intercept_ if intercept else 0 for col in data.columns}

    # Calculate Residuals for each column
    residuals = {
        col: data[col] - results[col].predict(benchmark.values.reshape(-1, 1)) for col in data.columns
    }

    # Create DataFrames for Betas and Alphas
    betas_df = pd.DataFrame(betas.items(), columns=["Fund", "Beta"]).set_index("Fund")
    alphas_df = pd.DataFrame(alphas.items(), columns=["Fund", "Alpha"]).set_index("Fund")

    # Treynor Ratio (Average Return / Beta)
    treynor_ratio = {col: data[col].mean() * annual_factor / betas[col] for col in data.columns}
    treynor_ratio_df = pd.DataFrame(treynor_ratio.items(), columns=["Fund", "Treynor Ratio"]).set_index("Fund")

    # Information Ratio (Alpha / Residual Standard Deviation)
    information_ratio = {col: alphas[col] * np.sqrt(annual_factor) / residuals[col].std() for col in data.columns}
    information_ratio_df = pd.DataFrame(information_ratio.items(), columns=["Fund", "Information Ratio"]).set_index("Fund")

    # Combine all statistics into a single DataFrame
    stats_df = pd.concat([betas_df, alphas_df, treynor_ratio_df, information_ratio_df], axis=1)

    # Combine Residuals into a DataFrame
    residuals_df = pd.DataFrame(residuals)

    return stats_df, residuals_df


def multivariate_regression_statistics(
    dependent_data: pd.Series, 
    independent_data: pd.DataFrame, 
    annual_factor: int = 12,
    intercept: bool = True
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Perform multivariate linear regression analysis to evaluate the relationship between a dependent variable (e.g., HFRI index) 
    and multiple independent variables (e.g., factors from Merrill Lynch data).
    
    This function calculates key regression metrics, including Betas, Intercept (Alpha), R-squared, Tracking Error, Treynor Ratio, 
    and Information Ratio. It returns a dictionary with these statistics and a DataFrame of the Beta coefficients for each independent variable.
    
    Parameters:
    ----------
    dependent_data : pd.Series
        Series containing the returns of the dependent variable (e.g., HFRI index).
        
    independent_data : pd.DataFrame
        DataFrame containing the independent variables (e.g., factor returns from Merrill Lynch data).
        
    annual_factor : int, optional (default=12)
        The factor used to annualize monthly returns and metrics. 
        Common values:
        - 12 for monthly data (annualized to a year)
        - 252 for daily data (annualized to a year assuming 252 trading days)
    intercept : bool, optional (default=True)
        If True, includes an intercept term (Alpha) in the regression; if False, no intercept will be included.
        
        
    Returns:
    -------
    Tuple[Dict[str, float], pd.DataFrame]
        - Dict: A dictionary containing key regression statistics:
            * Beta (list of beta coefficients for each factor)
            * Intercept (Alpha)
            * R-squared (R2)
            * Tracking Error (Standard deviation of residuals, annualized)
            * Treynor Ratio (Average return of dependent variable divided by Beta)
            * Information Ratio (Intercept / Tracking Error)
            
        - pd.DataFrame: A DataFrame of Beta coefficients for each independent variable.
    
    Concepts:
    ---------
    **Beta Coefficients:**
        Represent the sensitivity of the dependent variable to each of the independent variables (factors). 
        A higher beta indicates a stronger relationship between the factor and the dependent variable.
    
    **Intercept (Alpha):**
        The intercept from the regression model, representing the excess return of the dependent variable 
        that cannot be explained by the independent variables (factors).
    
    **R-squared (R2):**
        The proportion of the variance in the dependent variable that is predictable from the independent variables. 
        R2 values closer to 1 indicate a better fit.
    
    **Tracking Error:**
        The standard deviation of the residuals from the regression, indicating how closely the predicted returns track 
        the actual returns of the dependent variable. This is annualized using the `annual_factor` parameter.
    
    **Treynor Ratio:**
        A measure of risk-adjusted return, calculated as the average return of the dependent variable divided by the Beta coefficient.
        This is annualized by multiplying by `annual_factor`.
    
    **Information Ratio:**
        Similar to the Sharpe Ratio but focuses on the non-factor component of returns (Alpha + Residuals). 
        It is calculated as the Intercept divided by the Tracking Error, annualized.
        
    
    Example:
    ```
    merrill_data = fa.load_data(filename="proshares_analysis_data.xlsx", path="./data", sheet_name="merrill_factors")
    merrill_data.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    merrill_data.set_index("Date", inplace=True)

    hfri_data = data.loc[:, "HFRIFWI Index"]
    results, beta_df = multivariate_regression_statistics(hfri_data, merrill_data)

    # Display the results
    print("Multivariate Regression Statistics:")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Display Beta coefficients DataFrame
    display(beta_df)    
    ``` 
    
    """
    
    # Prepare the data for regression
    X = independent_data.values
    y = dependent_data.values.reshape(-1, 1)
    
    # Fit the linear regression model
    model = LinearRegression(fit_intercept=intercept)
    model.fit(X, y)
    
    # Extract regression coefficients and intercept
    beta = model.coef_[0]
    intercept_value = model.intercept_[0] if intercept else 0
    # Calculate residuals
    residuals = y - model.predict(X)
    
    # Calculate Tracking Error (Annualized)
    tracking_error = np.std(residuals) * np.sqrt(annual_factor)
    
    # Calculate R-squared
    r2 = model.score(X, y)
    
    # Calculate Treynor Ratio (Annualized)
    average_return = dependent_data.mean() * annual_factor
    treynor_ratio = average_return / np.mean(beta)
    
    
    
    # Calculate Information Ratio (Annualized)
    information_ratio = (intercept_value * np.sqrt(annual_factor)) / tracking_error
    
    # Prepare results dictionary
    results = {
        "Betas": beta,
        "Intercept": intercept_value,
        "R-squared": r2,
        "Tracking Error": tracking_error,
        "Treynor Ratio": treynor_ratio,
        "Information Ratio": information_ratio,
        "Fitted Mean": np.mean(model.predict(X)),
        "model": model
    }
    
    # Create DataFrame for Beta coefficients
    beta_df = pd.DataFrame({col: beta[i] for i, col in enumerate(independent_data.columns)}, index=["Beta"]).T
    
    return results, beta_df

def one_step_OOS_rolling_regression(
    feature_data: pd.DataFrame, 
    target_data: pd.Series, 
    window_size: int, 
    n_workers: int = 3, 
    threads_per_worker: int = 4,
    intercept: bool = True
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Performs a one-step out-of-sample rolling regression using Dask for parallel processing.
    
    This function trains multiple linear regression models on a rolling window of data, then makes predictions 
    for the next step out-of-sample. The models are trained in parallel using Dask. Additionally, the function calculates 
    out-of-sample (OOS) R-squared and other OOS performance statistics.
    
    Parameters:
    ----------
    feature_data : pd.DataFrame
        DataFrame containing the independent variables (factors) used for regression.
        
    target_data : pd.Series
        Series containing the dependent variable (e.g., HFRI index or any other target).
        
    window_size : int
        The size of the rolling window used for training the regression model.
        
    n_workers : int, optional (default=3)
        Number of worker processes to use for Dask's parallel processing.
        
    threads_per_worker : int, optional (default=4)
        Number of threads per worker process to use for parallel execution.
        
    intercept : bool, optional (default=True)
        If True, includes an intercept term in the regression; if False, no intercept will be included.
        
    Returns:
    -------
    Tuple[Dict[str, float], pd.Series]
        - Dict: A dictionary containing the following OOS statistics:
            * OOS R2 (Out-of-sample R-squared)
            * Total predictions
            * Mean OOS prediction error
        - pd.Series: Full OOS predictions data.
    
    Explanation:
    ------------
    The function performs rolling window regression by fitting a linear regression model over a specified window size.
    It then makes a one-step out-of-sample prediction. The models are trained in parallel using Dask, which speeds up 
    the computation. Afterward, the function validates the predictions and computes the OOS R2 to evaluate performance.
    ```
    window_size = 60
    oos_stats, oos_results = one_step_OOS_rolling_regression(merrill_data, hfri_data, window_size)

    # Display results
    print("Out-of-Sample Rolling Regression Statistics:")
    for key, value in oos_stats.items():
        print(f"{key}: {value}")

    print("Full OOS Results:")
    display(oos_results)
    ```
    
    """
    
    # Define the rolling training function for a single step
    def train_model(feature_data: pd.DataFrame, target_data: pd.Series, start_index: int, window_size: int):
        model = LinearRegression(fit_intercept=intercept)
        model.fit(
            X=feature_data.iloc[start_index:start_index + window_size].values, 
            y=target_data.iloc[start_index:start_index + window_size].values
        )
        predictions = model.predict(feature_data.iloc[start_index + window_size].values.reshape(1, -1))
        return predictions
    
    # Using Dask to parallelize the rolling window regression
    try:
        with Client(n_workers=n_workers, threads_per_worker=threads_per_worker) as client:
            # Generate start indices for rolling windows
            start_indices = list(range(target_data.shape[0] - window_size))
            futures = [
                client.submit(train_model, feature_data, target_data, start, window_size) 
                for start in start_indices
            ]
            wait(futures, return_when="ALL_COMPLETED")
            results = client.gather(futures)
            results = np.array(results).flatten()
    except Exception as e:
        print(f"Error during rolling regression: {e}")
        return {}, pd.Series([])

    # Validation
    num_windows = target_data.shape[0] - window_size
    cond2 = num_windows == len(results)
    model = LinearRegression(fit_intercept=intercept)
    model.fit(feature_data.iloc[:window_size].values, target_data.iloc[:window_size].values)
    cond1 = float(model.predict(feature_data.iloc[window_size].values.reshape(1, -1))) == results[0]

    if not (cond1 and cond2):
        raise ValueError("Validation failed")

    # Calculate Out-of-Sample R2
    OOS_r2 = r2_score(target_data[window_size:].values, results)
    
    # Compile OOS Statistics
    oos_stats = {
        "OOS R2": OOS_r2,
        "Total Predictions": len(results),
        "Mean OOS Prediction Error": np.mean(target_data[window_size:].values - results)
    }
    
    return oos_stats, pd.Series(results, index=target_data.index[window_size:])


def simplified_rolling_ols_regression(
    feature_data: pd.DataFrame, 
    target_data: pd.Series, 
    window_size: int,
    intercept: bool = True
) -> Tuple[float, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Perform a simplified rolling regression using `RollingOLS` from `statsmodels` and calculate out-of-sample R-squared.
    
    This function fits rolling linear regression models over a specified window size and computes replication performance 
    based on the rolling betas. It then calculates the OOS R-squared statistic to assess the predictive performance 
    and provides a correlation matrix between the actual and replicated data.
    
    Parameters:
    ----------
    feature_data : pd.DataFrame
        DataFrame containing the independent variables (factors) used for regression.
        
    target_data : pd.Series
        Series containing the dependent variable (e.g., HFRI index or any other target).
        
    window_size : int
        The size of the rolling window used for training the regression model.
        
    intercept : bool, optional (default=True)
        If True, includes an intercept term in the regression; if False, no intercept will be included.
        
    Returns:
    -------
    Tuple[float, pd.DataFrame, pd.Series, pd.DataFrame]
        - float: Out-of-Sample R-squared (OOS R2)
        - pd.DataFrame: DataFrame containing the rolling betas from the regression.
        - pd.Series: Series containing the replicated returns based on rolling predictions.
        - pd.DataFrame: Correlation matrix between actual and replicated OOS data.
    
    Explanation:
    ------------
    The function uses `RollingOLS` to fit rolling window regression models and predict target values based on a 
    rolling set of beta coefficients. It calculates the OOS R-squared to evaluate the predictive accuracy of the model.
    Additionally, the function computes the correlation between the actual and predicted OOS values to assess the strength 
    of their linear relationship.
    
    Usage:
    -----
    window_size = 60
    oos_r2, rolling_betas, replicated_rolling, oos_correlation = simplified_rolling_ols_regression(
        feature_data=merrill_data, 
        target_data=hfri_data, 
        window_size=window_size,
        intercept=True
    )
    
    print(f"The Out-of-Sample R-Squared of the replication is {round(oos_r2, 4)}")
    print("\nRolling Betas:")
    display(rolling_betas.head())
    
    print("\nReplicated Returns Based on Rolling Predictions:")
    display(replicated_rolling.head())
    
    print("\nCorrelation Matrix Between Actual and Replicated OOS Data:")
    display(oos_correlation)
    """
    
    # Optionally add a constant to the feature data for the intercept term
    if intercept:
        feature_data = sm.add_constant(feature_data)
    
    # Fit the rolling OLS model
    rolling_model = RollingOLS(target_data, feature_data, window=window_size).fit()
    
    # Extract rolling betas (coefficients)
    rolling_betas = rolling_model.params
    
    # Calculate replicated values using the rolling betas shifted by one period (to avoid look-ahead bias)
    replicated_rolling = (rolling_betas.shift() * feature_data).dropna().sum(axis=1)
    
    # Calculate OOS loss and OOS R-squared
    oos_loss = ((target_data.loc[replicated_rolling.index] - replicated_rolling) ** 2).sum()
    oos_mean = target_data.loc[replicated_rolling.index].dropna().mean()
    oos_loss_null = ((target_data.loc[replicated_rolling.index].dropna() - oos_mean) ** 2).sum()
    oos_r2 = 1 - (oos_loss / oos_loss_null)
    
    # Correlation matrix between actual and replicated OOS data
    actual_vs_replication = pd.DataFrame({
        "Actual": target_data.loc[replicated_rolling.index],
        "Replicated": replicated_rolling
    })
    oos_correlation = actual_vs_replication.corr()
    
    return oos_r2, rolling_betas, replicated_rolling, oos_correlation


