import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from risk_correlation.main import get_historical_data

# traning machine learning model
def train_correlation_model(data):
    # Prepare the data
    X = data.index.values.reshape(-1, 1)  # Independent variable (time)
    Y = data.values  # Dependent variables (asset prices)

    # Train the model
    model = LinearRegression()
    model.fit(X, Y)

    return model

# Training Dependency Model

def train_dependency_model(data, target_asset, dependent_asset):
    # Prepare the data
    X = data[dependent_asset].values.reshape(-1, 1)
    Y = data[target_asset].values

    # Train the model
    model = RandomForestRegressor()
    model.fit(X, Y)

    return model

# Now, fine tuning the correlation model
def fine_tune_correlation_model(model, new_data):
    # Update the model with new data
    X_new = new_data.index.values.reshape(-1, 1)
    Y_new = new_data.values

    model.fit(X_new, Y_new)

    return model

# Now, fine tuning the dependency model 
def fine_tune_dependency_model(model, new_data, target_asset, dependent_asset):
    # Update the model with new data
    X_new = new_data[dependent_asset].values.reshape(-1, 1)
    Y_new = new_data[target_asset].values

    model.fit(X_new, Y_new)

    return model

# generating correlation matrix and heatmap 
def plot_correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

# analyzing dependency models for feature importance 
def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.show()

# Generating residual plots
def plot_residuals(model, X, y):
    predictions = model.predict(X)
    residuals = y - predictions

    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals)
    plt.hlines(y=0, xmin=min(predictions), xmax=max(predictions), colors='r')
    plt.title("Residual Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()

# Revising portfolio for risk management
def adjust_portfolio(correlation_matrix, dependency_model, scenario_analysis, tickers):

    # setup output list
    adjust_portfolio_res = []
    # Example logic for adjusting the portfolio
    high_risk_assets = [asset for asset, corr in correlation_matrix[tickers[-1]].items() if corr > 0.7]
    adjust_portfolio_res.append("High-risk assets based on correlation analysis: {high_risk_assets}")

    important_features = dependency_model.feature_importances_
    adjust_portfolio_res.append("Important factors impacting {tickers[0]}: {important_features}")

    scenario_impact = scenario_analysis["impact"]
    adjust_portfolio_res.append("Scenario analysis impact: {scenario_impact}")

    return adjust_portfolio_res

# getting INPUT DATA & generate OUTPUT now
# collect user inputs
def get_models(tickers, start_date, end_date, query):

    # historical stock data 
    stock_data = get_historical_data(tickers, start_date, end_date)

    # converting ticker input into a list
    ticker_list = [s.strip() for s in list(tickers.split(","))]

    # generatre different models based on the user inputs
    correlation_model = train_correlation_model(stock_data)
    dependency_model = train_dependency_model(stock_data, target_asset=ticker_list[0], dependent_asset=ticker_list[-1])


    # fine-tune correlation the model
    new_data = get_historical_data(tickers, end_date, end_date + datetime.timedelta(days=2310))
    fine_tuned_correlation_model = fine_tune_correlation_model(correlation_model, new_data)

    # fine-tune dependency the model
    fine_tuned_dependency_model = fine_tune_dependency_model(dependency_model, new_data, target_asset=ticker_list[0], dependent_asset=ticker_list[-1])

    return correlation_model, dependency_model, fine_tuned_correlation_model, fine_tuned_dependency_model






