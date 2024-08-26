import os
import datetime
import streamlit as st
from dotenv import load_dotenv
from risk_correlation.main import get_historical_data, get_financial_news, generate_prompts, generate_correlation_response
from risk_correlation.ml_models import plot_feature_importances, plot_residuals, adjust_portfolio, plot_correlation_matrix, \
    get_models

load_dotenv()

# load API keys
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

st.title("Risk Correlation and Dependency Analysis")

with st.form(key="user_interaction"):

    # get stock ticker info
    tickers = st.text_input(
        label = "Please enter stock tickers with comma seperation.",
        max_chars= 40,
        placeholder = "e.g., APPL, MSFT, GOOGLE, AMZN, WMT"
    )

    news_query = st.text_input(

        label = "Please specify the sector of the selected stocks.",
        max_chars = 200,
        placeholder = "e.g., technology stocks"
    )

    start_date = st.date_input(
        label = "Select your start date for stock data collection.",
        format= "MM/DD/YYYY"
    )

    end_date = st.date_input(
        label = "Select your end date for stock data collection.",
        format= "MM/DD/YYYY"
    )

    rag_query = st.text_input(

        label = "Please specify second asset with which the dependency needs to be checked.",
        max_chars = 256,
        placeholder = "e.g., impact of interest rates on technology stocks..."
    )

    mkt_scenario = st.text_input(
        label = "Please input market scenario for analysis.", 
        max_chars = 256,
        placeholder = "e.g., a 20% drop in the S&P 500"
    )
    submit_button = st.form_submit_button("Submit")


if submit_button:

    # converting ticker input into a list
    ticker_list = [s.strip() for s in list(tickers.split(","))]

    with st.spinner("Processing your data now...."):

        # getting financial news data for the past 7 months
        financial_news = get_financial_news(news_query, end_date, end_date + datetime.timedelta(days=210), NEWS_API_KEY)

        # getting various prompts
        assets = tickers
        asset1 = ticker_list[0]
        asset2 = "interest rates"

        correlation_prompt, dependency_prompt, scenario_prompt = generate_prompts(rag_query, assets, asset1, asset2, mkt_scenario)
        
        # receive all models
        correlation_model, dependency_model, fine_tuned_correlation_model, fine_tuned_dependency_model = get_models(tickers, start_date, \
                                                                                                                    end_date, str(rag_query))
        # generating model responses 
        # correlation_response, dependency_response, scenario_response = get_responses(correlation_prompt, dependency_prompt, \
        #                                                                              scenario_prompt, assets, asset1, asset2, mkt_scenario)
        
        # correlation_chain, dependency_chain, scenario_chain = get_responses(correlation_prompt, dependency_prompt, \
        #                                                                              scenario_prompt, assets, asset1, asset2, mkt_scenario)

        # getting portfolio adjustment insights based on high-risk assets
        historical_data = get_historical_data(tickers, start_date, end_date)
        portfolio_rev_insights = adjust_portfolio(historical_data.corr(), fine_tuned_dependency_model, scenario_response, ticker_list)

    st.success("Data Processing Complete!")

    st.header("Historical Market Data")
    st.write(historical_data.head(10))

    st.header("Real-Time Financial News")
    for article in financial_news[:5]:
        st.subheader(article['title'])
        st.write(article['description'])
        st.write(f"Read more: {article['url']}")

    st.header("Display AI Generated Insights")
    
    st.subheader("The Correlation Prompt: ")
    st.write(correlation_prompt)
    st.subheader("The Dependency Prompt: ")
    st.write(dependency_prompt)
    st.subheader("The Scenario Prompt: ")
    st.write(scenario_prompt)

    # ======= Display ML Model Outputs =======

    ## correlation model coefficients
    st.write(f"Model Coefficients: {fine_tuned_correlation_model.coef_}")

    ## Correlation matrix and heatmap
    st.header("Correlation Matrix and Heatmap")
    st.plotly_chart(plot_correlation_matrix(historical_data))

    # ======= Showing different plots =======
    ## feature importance & dependency models 
    st.header("Feature Importances of Dependency Model")
    st.plotly_chart(plot_feature_importances(fine_tuned_dependency_model, ["Interest Rates", ticker_list[-1]]))

    ## residual plot
    new_data = get_historical_data(ticker_list, end_date, end_date + datetime.timedelta(days=210))
    X_new = new_data[ticker_list[-1]].values.reshape(-1, 1)
    y_new = new_data[ticker_list[0]].values
    st.header("Residual Plot of Dependency Model")
    st.plotly_chart(plot_residuals(fine_tuned_dependency_model, X_new, y_new))

    # Interpreting Scenario Analysis
    st.header("Scenario Analysis Interpretation")
    st.markdown("** Scenario Response is: ** " % scenario_response)
    st.write(scenario_response)

    # Dependency model feature importances
    st.subheader("Dependency Analysis")
    st.write(dependency_prompt)
    st.write(f"Model Feature Importances: {fine_tuned_dependency_model.feature_importances_}")

    # Portfolio Revision Insights
    st.header("Portfolio Adjustment Insights")
    for i in range(len(portfolio_rev_insights)):
        st.write(portfolio_rev_insights[i])
