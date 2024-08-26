# install dependencies
import os
import openai
import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import LLMChain 
from langchain.prompts import PromptTemplate

load_dotenv()

# importing API keys and define LLM
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_llm = OpenAI(model="text-davinci-003", temperature=0)

#invoking yahoo finance library and collect data of few stocks
def get_historical_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

# Getting financial news data 
def get_financial_news(query, from_date, to_date, api_key):
    url = f'https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=publishedAt&apiKey={api_key}'
    response = requests.get(url)
    news_data = response.json()
    articles = news_data['articles']
    return articles

# prompt design - define prompts for different analyses
def generate_prompts(query, assets, asset1, asset2, scenario):
   
    # Define a correlation prompt template
    correlation_prompt_template = PromptTemplate(
        input_variables=["assets"],
        template="Analyze the correlation between {assets} over the past five years."
    )

    # Define a dependency prompt template
    dependency_prompt_template = PromptTemplate(
        input_variables=["asset1", "asset2"],
        template="Examine the dependency of {asset1} on {asset2} during periods of high market volatility."
    )

    # Define a scenario prompt template
    scenario_analysis_prompt_template = PromptTemplate(
        input_variables=["scenario"],
        template="Generate a detailed scenario analysis for {scenario}."
    )

    return correlation_prompt_template, dependency_prompt_template, scenario_analysis_prompt_template


def get_responses(corr_template, depend_template, scenario_template, tickers, asset1, asset2, mkt_scenario):

    # create chains
    correlation_chain = LLMChain(llm=openai_llm, prompt=corr_template)
    dependency_chain = LLMChain(llm=openai_llm, prompt=depend_template)
    scenario_chain = LLMChain(llm=openai_llm, prompt=scenario_template)

    # correlation_response = correlation_chain.run({"assets": str(tickers)})
    # dependency_response = dependency_chain.run({"asset1": str(asset1), "asset2": str(asset2)})
    # scenario_response = scenario_chain.run({"scenario": str(mkt_scenario)})

    # return correlation_response, dependency_response, scenario_response
    return correlation_chain, dependency_chain, scenario_chain

