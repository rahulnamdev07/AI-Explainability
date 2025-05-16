import pandas as pd
import numpy as np
from prophet import Prophet
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool, AgentType
import shap
from sklearn.linear_model import LinearRegression

# 1. Generate synthetic Indian airlines data with holiday effects
def generate_airline_data():
    np.random.seed(42)
    date_range = pd.date_range(start='2020-01-01', end='2024-12-31', freq='MS')
    base_passengers = np.linspace(1000, 5000, len(date_range)) + np.random.normal(0, 200, len(date_range))
    holiday_boost = np.zeros(len(date_range))
    for i, date in enumerate(date_range):
        if date.month in [10, 11]:  # Diwali (typically Oct or Nov)
            holiday_boost[i] = 500 + np.random.normal(0, 50)
        elif date.month == 8:       # Raksha Bandhan (Aug)
            holiday_boost[i] = 300 + np.random.normal(0, 30)
    df = pd.DataFrame({
        'ds': date_range,
        'y': base_passengers + holiday_boost,
        'holiday_effect': holiday_boost
    })
    df['month'] = df['ds'].dt.month
    return df

# 2. Prophet forecasting and decomposition
def create_forecast(data):
    model = Prophet(yearly_seasonality=True)
    model.add_regressor('holiday_effect')
    model.add_regressor('month')
    model.fit(data)
    future = model.make_future_dataframe(periods=12, freq='MS')
    # Add regressors for the future
    future['holiday_effect'] = 0
    for i, date in enumerate(future['ds']):
        if date.month in [10, 11]:
            future.loc[i, 'holiday_effect'] = 500
        elif date.month == 8:
            future.loc[i, 'holiday_effect'] = 300
        future.loc[i, 'month'] = date.month
    forecast = model.predict(future)
    return model, forecast, future

# 3. Tool for Prophet decomposition
def prophet_decomposition():
    trend = forecast[['ds', 'trend']].tail(12).to_dict(orient='records')
    seasonality = forecast[['ds', 'yearly']].tail(12).to_dict(orient='records')
    return {"trend": trend, "seasonality": seasonality}

# 4. Tool for SHAP explanation
def prophet_shap():
    reg_cols = ['holiday_effect', 'month']
    X = future[reg_cols]
    y = model.predict(future)['yhat']
    linreg = LinearRegression().fit(X, y)
    import shap
    explainer = shap.Explainer(linreg, X)
    shap_values = explainer(X)
    last_12 = shap_values[-12:]
    mean_abs_shap = np.abs(last_12.values).mean(axis=0)
    feature_impact = dict(zip(reg_cols, mean_abs_shap))
    return feature_impact

# 5. Prompt for planning and explanation
PLANNING_PROMPT = """
You are a helpful agent that explains forecasts from Prophet models for Indian airline passenger data.
You have access to two tools:
- 'prophet_decomposition': returns trend and seasonality for the next 12 months.
- 'prophet_shap': returns the average SHAP value (feature impact) for each regressor over the forecast period.

First, output a plan for how you will explain the forecast, starting with 'Plan:' and a numbered list of steps.
Then, for each forecasted month, output a table row with:
Month Name | Forecast Value | Explanation (trend + seasonality + holiday effects + SHAP feature impact)

Consider that air traffic increases during Indian holidays like Diwali (Oct/Nov) and Raksha Bandhan (Aug).
Include SHAP-based insights on which variables most influenced the forecast.
"""

# 6. Main execution
if __name__ == "__main__":
    # Generate data and fit Prophet
    airline_data = generate_airline_data()
    model, forecast, future = create_forecast(airline_data[['ds', 'y', 'holiday_effect', 'month']])

    # Set up LLM and tools
    llm = Ollama(model="mistral")
    # Use closure to pass model, forecast, future to tools
    tools = [
        Tool(
            name="prophet_decomposition",
            description="Decompose the forecast into trend and seasonality for the next 12 months.",
            func=prophet_decomposition,
        ),
        Tool(
            name="prophet_shap",
            description="Compute the average SHAP value (feature impact) for each regressor over the forecast period.",
            func=prophet_shap,
        ),
    ]

    # Create agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    input_query = (
        PLANNING_PROMPT +
        "\nExplain the forecast for the next 12 months, highlighting effects during Diwali and Raksha Bandhan."
        " The Prophet model, forecast, and future dataframe are provided to the tools."
    )

    result = agent({"input": input_query})

    print("\n--- AGENT OUTPUT ---\n")
    print(result["output"])
