from datetime import datetime
from os import environ

import streamlit as st

from financegpt.app import AppController
from financegpt.app import RequestModel

ALLOWED_INTERVALS = ("W", "D", "H1")
DEFAULT_WINDOW_SIZE = 5
DEFAULT_LLM_MODEL = "gpt-3.5-turbo-1106"

connection_kwargs = {
    "username": environ["FINGPT_DB_USERNAME"],
    "password": environ["FINGPT_DB_PASSWORD"],
    "host": environ["FINGPT_DB_HOST"],
    "port": int(environ["FINGPT_DB_PORT"]),
    "db_name": environ["FINGPT_DB_NAME"],
}

controller = AppController(
    llm_model=DEFAULT_LLM_MODEL,
    connection_kwargs=connection_kwargs,
    window_size=DEFAULT_WINDOW_SIZE,
)

st.title("FinanceGPT")
# UI Controls based on RequestModel
user_msg = st.text_area("User Message")
historical_data_start_date = st.date_input(
    "Historical Data Start Date", datetime.today()
)
historical_data_end_date = st.date_input("Historical Data End Date", datetime.today())
historical_data_interval = st.selectbox("Historical Data Interval", ALLOWED_INTERVALS)
prediction_symbol = st.text_input("Prediction Symbol")
prediction_end_date = st.date_input("Prediction End Date", datetime.today())

if st.button("Submit"):
    # Create a request model instance
    assert historical_data_interval in ALLOWED_INTERVALS
    request_model = RequestModel(
        user_msg=user_msg,
        historical_data_start_date=historical_data_start_date,
        historical_data_end_date=historical_data_end_date,
        historical_data_interval=historical_data_interval,
        prediction_symbol=prediction_symbol,
        prediction_end_date=prediction_end_date,
    )

    try:
        response = controller.process_request(request_model)
        st.success("Response from model:")
        st.write(response)
    except Exception as e:
        st.error(f"Error processing request: {e}")
