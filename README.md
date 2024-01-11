# Project Overview

This project is a Python-based application that uses deep reinforcement learning to perform live trading and model training on ES emini futures data through Sierra Charts _(New to Sierra Charts and how to connect it to Python? Check out my articles here https://medium.com/@investinatech)_. Sierra Charts is an extremely powerful trading software that offers pinpoint data. The project is divided into several modules, each with a specific role in the overall functionality of the application.

## Project Structure

The project is organized into the following main directories:

- [`model_training`]: This directory contains the code for training machine learning models. It includes the `agents` module for training, `gym_envs` for the trading environment, and `preprocessing` for data preprocessing and feature engineering.

- [`live_trading`]: This directory contains the code for live trading. It includes the `trade_management` module for managing trades, `dataframe_pipeline` for processing data events into dataframes, and `events` for handling account and position events.

- [`sierracharts_data_downloader`]: This directory contains the `sc_data_downloader.py` script for downloading data from Sierra Charts.

The project also includes two main application files:

- [`model_training_app.py`]: This is the main entry point for the model training application.

- [`live_trading_app.py`]: This is the main entry point for the live trading application.

## Key Components

### Model Training

The model training process is handled by the [`run_training`] function in [`model_training/agents/training.py`]. This function uses the Proximal Policy Optimization (PPO) algorithm from the Stable Baselines3 library to train the model. The training data is preprocessed and feature engineered using the `DataFrameProcessor` and [`FeatureEngineering`] classes in [`model_training/preprocessing/dataframe_processor.py`] and [`model_training/preprocessing/feature_engineering.py`] respectively.

### Live Trading

The live trading process is handled by the `run_live_trading` function in [`live_trading/trade_management/run_live_trading.py`]. This function uses the trained model to make trading decisions. The trading data is processed into a dataframe using the [`DataEvent`], [`FootprintDataframe`], and [`MainDataframe`] classes in [`live_trading/dataframe_pipeline/data_event.py`], [`live_trading/dataframe_pipeline/footprint_dataframe.py`], and [`live_trading/dataframe_pipeline/main_dataframe.py`] respectively. The `AccountEvent` and `PositionEvent` classes in [`live_trading/events/account_event.py`] and [`live_trading/events/position_event.py`] are used to handle account and position events.

## Running the Applications

Before running the applications, ensure the following steps are completed:

1. **Download Sierra Chartbook**: I provided the necessary chartbook file [`SiarraCharts_Chartbook.Cht`] for Sierra Charts.

2. **Set Up Virtual Environment**: Create a virtual environment (venv) to manage Python dependencies.
    - Use `python -m venv .venv` to create a new venv.
    - Use `.venv\Scripts\activate` to activate the venv.

3. **Install Dependencies**: Install the project's dependencies using `pip install -r requirements.txt`.

4. **Environment Configuration**:
    - Create a `.env` file to store your configuration and sensitive data securely.
5. **Data Preparation**:
    - Run the Sierra Chart (SC) data downloader: Execute the `sc_data_downloader.py` script from the [`sierracharts_data_downloader`] directory to fetch necessary trading data.
6. **Model Training**:
    - Run the model training application: Execute the [`model_training_app.py`] script located in the root of the project directory.
7. **Live Trading**:
    - Run the live trading application: Execute the [`live_trading_app.py`] script for live trading operations.

Ensure each step is successfully completed before proceeding to the next. For detailed instructions on each step, refer to the respective sections of this documentation.


## Dependencies

The project's dependencies are listed in the [`requirements.txt`] file. To install these dependencies, run `pip install -r requirements.txt`.

## Environment Variables

The project uses environment variables to store sensitive information such as API keys. These variables are loaded from a [`.env`] file at runtime using the python-dotenv library.

## Data

The project uses data from Sierra Charts, which is downloaded using the `sc_data_downloader.py` script in the [`sierracharts_data_downloader`] directory.

## Ignored Files

The [`.gitignore`] file lists the files and directories that are ignored by Git. This includes Python cache files, environment variable files, log files, data files, Jupyter Notebook checkpoints, and IDE-specific files.

## Conclusion

This project is a comprehensive application that uses deep reinforcement learning for live trading. It demonstrates the use of various Python libraries and techniques, including data preprocessing, feature engineering, deep reinforcement learning, and live trading.