# Project Overview

This project is a Python-based application that uses machine learning to perform live trading and model training on ES emini futures data through Sierra Charts _(New to Sierra Charts and how to connect it to Python? Check out my articles here https://medium.com/@investinatech)_. Sierra Charts is an extremely powerful trading software that offers pinpoint data. The project is divided into several modules, each with a specific role in the overall functionality of the application.

## Project Structure

The project is organized into the following main directories:

- [`model_training`](command:_github.copilot.openRelativePath?%5B%22model_training%22%5D "model_training"): This directory contains the code for training machine learning models. It includes the `agents` module for training, `gym_envs` for the trading environment, and `preprocessing` for data preprocessing and feature engineering.

- [`live_trading`](command:_github.copilot.openRelativePath?%5B%22live_trading%22%5D "live_trading"): This directory contains the code for live trading. It includes the `trade_management` module for managing trades, `dataframe_pipeline` for processing data events into dataframes, and `events` for handling account and position events.

- [`sierracharts_data_downloader`](command:_github.copilot.openRelativePath?%5B%22sierracharts_data_downloader%22%5D "sierracharts_data_downloader"): This directory contains the `sc_data_downloader.py` script for downloading data from Sierra Charts.

The project also includes two main application files:

- [`model_training_app.py`](command:_github.copilot.openRelativePath?%5B%22model_training_app.py%22%5D "model_training_app.py"): This is the main entry point for the model training application.

- [`live_trading_app.py`](command:_github.copilot.openRelativePath?%5B%22live_trading_app.py%22%5D "live_trading_app.py"): This is the main entry point for the live trading application.

## Key Components

### Model Training

The model training process is handled by the [`run_training`](command:_github.copilot.openSymbolInFile?%5B%22model_training%2Fagents%2Ftraining.py%22%2C%22run_training%22%5D "model_training/agents/training.py") function in [`model_training/agents/training.py`](command:_github.copilot.openSymbolInFile?%5B%22model_training%2Fagents%2Ftraining.py%22%2C%22model_training%2Fagents%2Ftraining.py%22%5D "model_training/agents/training.py"). This function uses the Proximal Policy Optimization (PPO) algorithm from the Stable Baselines3 library to train the model. The training data is preprocessed and feature engineered using the `DataFrameProcessor` and [`FeatureEngineering`](command:_github.copilot.openSymbolInFile?%5B%22model_training%2Fpreprocessing%2Ffeature_engineering.py%22%2C%22FeatureEngineering%22%5D "model_training/preprocessing/feature_engineering.py") classes in [`model_training/preprocessing/dataframe_processor.py`](command:_github.copilot.openSymbolInFile?%5B%22model_training%2Fpreprocessing%2Fdataframe_processor.py%22%2C%22model_training%2Fpreprocessing%2Fdataframe_processor.py%22%5D "model_training/preprocessing/dataframe_processor.py") and [`model_training/preprocessing/feature_engineering.py`](command:_github.copilot.openSymbolInFile?%5B%22model_training%2Fpreprocessing%2Ffeature_engineering.py%22%2C%22model_training%2Fpreprocessing%2Ffeature_engineering.py%22%5D "model_training/preprocessing/feature_engineering.py") respectively.

### Live Trading

The live trading process is handled by the `run_live_trading` function in [`live_trading/trade_management/run_live_trading.py`](command:_github.copilot.openSymbolInFile?%5B%22live_trading%2Ftrade_management%2Frun_live_trading.py%22%2C%22live_trading%2Ftrade_management%2Frun_live_trading.py%22%5D "live_trading/trade_management/run_live_trading.py"). This function uses the trained model to make trading decisions. The trading data is processed into a dataframe using the [`DataEvent`](command:_github.copilot.openSymbolInFile?%5B%22live_trading%2Fdataframe_pipeline%2Fdata_event.py%22%2C%22DataEvent%22%5D "live_trading/dataframe_pipeline/data_event.py"), [`FootprintDataframe`](command:_github.copilot.openSymbolInFile?%5B%22live_trading%2Fdataframe_pipeline%2Ffootprint_dataframe.py%22%2C%22FootprintDataframe%22%5D "live_trading/dataframe_pipeline/footprint_dataframe.py"), and [`MainDataframe`](command:_github.copilot.openSymbolInFile?%5B%22live_trading%2Fdataframe_pipeline%2Fmain_dataframe.py%22%2C%22MainDataframe%22%5D "live_trading/dataframe_pipeline/main_dataframe.py") classes in [`live_trading/dataframe_pipeline/data_event.py`](command:_github.copilot.openSymbolInFile?%5B%22live_trading%2Fdataframe_pipeline%2Fdata_event.py%22%2C%22live_trading%2Fdataframe_pipeline%2Fdata_event.py%22%5D "live_trading/dataframe_pipeline/data_event.py"), [`live_trading/dataframe_pipeline/footprint_dataframe.py`](command:_github.copilot.openSymbolInFile?%5B%22live_trading%2Fdataframe_pipeline%2Ffootprint_dataframe.py%22%2C%22live_trading%2Fdataframe_pipeline%2Ffootprint_dataframe.py%22%5D "live_trading/dataframe_pipeline/footprint_dataframe.py"), and [`live_trading/dataframe_pipeline/main_dataframe.py`](command:_github.copilot.openSymbolInFile?%5B%22live_trading%2Fdataframe_pipeline%2Fmain_dataframe.py%22%2C%22live_trading%2Fdataframe_pipeline%2Fmain_dataframe.py%22%5D "live_trading/dataframe_pipeline/main_dataframe.py") respectively. The `AccountEvent` and `PositionEvent` classes in [`live_trading/events/account_event.py`](command:_github.copilot.openSymbolInFile?%5B%22live_trading%2Fevents%2Faccount_event.py%22%2C%22live_trading%2Fevents%2Faccount_event.py%22%5D "live_trading/events/account_event.py") and [`live_trading/events/position_event.py`](command:_github.copilot.openSymbolInFile?%5B%22live_trading%2Fevents%2Fposition_event.py%22%2C%22live_trading%2Fevents%2Fposition_event.py%22%5D "live_trading/events/position_event.py") are used to handle account and position events.

## Running the Applications

To run the model training application, execute the [`model_training_app.py`](command:_github.copilot.openRelativePath?%5B%22model_training_app.py%22%5D "model_training_app.py") script. To run the live trading application, execute the [`live_trading_app.py`](command:_github.copilot.openRelativePath?%5B%22live_trading_app.py%22%5D "live_trading_app.py") script.

## Dependencies

The project's dependencies are listed in the [`requirements.txt`](command:_github.copilot.openRelativePath?%5B%22requirements.txt%22%5D "requirements.txt") file. To install these dependencies, run `pip install -r requirements.txt`.

## Environment Variables

The project uses environment variables to store sensitive information such as API keys. These variables are loaded from a [`.env`](command:_github.copilot.openRelativePath?%5B%22.env%22%5D ".env") file at runtime using the python-dotenv library.

## Data

The project uses data from Sierra Charts, which is downloaded using the `sc_data_downloader.py` script in the [`sierracharts_data_downloader`](command:_github.copilot.openRelativePath?%5B%22sierracharts_data_downloader%22%5D "sierracharts_data_downloader") directory.

## Ignored Files

The [`.gitignore`](command:_github.copilot.openRelativePath?%5B%22.gitignore%22%5D ".gitignore") file lists the files and directories that are ignored by Git. This includes Python cache files, environment variable files, log files, data files, Jupyter Notebook checkpoints, and IDE-specific files.

## Conclusion

This project is a comprehensive application that uses machine learning for live trading. It demonstrates the use of various Python libraries and techniques, including data preprocessing, feature engineering, machine learning, and live trading.