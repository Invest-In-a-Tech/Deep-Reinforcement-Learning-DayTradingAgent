################################################################
# IMPORTS 
# ==============================================================
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sklearn.model_selection import KFold
from model_training.preprocessing.dataframe_processor import DataFrameProcessor
from model_training.preprocessing.feature_engineering import FeatureEngineering
from model_training.gym_envs.trading_env import TradingEnv
import os




################################################################
# run_training
# ==============================================================
def run_training():
    """
    Executes the training pipeline for a financial trading model using the ES dataset. This comprehensive 
    process includes data preprocessing, feature engineering, time series cross-validation, and training 
    a model using Proximal Policy Optimization (PPO) from the Stable Baselines3 library. 

    The pipeline steps are as follows:
    1. Preprocess the data using `DataFrameProcessor` to clean and prepare the dataset.
    2. Apply feature engineering to enhance the dataset with relevant financial indicators.
    3. Perform time series cross-validation using `TimeSeriesSplit` to partition the data for training and testing.
    4. Scale features using `MinMaxScaler` for normalization.
    5. Train a PPO model on the training set for each fold, saving model checkpoints and scalers.
    6. Evaluate the model on the test set, collecting total rewards as evaluation metrics.
    7. Save the trained model and scaler for each fold for future use or analysis.

    The function logs training details, scales data for model input, and saves the trained models and scalers. 
    Evaluation results are printed at the end, providing insights into the model's performance across different folds.
    """
    processor = DataFrameProcessor(os.path.join('model_training', 'data', 'example_esDataset.csv'))
    df = processor.process_data()
    feature_engineer = FeatureEngineering(df)
    df_enhanced = feature_engineer.perform_feature_engineering()
    n_splits = 2 # Number of splits for Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=n_splits) # TimeSeriesSplit
    evaluation_results = []  # Store evaluation metrics for each fold 
    



    ############################################################
    # Loop through each fold of the TimeSeriesSplit
    # ==========================================================
    for fold_num, (train_index, test_index) in enumerate(tscv.split(df_enhanced), start=1):
        print(f"Fold {fold_num}:")
        
        # Define the log path for this fold
        log_path = os.path.join('model_training', 'training', 'logs', f'fold_{fold_num}')
        os.makedirs(log_path, exist_ok=True)
        print(f"Log path for fold {fold_num}: {log_path}")       
            
        # Splitting the data
        features_train, features_test = df_enhanced.iloc[train_index], df_enhanced.iloc[test_index]
      
        # Scaling the features
        scaler = MinMaxScaler()
        features_train_scaled = scaler.fit_transform(features_train)
        features_test_scaled = scaler.transform(features_test)
        
        # Convert the scaled data into a DataFrame
        features_train_scaled = pd.DataFrame(features_train_scaled, columns=features_train.columns, index=features_train.index)
        features_test_scaled = pd.DataFrame(features_test_scaled, columns=features_test.columns, index=features_test.index)
        print(f"After scaling, training data shape: {features_train_scaled.shape}, test data shape: {features_test_scaled.shape}")

        # Initialize and train your model
        #env_train = TradingEnv(features_train_scaled, start_time="08:30:00", end_time="14:30:00")
        env_train = TradingEnv(features_train, start_time="08:30:00", end_time="14:30:00")
        model = PPO("MlpPolicy", env_train, verbose=1, tensorboard_log=log_path)
        model.learn(total_timesteps=500)

        
        # Evaluate your model on the test set
        #env_test = TradingEnv(features_test_scaled, start_time="08:30:00", end_time="14:30:00")
        env_test = TradingEnv(features_test, start_time="08:30:00", end_time="14:30:00")
        obs = env_test.reset()
        total_rewards = 0
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env_test.step(action)
            total_rewards += reward
            env_test.render(action=action, reward=reward)
            
            

        # Store evaluation results
        evaluation_results.append(total_rewards)       
        
        # Define the path to save the model
        PPO_path = os.path.join('model_training', 'training', 'saved_models', f'PPO_fold_{fold_num}')
        
        # Save the mode
        model.save(PPO_path)    

        # Create the directory for the scaler if it doesn't exist
        scaler_directory = os.path.join('model_training', 'training', 'saved_scalers')
        os.makedirs(scaler_directory, exist_ok=True)
        
        # Saving the scaler
        scaler_filename = os.path.join(scaler_directory, f'scaler_fold_{fold_num}')
        joblib.dump(scaler, scaler_filename)

    # Analysis of evaluation results across all folds
    print("Evaluation Results:", evaluation_results)