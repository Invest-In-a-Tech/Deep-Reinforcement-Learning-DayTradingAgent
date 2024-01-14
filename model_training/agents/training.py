from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import PPO
from sklearn.model_selection import KFold
from model_training.preprocessing.dataframe_processor import DataFrameProcessor
from model_training.preprocessing.feature_engineering import FeatureEngineering
from model_training.gym_envs.trading_env import TradingEnv
import os


def run_training():
    processor = DataFrameProcessor(os.path.join('model_training', 'data', 'example_esDataset.csv'))
    df = processor.process_data()

    feature_engineer = FeatureEngineering(df)
    df_enhanced = feature_engineer.perform_feature_engineering()

    features = df_enhanced

    seed = 42
    kf = KFold(n_splits=2, shuffle=True, random_state=seed)
    for fold_num, (train_index, test_index) in enumerate(kf.split(features), start=1):
        # Splitting the data
        features_train, features_test = features.iloc[train_index], features.iloc[test_index]

        # Scaling the features
        scaler = MinMaxScaler()
        features_train_scaled = scaler.fit_transform(features_train)
        features_test_scaled = scaler.transform(features_test)
        features_train_scaled = pd.DataFrame(features_train_scaled, columns=features_train.columns, index=features_train.index)
        features_test_scaled = pd.DataFrame(features_test_scaled, columns=features_test.columns, index=features_test.index)
        print(f"After scaling, training data shape: {features_train_scaled.shape}, test data shape: {features_test_scaled.shape}")

        log_path = os.path.join('model_training', 'training', 'logs')
        print(log_path) 
          
        # Create the directory for the scaler if it doesn't exist
        scaler_directory = os.path.join('model_training', 'training', 'saved_scalers')
        os.makedirs(scaler_directory, exist_ok=True)

        # Saving the scaler
        scaler_filename = os.path.join(scaler_directory, f'scaler_fold_{fold_num}')
        joblib.dump(scaler, scaler_filename)


        # Setting up the environment and model
        env = TradingEnv(features_train_scaled)
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
        model.learn(total_timesteps=20000)

        # Define the path to save the model
        PPO_path = os.path.join('model_training', 'training', 'saved_models', f'PPO_fold_{fold_num}')
        
        # Save the mode
        model.save(PPO_path)    

        # Resetting the environment for evaluation
        obs = env.reset()
        print(f"Testing model, initial observation shape: {obs.shape}")
    
        # Initializing variables for evaluation
        eval_rewards = []
        eval_actions = []
        total_rewards = 0
        i = 0

        # Evaluation loop
        while i < 1000:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            eval_rewards.append(reward)
            eval_actions.append(action)
            total_rewards += reward

            env.render(step_num=i, action=action, reward=reward)
            #env.log_data(step_num=i, action=action, reward=reward, total_rewards=total_rewards)

            if done:
                print('Episode ended')
                obs = env.reset()  # Resetting the environment
                total_rewards = 0  # Reset total rewards for the new episode
            else:
                i += 1

        # Calculate evaluation metrics
        total_eval_reward = np.sum(eval_rewards)
        mean_eval_reward = np.mean(eval_rewards)
        print(f'Total Evaluation Reward: {total_eval_reward}')
        print(f'Mean Evaluation Reward: {mean_eval_reward}')

        env.close()