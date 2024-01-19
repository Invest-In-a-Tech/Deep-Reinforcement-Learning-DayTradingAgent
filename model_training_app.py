# main.py 


from model_training.agents.ppo_train_test_split import run_training
#from model_training.agents.test_code import run_training

def main():
    
    # Call the training function or class method from training.py here
    run_training()
    #run_testing()

if __name__ == "__main__":
    main()
    