import pandas as pd
import random
from ConfigSpace import ConfigurationSpace, read_and_write
from smbo import SequentialModelBasedOptimization
from typing import List, Tuple, Dict
from surrogate_model import SurrogateModel
import argparse
import matplotlib.pyplot as plt
import math
import pickle

def save_experiment_data(data, filename="experiment_data.pkl"):
    """Save experiment data to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_experiment_data(filename="experiment_data.pkl"):
    """Load experiment data from a pickle file."""

    with open(filename, 'rb') as f:
        return pickle.load(f)

def read_files(dataset_path: str, config_path: str):
    # Read the dataset containing configurations and performance
    dataset = pd.read_csv(dataset_path)
    
    # Read the configuration space definition (JSON format assumed for ConfigSpace)
    config_space= ConfigurationSpace.from_json(config_path)
    
    return dataset, config_space



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config-performances/config_performances_dataset-1457.csv')
    return parser.parse_args()

class SuccesiveHalving:
    def __init__(self, arms, budget, groundtruth,config_space,predefined_anchors= None) -> None:
        self.arms= arms
        self.budget= budget
        self.predefined_anchors= predefined_anchors
        self.groundtruth= groundtruth
        self.config_space= config_space
    def run(self):
        B= self.budget
        arms= self.arms
        S= self.config_space.sample_configuration(arms)
        S= [dict(conf) for conf in S]
        total_budget= 0

        # Initialize variables
        halving_steps = math.ceil(math.log2(arms))
        bandit_performance = {i: [] for i in range(len(S))}  
        best_score_so_far= [1]
        cumilative_budget= [0]

        for i in range(halving_steps):
            budget = B / (len(S) * halving_steps) 
            if self.predefined_anchors:   
                budget= self.predefined_anchors[i + 4]
            total_budget += budget* len(S)
            [bandit.update({"anchor_size": budget}) for bandit in S]
            [bandit.update({"score": self.groundtruth.predict(bandit)}) for bandit in S]

            best_score = max(bandit["score"] for bandit in S)
            best_score_so_far.append(min(best_score_so_far[-1], best_score))
            cumilative_budget.append(cumilative_budget[-1] + budget* len(S))
            
            for idx, bandit in enumerate(S):
                bandit_performance[idx].append(bandit["score"])
            #print(f"Halving step {i} budget: {budget}")
            
            S = sorted(S, key=lambda x: x["score"])
            S = S[:math.ceil(len(S) / 2)]

        best_score_so_far.pop(0)
        cumilative_budget.pop(0)

        return bandit_performance, best_score_so_far, cumilative_budget




# Main function to run the steps
def main(args):
    # File paths for dataset and config space (replace with correct paths)
    dataset_path = args.configurations_performance_file
    config_path = args.config_space_file

    # Read the files
    dataset, config_space = read_files(dataset_path, config_path)

    # Train the groundtruth surrogate model
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(dataset)
    
    # Budget

    B= 120000
    arms= 423
    S= config_space.sample_configuration(arms)
    S= [dict(conf) for conf in S]
    anchors= [16, 23, 32, 45, 64, 91, 128, 181, 256, 362, 512, 724, 1024, 1200]
    
    all_experiment_data = []
    for i in range(500):
        sh_instance= SuccesiveHalving(arms, B, surrogate_model, config_space,predefined_anchors=anchors)
        bandit_performance, best_so_far, cumulative_budget= sh_instance.run()
        ## write all experiment data to a pkl file a
        experiment_data = {
            "bandit_performance": bandit_performance,
            "best_so_far": best_so_far,
            "cumulative_budget": cumulative_budget
        }
        print(f"Experiment {i} finished.")
        all_experiment_data.append(experiment_data)


    save_experiment_data(all_experiment_data, filename="sh_500_pre_anchors")

    exit()

    # Plot the performance of each bandit over the halving steps
    for bandit_id, scores in bandit_performance.items():
        plt.plot(range(len(scores)), scores, marker='o', label=f'Bandit {bandit_id}')
        
    
    


    plt.xlabel('Halving Step')
    plt.ylabel('Score')
    plt.title('Performance of Each Bandit in Successive Halving')
    plt.grid(True)
    plt.show()


    # Plotting Budget Spent vs. Best Score Found So Far
    plt.plot(cumulative_budget, best_so_far, marker='o')
    plt.xlabel("Cumulative Budget Spent")
    plt.ylabel("Best Score Found So Far")
    plt.title("Budget Spent vs. Best Score Found So Far")
    plt.grid(True)
    plt.show()







# Run the main function
if __name__ == "__main__":
    main(parse_args())
