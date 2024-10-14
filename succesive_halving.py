import pandas as pd
import random
from ConfigSpace import ConfigurationSpace, read_and_write
from smbo import SequentialModelBasedOptimization
from typing import List, Tuple, Dict
from surrogate_model import SurrogateModel
import argparse
import matplotlib.pyplot as plt
import math


def read_files(dataset_path: str, config_path: str):
    # Read the dataset containing configurations and performance
    dataset = pd.read_csv(dataset_path)
    
    # Read the configuration space definition (JSON format assumed for ConfigSpace)
    config_space= ConfigurationSpace.from_json(config_path)
    
    return dataset, config_space



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=25)

    return parser.parse_args()

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

    B= 100000
    arms= 1000
    S= config_space.sample_configuration(arms)
    S= [dict(conf) for conf in S]

    # Initialize variables
    halving_steps = math.ceil(math.log2(arms))
    bandit_performance = {i: [] for i in range(len(S))}  # Track performance of each bandit

    for i in range(halving_steps):
        budget = B / (len(S) * halving_steps)    
        [bandit.update({"anchor_size": budget}) for bandit in S]
        [bandit.update({"score": surrogate_model.predict(bandit)}) for bandit in S]
        
        for idx, bandit in enumerate(S):
            bandit_performance[idx].append(bandit["score"])
        
        S = sorted(S, key=lambda x: x["score"])
        print("-----------------",i)
        for i,s in enumerate(S):
            print(f"{i}: {s['score']}")

        S = S[:math.ceil(len(S) / 2)]
        for i,s in enumerate(S):
            print(f"{i}: {s['score']}")

    # Plot the performance of each bandit over the halving steps
    for bandit_id, scores in bandit_performance.items():
        plt.plot(range(len(scores)), scores, marker='o', label=f'Bandit {bandit_id}')

    plt.xlabel('Halving Step')
    plt.ylabel('Score')
    plt.title('Performance of Each Bandit in Successive Halving')
    plt.grid(True)
    plt.legend()
    plt.show()









        



# Run the main function
if __name__ == "__main__":
    main(parse_args())
