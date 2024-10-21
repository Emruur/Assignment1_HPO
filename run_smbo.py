import pandas as pd
import random
from ConfigSpace import ConfigurationSpace, read_and_write
from smbo import SequentialModelBasedOptimization
from typing import List, Tuple, Dict
from surrogate_model import SurrogateModel
import argparse
import matplotlib.pyplot as plt
import numpy as np


def read_files(dataset_path: str, config_path: str):
    # Read the dataset containing configurations and performance
    dataset = pd.read_csv(dataset_path)
    
    # Read the configuration space definition (JSON format assumed for ConfigSpace)
    config_space= ConfigurationSpace.from_json(config_path)
    
    return dataset, config_space


def create_capital_phi(model, config_space, anchor_size,n_samples: int = 100) -> List[Tuple[Dict, float]]:
    """
    Randomly get n_samples from the configuration space in a batch, run the surrogate model 
    to predict their performance individually, and return a list of tuples with (config_dict, performance).
    
    :param model: The surrogate model used for performance prediction.
    :param config_space: The configuration space to sample from.
    :param n_samples: Number of random samples to generate.
    :return: A list of tuples where each tuple contains a configuration dictionary and its performance.
    """
    capital_phi = []

    # Sample n_samples configurations as a batch
    configurations = config_space.sample_configuration(n_samples)

    # Predict performance for each configuration
    for conf in configurations:
        # Convert the configuration to a dictionary
        config_dict = dict(conf)
        ## TODO ANCHOR SIZE
        # Predict the performance of the configuration using the surrogate model
        config_dict["anchor_size"]= anchor_size
        config_performance = model.predict(config_dict)
        
        # Append the configuration and its performance as a tuple to capital_phi
        capital_phi.append((config_dict, config_performance))
    
    return capital_phi

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
    
    # Sample N=100 configurations and create capital_phi
    #TODO SORU: increasing n_smaples makes stuff weird? 
    all_perf_scores= []
    results = {
            'random_search': [1.0]
    }
    for i in range(10):

        capital_phi = create_capital_phi(surrogate_model,config_space,args.max_anchor_size, n_samples=25)

        # Initialize and fit the SMBO
        smbo = SequentialModelBasedOptimization(config_space, args.max_anchor_size)

        smbo.initialize(capital_phi)
        smbo.fit_model()

        performance_scores = []
        
        for _ in range(args.num_iterations):
            # Get the config with most improvement(hopefully)
            promising_configuration= smbo.select_configuration()
            promising_configuration_dict= dict(promising_configuration)
            promising_configuration_dict["anchor_size"]= args.max_anchor_size

            # "Train" the config and get the score
            config_performance= surrogate_model.predict(promising_configuration_dict)
            if i == 0:
                results['random_search'].append(min(results['random_search'][-1], config_performance))  
            else:
                performance_scores.append(config_performance)
            smbo.update_runs((promising_configuration, config_performance))
            print("DEBUG: ")
            print(config_performance)
            print(len(smbo.R))

        if i!= 0:
            all_perf_scores.append(performance_scores)

    # Convert the list of performance scores to a NumPy array
    perf_scores_array = np.array(all_perf_scores)

    # Calculate the mean and variance across runs (axis 0 corresponds to the different runs)
    mean_scores = np.mean(perf_scores_array, axis=0)
    std_scores = np.std(perf_scores_array, axis=0)

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot the mean scores
    plt.plot(mean_scores, label='Mean Performance')

    # Plot the variance as a shaded area
    plt.fill_between(range(len(mean_scores)),
                    mean_scores - std_scores, 
                    mean_scores + std_scores, 
                    color='gray', alpha=0.2, label='Variance')

    # Labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Performance Score')
    plt.title('Performance over Iterations')
    plt.legend()

    # Show the plot
    plt.show()

    ## PART 2 compare with random search
    print("Comparing with random search")

    plt.plot(range(len(results['random_search']) - 1), results['random_search'][1:])
    plt.yscale('log')
    plt.show()


# Run the main function
if __name__ == "__main__":
    main(parse_args())
