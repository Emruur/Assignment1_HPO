import pandas as pd
import random
from ConfigSpace import ConfigurationSpace, read_and_write
from smbo import SequentialModelBasedOptimization
from typing import List, Tuple, Dict
from surrogate_model import SurrogateModel
import argparse
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from random_search import RandomSearch
import warnings
from sklearn.exceptions import ConvergenceWarning
import pickle
import os

# Suppress specific warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Suppress lbfgs optimization warnings
warnings.filterwarnings("ignore", message="lbfgs failed to converge")
pd.set_option('future.no_silent_downcasting', True)

def save_results(file_path, results):
    """
    Save the results to a file using pickle.
    
    :param file_path: Path to the file where results will be saved.
    :param results: The results to be saved (e.g., smbo_results or rs_results).
    """
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)

def load_results(file_path):
    """
    Load the results from a file if they exist.
    
    :param file_path: Path to the file where results are saved.
    :return: The loaded results if the file exists, otherwise None.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    return None

def random_search_experiment(args, config_space,groundtruth, n_runs= 10) -> list[list[float]]:
    random_search = RandomSearch(config_space)
    run_results= []
    for _ in range(n_runs):
        run_result= []
        for idx in range(args.num_iterations):
            theta_new = dict(random_search.select_configuration())
            # Check if any value in theta_new is None
            theta_new['anchor_size'] = args.max_anchor_size
            performance = groundtruth.predict(theta_new)
            run_result.append(performance)
            random_search.update_runs((theta_new, performance))  

        run_results.append(run_result)
    return run_results

def smbo_experiment(args, config_space,groundtruth, n_runs= 10) -> list[list[float]]:
    all_perf_scores= []
    for _ in range(n_runs):

        capital_phi = create_capital_phi(groundtruth,config_space,args.max_anchor_size, n_samples=25)

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
            config_performance= groundtruth.predict(promising_configuration_dict)
            performance_scores.append(config_performance)
            smbo.update_runs((promising_configuration, config_performance))

        all_perf_scores.append(performance_scores)
    return all_perf_scores


def compare_smbo_and_random_search(smbo_scores: list[list[float]], random_search_scores: list[list[float]]):
    """
    Compare SMBO and Random Search by plotting:
    1. Mean and variance of the performance across multiple runs.
    2. Mean with min and max bounds across multiple runs.
    3. Mean best-so-far performance across all runs.
    4. Best-so-far among three randomly selected runs from SMBO and Random Search.

    :param smbo_scores: A list of lists where each inner list represents the performance scores for SMBO over iterations.
    :param random_search_scores: A list of lists where each inner list represents the performance scores for Random Search over iterations.
    """
    # Convert the lists of performance scores to NumPy arrays for easier manipulation
    smbo_scores_array = np.array(smbo_scores)
    random_search_scores_array = np.array(random_search_scores)

    #### Plot 1: Mean and variance comparison ####
    smbo_mean_scores = np.mean(smbo_scores_array, axis=0)
    smbo_std_scores = np.std(smbo_scores_array, axis=0)

    random_mean_scores = np.mean(random_search_scores_array, axis=0)
    random_std_scores = np.std(random_search_scores_array, axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(smbo_mean_scores, label='SMBO Mean Performance', color='blue')
    plt.fill_between(range(len(smbo_mean_scores)),
                     smbo_mean_scores - smbo_std_scores,
                     smbo_mean_scores + smbo_std_scores,
                     color='blue', alpha=0.2, label='SMBO Variance')

    plt.plot(random_mean_scores, label='Random Search Mean Performance', color='green')
    plt.fill_between(range(len(random_mean_scores)),
                     random_mean_scores - random_std_scores,
                     random_mean_scores + random_std_scores,
                     color='green', alpha=0.2, label='Random Search Variance')

    plt.xlabel('Iteration')
    plt.ylabel('Performance Score')
    plt.title('SMBO vs Random Search: Mean and Variance Comparison')
    plt.legend()
    plt.show()

    #### Plot 2: Mean with Min and Max bounds ####
    smbo_min_scores = np.min(smbo_scores_array, axis=0)
    smbo_max_scores = np.max(smbo_scores_array, axis=0)

    random_min_scores = np.min(random_search_scores_array, axis=0)
    random_max_scores = np.max(random_search_scores_array, axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(smbo_mean_scores, label='SMBO Mean Performance', color='blue')
    plt.fill_between(range(len(smbo_mean_scores)),
                     smbo_min_scores,
                     smbo_max_scores,
                     color='blue', alpha=0.2, label='SMBO Min-Max')

    plt.plot(random_mean_scores, label='Random Search Mean Performance', color='green')
    plt.fill_between(range(len(random_mean_scores)),
                     random_min_scores,
                     random_max_scores,
                     color='green', alpha=0.2, label='Random Search Min-Max')

    plt.xlabel('Iteration')
    plt.ylabel('Performance Score')
    plt.title('SMBO vs Random Search: Mean with Min-Max Bounds')
    plt.legend()
    plt.show()

    #### Plot 3: Mean Best-So-Far Performance ####
    smbo_best_so_far_all_runs = np.minimum.accumulate(smbo_scores_array, axis=1)
    random_search_best_so_far_all_runs = np.minimum.accumulate(random_search_scores_array, axis=1)

    smbo_mean_best_so_far = np.mean(smbo_best_so_far_all_runs, axis=0)
    random_search_mean_best_so_far = np.mean(random_search_best_so_far_all_runs, axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(smbo_mean_best_so_far, label='SMBO Mean Best-So-Far', color='blue')
    plt.plot(random_search_mean_best_so_far, label='Random Search Mean Best-So-Far', color='green')

    plt.xlabel('Iteration')
    plt.ylabel('Best-So-Far Performance')
    plt.title('SMBO vs Random Search: Mean Best-So-Far Performance')
    plt.legend()
    plt.show()

    #### Plot 4: Best-So-Far Among Three Random Runs ####
    plt.figure(figsize=(8, 6))
    
    # Randomly select three runs from SMBO
    smbo_runs = random.sample(range(len(smbo_scores)), 3)
    for i, smbo_run in enumerate(smbo_runs):
        smbo_best_so_far = np.minimum.accumulate(smbo_scores_array[smbo_run])
        plt.plot(smbo_best_so_far, label=f'SMBO Best-So-Far Run {smbo_run}', color=f'blue', linestyle=['-', '--', ':'][i])

    # Randomly select three runs from Random Search
    random_search_runs = random.sample(range(len(random_search_scores)), 3)
    for i, rs_run in enumerate(random_search_runs):
        rs_best_so_far = np.minimum.accumulate(random_search_scores_array[rs_run])
        plt.plot(rs_best_so_far, label=f'Random Search Best-So-Far Run {rs_run}', color=f'green', linestyle=['-', '--', ':'][i])

    plt.xlabel('Iteration')
    plt.ylabel('Best-So-Far Performance')
    plt.title('Best-So-Far Performance: SMBO vs Random Search (Three Random Runs)')
    plt.legend()
    plt.show()


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

    smbo_file_path = 'results2/smbo_results_1.pkl'
    rs_file_path = 'results2/rs_results_1.pkl'

    # Try to load existing results
    smbo_results = load_results(smbo_file_path)
    rs_results = load_results(rs_file_path)

    # If results do not exist, run the experiments and save the results
    if smbo_results is None:
        print("Running SMBO experiment...")
        smbo_results = smbo_experiment(args, config_space, surrogate_model)
        save_results(smbo_file_path, smbo_results)

    if rs_results is None:
        print("Running Random Search experiment...")
        rs_results = random_search_experiment(args, config_space, surrogate_model)
        save_results(rs_file_path, rs_results)

    compare_smbo_and_random_search(smbo_results, rs_results)

# Run the main function
if __name__ == "__main__":
    main(parse_args())
