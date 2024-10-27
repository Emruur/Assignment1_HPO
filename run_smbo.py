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

def random_search_experiment(args, config_space,groundtruth, n_runs= 10, num_iterations= None) -> list[list[float]]:
    num_iter= args.num_iterations
    if num_iterations != None:
        num_iter = num_iterations
    random_search = RandomSearch(config_space)
    run_results= []
    for i in range(n_runs):
        print(f"starting rs run {i}")
        run_result= []
        for idx in range(num_iter):
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
    for i in range(n_runs):
        print(f"starting smbo run {i}")

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

def compare_smbo_rs_sh(smbo_scores: list[list[float]], random_search_scores: list[list[float]], sh_results: list[dict]):
    """
    Compare SMBO, Random Search, and Successive Halving (SH) by plotting:
    1. Mean performance with variance bounds for SMBO and Random Search (over iterations).
    2. Mean best-so-far performance across SMBO, Random Search, and SH, with SH plotted against cumulative budget.
       Minimum values displayed in legend.
    3. Best-so-far performance across three randomly selected runs for SMBO, Random Search, and SH, plotted against cumulative budget.
       Each curve's minimum value displayed in legend.
    4. Best-so-far performance for three random RS instances over iterations.
    5. Successive Halving single-run bandit performance across halving steps.
    
    :param smbo_scores: A list of lists where each inner list represents the performance scores for SMBO over iterations.
    :param random_search_scores: A list of lists where each inner list represents the performance scores for Random Search over iterations.
    :param sh_results: List of dictionaries, where each dictionary contains `bandit_performance`, `best_so_far`, and `cumulative_budget` for a separate SH run.
    """
    smbo_array = np.array(smbo_scores)
    random_search_array = np.array(random_search_scores)

    # Process SH results
    sh_best_so_far_all_runs = [result["best_so_far"] for result in sh_results]
    sh_cumulative_budget_all_runs = [result["cumulative_budget"] for result in sh_results]

    # Calculate mean SH best-so-far and cumulative budget across all SH runs
    sh_mean_best_so_far = np.mean(sh_best_so_far_all_runs, axis=0)
    sh_mean_cumulative_budget = np.mean(sh_cumulative_budget_all_runs, axis=0)

    # Minimum values for Plot 2
    smbo_min_val = np.min(np.mean(np.minimum.accumulate(smbo_array, axis=1), axis=0))
    random_min_val = np.min(np.mean(np.minimum.accumulate(random_search_array, axis=0), axis=0))
    sh_min_val = np.min(sh_mean_best_so_far)

    # Select three SH runs for random best-so-far plot
    sh_best_so_far_sampled = [result["best_so_far"] for result in random.sample(sh_results, 3)]
    sh_cumulative_budget_sampled = [result["cumulative_budget"] for result in random.sample(sh_results, 3)]

    # Use a single SH run for bandit performance
    sh_single_run = sh_results[0]
    sh_bandit_perf = sh_single_run["bandit_performance"]

    # Calculate the cumulative budget for SMBO and Random Search (assuming 1200 budget per iteration)
    smbo_best_so_far = np.minimum.accumulate(smbo_array, axis=1)
    random_search_best_so_far = np.minimum.accumulate(random_search_array, axis=1)

    smbo_cumulative_budget = np.arange(1, smbo_array.shape[1] + 1) * 1200
    random_cumulative_budget = np.arange(1, random_search_array.shape[1] + 1) * 1200

    smbo_mean_best_so_far = np.mean(smbo_best_so_far, axis=0)
    random_mean_best_so_far = np.mean(random_search_best_so_far, axis=0)

    #### Plot 1: Mean Performance with Variance Bounds over Iterations ####
    smbo_mean_scores = np.mean(smbo_array, axis=0)
    smbo_std_scores = np.std(smbo_array, axis=0)
    random_mean_scores = np.mean(random_search_array, axis=0)
    random_std_scores = np.std(random_search_array, axis=0)

    smbo_iterations = np.arange(1, smbo_array.shape[1] + 1)
    random_iterations = np.arange(1, random_search_array.shape[1] + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(smbo_iterations, smbo_mean_scores, label='SMBO Mean Performance', color='blue')
    plt.fill_between(smbo_iterations,
                     smbo_mean_scores - smbo_std_scores,
                     smbo_mean_scores + smbo_std_scores,
                     color='blue', alpha=0.2, label='SMBO Variance')

    plt.plot(random_iterations, random_mean_scores, label='Random Search Mean Performance', color='green')
    plt.fill_between(random_iterations,
                     random_mean_scores - random_std_scores,
                     random_mean_scores + random_std_scores,
                     color='green', alpha=0.2, label='Random Search Variance')

    plt.xlabel('Iteration')
    plt.ylabel('Performance Score')
    plt.title('SMBO vs Random Search: Mean Performance with Variance Bounds (Over Iterations)')
    plt.legend()
    plt.show()

    #### Plot 2: Mean Best-So-Far Performance with SH Cumulative Budget (Min Values in Legend) ####
    plt.figure(figsize=(8, 6))
    plt.plot(smbo_cumulative_budget, smbo_mean_best_so_far, label=f'SMBO Mean Best-So-Far (Min: {smbo_min_val:.2f})', color='blue')
    plt.plot(random_cumulative_budget, random_mean_best_so_far, label=f'RS Mean Best-So-Far (Min: {random_min_val:.2f})', color='green')
    plt.plot(sh_mean_cumulative_budget, sh_mean_best_so_far, label=f'SH Mean Best-So-Far (Min: {sh_min_val:.2f})', color='red')

    plt.xlabel('Cumulative Budget')
    plt.ylabel('Best-So-Far Performance')
    plt.title('Mean Best-So-Far Performance: SMBO, Random Search, and SH')
    plt.legend()
    plt.show()

    #### Plot 3: Best-So-Far Among Three Random Runs for SMBO, RS, and SH (Cumulative Budget with Min Values) ####
    plt.figure(figsize=(8, 6))

    smbo_runs = random.sample(range(len(smbo_scores)), 3)
    for i, smbo_run in enumerate(smbo_runs):
        smbo_best_so_far = np.minimum.accumulate(smbo_array[smbo_run])
        smbo_min = np.min(smbo_best_so_far)
        plt.plot(smbo_cumulative_budget, smbo_best_so_far, label=f'SMBO Best-So-Far Run {smbo_run}, Min: {smbo_min:.3f}', color='blue', linestyle=['-', '--', ':'][i])

    random_runs = random.sample(range(len(random_search_scores)), 3)
    for i, rs_run in enumerate(random_runs):
        rs_best_so_far = np.minimum.accumulate(random_search_array[rs_run])
        rs_min = np.min(rs_best_so_far)
        plt.plot(random_cumulative_budget, rs_best_so_far, label=f'RS Best-So-Far Run {rs_run}, Min: {rs_min:.3f}', color='green', linestyle=['-', '--', ':'][i])

    for i in range(3):
        sh_best_so_far = sh_best_so_far_sampled[i]
        sh_min = np.min(sh_best_so_far)
        plt.plot(sh_cumulative_budget_sampled[i], sh_best_so_far, label=f'SH Best-So-Far Run {i+1}, Min: {sh_min:.3f}', color='red', linestyle=['-', '--', ':'][i])

    plt.xlabel('Cumulative Budget')
    plt.ylabel('Best-So-Far Performance')
    plt.title('Best-So-Far Performance: SMBO, Random Search, and SH (Three Random Runs)')
    plt.legend()
    plt.show()

    #### Plot 4: Best-So-Far for Three Random RS Instances Over Iterations ####
    plt.figure(figsize=(8, 6))

    random_rs_runs = random.sample(range(len(random_search_scores)), 3)
    for i, rs_run in enumerate(random_rs_runs):
        rs_best_so_far = np.minimum.accumulate(random_search_array[rs_run])
        plt.plot(random_iterations, rs_best_so_far, label=f'Random Search Best-So-Far Run {rs_run}', color='green', linestyle=['-', '--', ':'][i])

    plt.xlabel('Iteration')
    plt.ylabel('Best-So-Far Performance')
    plt.title('Random Search: Best-So-Far Performance (Three Random Runs)')
    plt.legend()
    plt.show()

    #### Plot 5: Successive Halving Bandit Performance Across Halving Steps (Single Run) ####
    plt.figure(figsize=(8, 6))
    for scores in sh_bandit_perf.values():
        plt.plot(range(len(scores)), scores, marker='o')

    plt.xlabel('Halving Step')
    plt.ylabel('Score')
    plt.title('Successive Halving Bandit Performance Across Halving Steps (Single Run)')
    plt.grid(True)
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
        # Predict the performance of the configuration using the surrogate model
        config_dict["anchor_size"]= anchor_size
        config_performance = model.predict(config_dict)
        
        # Append the configuration and its performance as a tuple to capital_phi
        capital_phi.append((config_dict, config_performance))
    
    return capital_phi

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config-performances/config_performances_dataset-6.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=50)
    parser.add_argument('--random_seed', type=int, default=42)  # Add this line

    return parser.parse_args()

# Main function to run the steps
def main(args):
    # File paths for dataset and config space (replace with correct paths)
    dataset_path = args.configurations_performance_file
    config_path = args.config_space_file
    

    # Read the files
    dataset, config_space = read_files(dataset_path, config_path)

    config_space.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Train the groundtruth surrogate model
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(dataset)

    smbo_file_path = 'smbo_results_100_1200_6.pkl'
    rs_file_path = 'rs_results_500_1200_6.pkl'
    sh_file_path= 'sh_1000_pre_anchors.pkl'


    # Try to load existing results
    smbo_results = load_results(smbo_file_path)
    rs_results = load_results(rs_file_path)
    sh_results= load_results(sh_file_path)

    # If results do not exist, run the experiments and save the results
    if smbo_results is None:
        print("Running SMBO experiment...")
        smbo_results = smbo_experiment(args, config_space, surrogate_model, n_runs=100)
        save_results(smbo_file_path, smbo_results)

    if rs_results is None:
        print("Running Random Search experiment...")
        rs_results = random_search_experiment(args, config_space, surrogate_model, num_iterations=75, n_runs= 100)
        save_results(rs_file_path, rs_results)

    if sh_results is None:
        print("Fuck")
        pass

    ## seed 2-6-7 -- 01
    ## 2-7- 1200
    #compare_smbo_rs_sh(smbo_results, rs_results, sh_results)

# Run the main function
if __name__ == "__main__":
    main(parse_args())
