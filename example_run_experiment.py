import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='lcdb_configs.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--max_anchor_size', type=int, default=1600)
    parser.add_argument('--num_iterations', type=int, default=50)

    return parser.parse_args()

def run_random_sampling_experiments(args, num_experiments=10):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    random_search = RandomSearch(config_space)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df, test=True)
    
    all_results = []

    for experiment_idx in range(num_experiments):
        results = []
        for idx in range(args.num_iterations):
            theta_new = dict(random_search.select_configuration())
            theta_new['anchor_size'] = args.max_anchor_size
            performance = surrogate_model.predict(theta_new)
            results.append(performance)

            random_search.update_runs((theta_new, performance))

        all_results.append(results)

    # Convert the results to a numpy array for easier computation of mean and variance
    all_results = np.array(all_results)
    
    # Calculate the mean and variance of the experiments
    mean_results = np.mean(all_results, axis=0)
    std_results = np.std(all_results, axis=0)

    # Plot the mean with variance bounds
    plt.fill_between(range(args.num_iterations), 
                     mean_results - std_results, 
                     mean_results + std_results, 
                     color='gray', alpha=0.3, label='Mean ± Variance')

    # Plot the mean line
    plt.plot(range(args.num_iterations), mean_results, color='black', label='Mean', linewidth=2)

    plt.xlabel('Iterations')
    plt.ylabel('Performance')
    plt.yscale('log')
    plt.legend()
    plt.show()


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)

    random_search = RandomSearch(config_space)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df, test= True)
    results = {
        'random_search': [1.0]
    }

    best_performer= None,1

    for idx in range(args.num_iterations):
        theta_new = dict(random_search.select_configuration())
        # Check if any value in theta_new is None
        theta_new['anchor_size'] = args.max_anchor_size
        performance = surrogate_model.predict(theta_new)
        # ensure to only record improvements
        results['random_search'].append(min(results['random_search'][-1], performance))  
        ## TODO remove me
        if performance < best_performer[1]:
            best_performer= theta_new, performance

            
        random_search.update_runs((theta_new, performance))  

    print(f"Best performing parameters with score: {best_performer[1]}")
    print(best_performer[0])

    plt.plot(range(len(results['random_search']) - 1), results['random_search'][1:])
    plt.yscale('log')
    plt.show()

    run_random_sampling_experiments(args)





if __name__ == '__main__':
    run(parse_args())
