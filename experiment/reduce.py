import pickle
import random

def reduce_sh_results(input_file: str, output_file: str, num_samples: int = 500):
    """
    Load a large SH results file, sample a specified number of entries, and save to a new file.
    
    :param input_file: Path to the input .pkl file (e.g., 'sh_1000_pre_anchors.pkl').
    :param output_file: Path to the output .pkl file (e.g., 'sh_500_pre_anchors.pkl').
    :param num_samples: Number of entries to sample from the input file.
    """
    # Load the original SH results
    with open(input_file, 'rb') as f:
        sh_results = pickle.load(f)
    
    # Randomly sample the specified number of entries
    sh_sampled_results = random.sample(sh_results, num_samples)
    
    # Save the sampled entries to the new file
    with open(output_file, 'wb') as f:
        pickle.dump(sh_sampled_results, f)

    print(f"Sampled {num_samples} entries from {input_file} and saved to {output_file}.")

# Usage
reduce_sh_results(input_file='sh_1000_pre_anchors.pkl', output_file='sh_500_pre_anchors.pkl')
