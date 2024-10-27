#!/bin/bash

# Loop over random seeds
for seed in {1..10}
do
    # Run the Python script with the current random seed
    python run_smbo.py --random_seed $seed --max_anchor_size 1200
    
    # Create a folder for the results based on the seed
    result_folder="results_seed_$seed"
    mkdir -p $result_folder
    
    # Move all generated .pkl files to the corresponding folder
    mv *.pkl $result_folder/
    
    # Print a message indicating the run is complete
    echo "Run with seed $seed completed and files moved to $result_folder."
done