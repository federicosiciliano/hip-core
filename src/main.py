## IMPORTS
import numpy as np
import os
import itertools as it
from multiprocessing.pool import Pool
#from multiprocessing import set_start_method
#set_start_method('spawn')
from functools import partial
import torch
torch.set_num_threads(1) #to solve freeze of torch when using multiprocessing


## Define paths

#every path should start from the project folder:
project_folder = "../"

#Config folder should contain hyperparameters configurations
cfg_folder = os.path.join(project_folder,"cfg")

#Data folder should contain raw and preprocessed data
data_folder = os.path.join(project_folder,"data")
raw_data_folder = os.path.join(data_folder,"raw")
processed_data_folder = os.path.join(data_folder,"processed")

#Source folder should contain all the (essential) source code
source_folder = os.path.join(project_folder,"src")

#The out folder should contain all outputs: models, results, plots, etc.
out_folder = os.path.join(project_folder,"out")
results_folder = os.path.join(out_folder,"results")

import exp_utils

from metrics import *
from hip_core import *
from data_utils import *

def main_function(model,
                  continuous_features_ids, categorical_features_ids,
                  categorical_features_values, features_min_max_values,
                  ordinal_to_categorical_features,
                  lambda_params, num_cf_per_iteration_per_user, num_iterations,
                  ord_dataset,
                  norm_loc = 0):
    # Phase 0: Prepare data
    true_norm_params,estimated_norm_params,true_cat_params,estimated_cat_params = prepare_preference_distributions(ord_dataset, continuous_features_ids, categorical_features_ids, categorical_features_values, norm_loc=norm_loc)

    # Metrics to save each epoch
    metrics = []

    #counterfactuals_with_preference to save each epoch
    counterfactuals_with_preference = np.zeros((num_iterations,ord_dataset.shape[0],num_cf_per_iteration_per_user,ord_dataset.shape[1]))

    #counterfactuals_true_preferences = {user_id: {feature_id: [] for feature_id in test_dataset.columns} for user_id in test_dataset.index}
    for iter_num in range(num_iterations):
        print(f"ITERATION {iter_num}")
        # Phase 1: Generate counterfactuals
        counterfactuals_feature_values = generate_counterfactuals(ord_dataset, model, lambda_params,
                                                                    categorical_features_ids, continuous_features_ids,
                                                                    estimated_norm_params, estimated_cat_params,
                                                                    features_min_max_values, ordinal_to_categorical_features,
                                                                    num_cf_per_iteration_per_user)

        # Phase 2: Generate preferences
        # Generate preferences for each counterfactual
        counterfactuals_preference_values = compute_preference(counterfactuals_feature_values, ord_dataset.values, continuous_features_ids, categorical_features_ids, true_norm_params, true_cat_params, cfg["preference.decimal_precision"])

        if iter_num > 0:
            # Check if the counterfactuals are the same as the previous iteration
            if (counterfactuals_feature_values == counterfactuals_with_preference[iter_num-1][:,:,:-1]).all():
                print("Generated counterfactuals are the same as the previous iteration")
                break

        # Append the counterfactuals to the list
        counterfactuals_with_preference[iter_num] = np.concatenate([counterfactuals_feature_values,
                                                                    counterfactuals_preference_values[:,:,None]],
                                                                    axis=-1)

        # Phase 3: Update preferences
        # Update preferences for each counterfactual
        # Create a single vector containing all params
        new_params = estimate_preferences(counterfactuals_with_preference[:(iter_num+1)], ord_dataset.values, continuous_features_ids, categorical_features_ids, estimated_norm_params, estimated_cat_params)

        # Update the parameters
        start = 0
        estimated_norm_params["loc"] = new_params[start:start+estimated_norm_params["loc"].size].reshape(estimated_norm_params["loc"].shape)
        start += estimated_norm_params["loc"].size
        estimated_norm_params["scale"] = new_params[start:start+estimated_norm_params["scale"].size].reshape(estimated_norm_params["scale"].shape)
        start += estimated_norm_params["scale"].size
        estimated_cat_params = new_params[start:].reshape(estimated_cat_params.shape)
        
        # Phase 4: Compute metrics
        # Compute metrics
        metrics_this_iter, generated_counterfactuals = test_model(ord_dataset, model, lambda_params, categorical_features_ids, continuous_features_ids, estimated_norm_params, estimated_cat_params, true_norm_params, true_cat_params, features_min_max_values, ordinal_to_categorical_features)
        metrics.append(metrics_this_iter)
    return metrics, counterfactuals_with_preference, generated_counterfactuals

# # MAIN

cfg = exp_utils.cfg.load_configuration(config_path=cfg_folder)

features_types = cfg["data.features_types"]

## Data

# Import whole dataset
dataset = load_data(cfg["data.name"])
dataset

### Dataset specific definitions

categorical_features_values, categorical_to_ordinal_features, ordinal_to_categorical_features, continuous_features_ids, categorical_features_ids, features_min_max_values = prepare_data(dataset, cfg["data.features_types"])

## Model

m_model = get_model(cfg["data.name"], cfg["model"])
acc = np.mean((m_model.predict(dataset.iloc[:,:-1])>=0.5)*1 == dataset.iloc[:,-1].values)
print("Accuracy: ", acc)


## Predictions

## Initialize

#dataset to target 0
dataset = dataset.loc[dataset.iloc[:,-1]==0]

#convert dataset to ordinal
complete_ord_dataset = dataset.copy()
for f in categorical_features_values.keys():
    complete_ord_dataset[f] = complete_ord_dataset[f].apply(lambda x: categorical_to_ordinal_features[f][x])

## Recourse

if cfg["recourse.lambda_params"] is None:
    len_comb = 5
    poss_probs = np.array([1/i for i in range(1,len_comb+1)]+[0]+[0.666666]).round(3)
    lambda_params_list = [list(combo) for combo in it.product(*[poss_probs for _ in range(len_comb)]) if sum(combo) == 1 and combo[0]>=1/len_comb]
else:
    lambda_params_list = cfg["recourse.lambda_params"]

results_keys = ["metrics","counterfactuals_with_preference", "generated_counterfactuals"]


if __name__ == "__main__":
    for lambda_params in lambda_params_list:
        cfg["recourse.lambda_params"] = lambda_params
        for num_cf_per_iteration_per_user in cfg.sweep("recourse.num_cf_per_iteration_per_user"):
            for norm_loc in cfg.sweep("preference.norm_loc"):
                exp_found, experiment_id = exp_utils.exp.get_set_experiment_id(cfg)
                print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)
                print(cfg["__exp__.experiment_id"])
                # Check if an experiment was found
                if exp_found:
                    # Print the values of exp_found (if found = true) and experiment_id
                    # If an experiment was found, go to next execution and display a message
                    continue
                # If no experiment was found, continue with execution
                pool_func = partial(main_function,m_model,
                                                    continuous_features_ids, categorical_features_ids,
                                                    categorical_features_values, features_min_max_values,
                                                    ordinal_to_categorical_features,
                                                    lambda_params, num_cf_per_iteration_per_user, cfg["recourse.num_iterations"],
                                                    norm_loc = norm_loc)
                with Pool(32) as pool:
                    output = pool.map(pool_func,
                                    [complete_ord_dataset.iloc[i:(i+1)] for i in range(len(complete_ord_dataset))])

                results = {key: [] for key in results_keys}
                for metrics_i, counterfactuals_with_preference_i, generated_counterfactuals_i in output:
                    results["metrics"].append(np.array(metrics_i))
                    results["counterfactuals_with_preference"].append(np.array(counterfactuals_with_preference_i))
                    results["generated_counterfactuals"].append(np.array(generated_counterfactuals_i))

                if not os.path.isdir(os.path.join(results_folder,experiment_id)):
                    os.makedirs(os.path.join(results_folder,experiment_id))
                    
                # Save results to npy
                for key, value in results.items():
                    np.save(os.path.join(results_folder,experiment_id,f"{key}.npy"), np.array(value, dtype=object), allow_pickle=True)

                # Save experiment and print the current configuration
                #save_experiment_and_print_config(cfg)
                exp_utils.exp.save_experiment(cfg)

                # Print completion message
                print("An execution is completed")
                print("######################################################################")
                print()


