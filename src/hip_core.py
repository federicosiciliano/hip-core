## IMPORTS
import numpy as np
import os
from scipy.optimize import minimize
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

from metrics import *

## Counterfactual Generator

def counterfactuals_optim_func(params, params_min_max_values, dataset, model, lambda_params, categorical_features_ids, continuous_features_ids, estimated_norm_params, estimated_cat_params, ordinal_to_categorical_features):
    current_values = dataset.values
    params_reshaped = params.reshape(len(dataset), -1, len(params_min_max_values)) #params shape = (num_samples, num_counterfactuals, num_features)
    num_samples, num_counterfactuals, num_features = params_reshaped.shape

    params_flatten = params_reshaped.reshape(num_samples*num_counterfactuals,num_features)

    validity_value = compute_validity(params_flatten, model).reshape(num_samples,num_counterfactuals)

    preference_value = compute_preference(params_reshaped, current_values, continuous_features_ids, categorical_features_ids, estimated_norm_params, estimated_cat_params)

    # Compute sparsity as norm_0
    sparsity_value = compute_sparsity0(current_values[:,None,:-1], params_reshaped)
    sparsity_value = 1-sparsity_value #to minimize
    
    # Compute proximity as cosine distance
    repeated_current_values_flatten = np.repeat(current_values[:,None,:-1],num_counterfactuals,axis=1).reshape(num_samples*num_counterfactuals,num_features)
    proximity_value = compute_proximity(repeated_current_values_flatten, params_flatten).reshape(num_samples,num_counterfactuals)

    # Compute diversity = average cosine similarity between counterfactual
    if num_counterfactuals > 1:
        diversity_value = np.repeat(compute_diversity(params_reshaped),num_counterfactuals).reshape(num_samples,num_counterfactuals)
    else:
        diversity_value = np.zeros((num_samples,num_counterfactuals))

    #final_value = out_u + pref_u + diversity_u + sparsity_u + proximity_u
    final_value = np.einsum("i,i...->i...",lambda_params,np.array([validity_value, preference_value, sparsity_value, proximity_value, diversity_value])).mean()
    return final_value

def preference_optim_func(params, orig_shapes, data, current_values, continuous_features_ids, categorical_features_ids):
    # Convert params to original
    params_orig = []
    start = 0
    for shape in orig_shapes:
        app = np.prod(shape)
        params_orig.append(params[start:start+app].reshape(shape))
        start += app
    norm_params = {"loc":params_orig[0], "scale":params_orig[1]}
    cat_params = params_orig[2]

    counterfactuals_feature_values = data[:,:,:-1]
    true_preference_values = data[:,:,-1]

    estimated_preference_value = compute_preference(counterfactuals_feature_values.astype(int), current_values, continuous_features_ids, categorical_features_ids, norm_params, cat_params)

    squared_error = np.square(estimated_preference_value-true_preference_values).mean()
    return squared_error

def generate_counterfactuals(ord_dataset, model, lambda_params, categorical_features_ids, continuous_features_ids, estimated_norm_params, estimated_cat_params, features_min_max_values, ordinal_to_categorical_features, num_cf_per_iteration_per_user):
    # Phase 1: Generate counterfactuals
    initial_params = np.repeat(ord_dataset.values[:,None,:-1],num_cf_per_iteration_per_user,axis=1).reshape(-1) #must have one dimension for minimize

    min_func = lambda x: -counterfactuals_optim_func(x, features_min_max_values, ord_dataset, model, lambda_params, categorical_features_ids, continuous_features_ids, estimated_norm_params, estimated_cat_params, ordinal_to_categorical_features)
    bounds = np.tile(features_min_max_values, (num_cf_per_iteration_per_user*ord_dataset.shape[0],1))

    print("Starting optimization")
    # Minimize the MSE
    result_A = minimize(min_func, initial_params, method='Powell', bounds=bounds) #, options={"maxiter":1}

    counterfactuals_feature_values = result_A.x.reshape(ord_dataset.shape[0],num_cf_per_iteration_per_user,ord_dataset.shape[1]-1).round().astype(int)
    print("Finished optimization")

    return counterfactuals_feature_values

def estimate_preferences(counterfactuals_with_preference, current_values, continuous_features_ids, categorical_features_ids, estimated_norm_params, estimated_cat_params, eps = 1e-3):
    initial_guess = np.concatenate([estimated_norm_params["loc"].flatten(),
                                        estimated_norm_params["scale"].flatten(),
                                        estimated_cat_params.flatten()],
                                        axis=0)

    # Function to minimize
    min_func = lambda x: preference_optim_func(x, [estimated_norm_params["loc"].shape,
                                                    estimated_norm_params["scale"].shape,
                                                    estimated_cat_params.shape],
                                                    counterfactuals_with_preference.reshape(counterfactuals_with_preference.shape[1],-1,counterfactuals_with_preference.shape[3]),
                                                    current_values,
                                                    continuous_features_ids,
                                                    categorical_features_ids)

    # norm loc must be 0
    norm_loc_bounds = np.zeros((estimated_norm_params["loc"].size,2))
    norm_loc_bounds[:,0] = -5
    norm_loc_bounds[:,1] = +5
    norm_scale_bounds = np.zeros((estimated_norm_params["scale"].size,2))
    norm_scale_bounds[:,0] = eps
    norm_scale_bounds[:,1] = 5 # max scale
    cat_bounds = np.zeros((estimated_cat_params.size,2))
    cat_bounds[:,1] = 1

    params_bounds = np.concatenate([norm_loc_bounds,
                                    norm_scale_bounds,
                                    cat_bounds],
                                    axis=0)

    result_B = minimize(min_func, initial_guess, method='Powell', bounds=params_bounds) #method can be changed # constraints = constraint
    return result_B.x
