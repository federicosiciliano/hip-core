## IMPORTS
import pandas as pd
import numpy as np
import os
import pickle as pkl
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
from hip_core import *

### Test

def test_model(dataset, model, lambda_params, categorical_features_ids, continuous_features_ids, estimated_norm_params, estimated_cat_params, true_norm_params, true_cat_params, features_min_max_values, ordinal_to_categorical_features):
    current_values = dataset.values
    num_samples = dataset.shape[0]
    num_features = dataset.shape[1]-1
    counterfactuals_feature_values = generate_counterfactuals(dataset, model, lambda_params,
                                                                categorical_features_ids, continuous_features_ids,
                                                                estimated_norm_params, estimated_cat_params,
                                                                features_min_max_values, ordinal_to_categorical_features,
                                                                1)

    counterfactuals_feature_values_flatten = counterfactuals_feature_values.reshape(dataset.shape[0],-1)

    validity_value = compute_validity(counterfactuals_feature_values_flatten, model).reshape(num_samples,1)
    
    preference_value = compute_preference(counterfactuals_feature_values, current_values, continuous_features_ids, categorical_features_ids, true_norm_params, true_cat_params)

    sparsity_value = compute_sparsity0(current_values[:,None,:-1], counterfactuals_feature_values)
    
    # Compute proximity as cosine distance
    proximity_value = compute_proximity(current_values[:,:-1], counterfactuals_feature_values_flatten).reshape(num_samples,1)

    metrics = [validity_value.flatten(),
                preference_value.flatten(),
                sparsity_value.flatten(),
                proximity_value.flatten()]

    return metrics, counterfactuals_feature_values

def load_data(name):
    if name in {"adult_income"}:
        dataset = pd.read_csv(os.path.join(processed_data_folder, f"{name}.csv"))
    elif name in {"HELOC"}:
        dataset = pd.read_csv(os.path.join(raw_data_folder, f"{name}_dataset_v1.csv"))
        dataset = dataset.replace("Bad",0)
        dataset = dataset.replace("Good",1)
        dataset = dataset[list(dataset.columns[1:]) + ['RiskPerformance']] #Put first column to the end
    elif name in {"GiveMeSomeCredit"}:
        dataset = pd.read_csv(os.path.join(raw_data_folder, f"{name}.csv"))
        dataset = dataset[list(dataset.columns[1:]) + ['SeriousDlqin2yrs']] #Put first column to the end
    elif name in {"default of credit card clients"}:
        dataset = pd.read_csv(os.path.join(raw_data_folder, f"{name}.csv"), delimiter=";")
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")
    return dataset


def get_model(name, model_type):
    model = pkl.load(open(f"../out/models/{name}_{model_type}_model.pkl", "rb"))
    return model


def prepare_data(dataset, features_types):
    categorical_features_values = {}
    for f in dataset.columns:
        if features_types[f] == 'categorical':
            categorical_features_values[f] = list(dataset[f].unique())

    categorical_to_ordinal_features = {}
    for f,categories in categorical_features_values.items():
        categorical_to_ordinal_features[f] = {category: i for i, category in enumerate(categories)}

    if len(categorical_features_values) == 0:
        ordinal_to_categorical_features = None
    else:
        max_len_categorical = max([len(categories) for categories in categorical_features_values.values()])
        ordinal_to_categorical_features = np.empty((len(categorical_features_values), max_len_categorical), dtype=object)
        for i,(f,categories) in enumerate(categorical_features_values.items()):
            ordinal_to_categorical_features[i,:len(categories)] = categories

    continuous_features_ids = [i for i,f in enumerate(features_types.keys()) if features_types[f] == 'continuous']

    categorical_features_ids = np.array([i for i,f in enumerate(features_types.keys()) if features_types[f] == 'categorical'])

    features_min_max_values = []
    for f in dataset.columns:
        if features_types[f] == 'continuous':
            features_min_max_values.append((dataset[f].min(), dataset[f].max()))
        elif features_types[f] == 'categorical':
            features_min_max_values.append((0,len(categorical_features_values[f])-1))

    return categorical_features_values, categorical_to_ordinal_features, ordinal_to_categorical_features, continuous_features_ids, categorical_features_ids, features_min_max_values


def prepare_preference_distributions(dataset, continuous_features_ids, categorical_features_ids, categorical_features_values, norm_loc = 0, eps = 1e-3):
    np.random.seed(0)
    true_norm_params,estimated_norm_params = {},{}
    true_norm_params["loc"] = np.ones((dataset.shape[0],len(continuous_features_ids)))*norm_loc
    true_norm_params["scale"] = np.random.uniform(eps, 5,(dataset.shape[0],len(continuous_features_ids)))
    estimated_norm_params["loc"] = np.random.uniform(-5, 5,(dataset.shape[0],len(continuous_features_ids)))
    estimated_norm_params["scale"] = np.random.uniform(eps, 5,(dataset.shape[0],len(continuous_features_ids)))

    if len(categorical_features_values) == 0:
        max_num_categories = 0
        true_cat_params = np.empty((dataset.shape[0],0,0))
        estimated_cat_params = np.empty((dataset.shape[0],0,0))
    else:
        max_num_categories = max([len(categories) for categories in categorical_features_values.values()])
        true_cat_params = np.random.dirichlet(np.ones(max_num_categories),size=(dataset.shape[0],len(categorical_features_values)))
        estimated_cat_params = np.random.dirichlet(np.ones(max_num_categories),size=(dataset.shape[0],len(categorical_features_values)))
        current_values = dataset.iloc[:,categorical_features_ids].values
        true_cat_params[np.arange(current_values.shape[0])[:, np.newaxis], np.arange(current_values.shape[1]), current_values] = 1
        estimated_cat_params[np.arange(current_values.shape[0])[:, np.newaxis], np.arange(current_values.shape[1]), current_values] = 1

    return true_norm_params,estimated_norm_params,true_cat_params,estimated_cat_params