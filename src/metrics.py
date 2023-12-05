## IMPORTS
from scipy.stats import norm
import numpy as np
import sklearn


# Metrics
def compute_validity(x, model):
    return model.predict(x)

def compute_sparsity0(x0, x):
    sparsity_value = (x!=x0).mean(-1)
    return sparsity_value

def compute_sparsity1(x0, x):
    sparsity_value = (x-x0).mean(-1)
    return sparsity_value

def compute_proximity(x0, x):
    return np.diag(sklearn.metrics.pairwise.cosine_similarity(x0, x))


def compute_norm_preference(counter_norm, current_norm, norm_params):
    norm_pref = norm.pdf(current_norm-counter_norm,
                        norm_params["loc"][:,None],
                        norm_params["scale"][:,None])
    return norm_pref

def compute_cat_preference(counter_cat, cat_params):
    #counter_cat has shape (n_users, n_counterfactuals, n_categorical_features)
    #cat_params has shape (n_users, n_categorical_features, n_categories)
    i, j, k = np.indices(counter_cat.shape)

    # Use advanced indexing to construct the output
    cat_pref = cat_params[i, k, counter_cat[i, j, k]]
    return cat_pref

def compute_preference(counterfactuals_feature_values, current_values, continuous_features_ids, categorical_features_ids, norm_params, cat_params, decimal_precision=16):
    # for categorical variables
    if len(categorical_features_ids) == 0:
        cat_pref = np.empty(0)
    else:
        cat_pref = compute_cat_preference(counterfactuals_feature_values[:,:,categorical_features_ids].astype(int), cat_params)
    # for norm variables
    norm_pref = compute_norm_preference(counterfactuals_feature_values[:,:,continuous_features_ids], current_values[:,None,continuous_features_ids], norm_params)

    #change precision of preference
    cat_pref = np.round(cat_pref,decimal_precision)
    norm_pref = np.round(norm_pref,decimal_precision)

    #final_pref = cat_pref.sum(-1) * norm_pref.sum(-1) #as product
    final_pref = (cat_pref.sum(-1) + norm_pref.sum(-1))/(cat_pref.shape[-1] + norm_pref.shape[-1]) #as mean

    return final_pref

def compute_diversity(x):
    num_samples, num_counterfactuals, num_features = x.shape
    diversity_value = np.zeros((num_samples))
    for i in range(num_samples):
      app = sklearn.metrics.pairwise.cosine_distances(x[i],x[i])
      app2 = np.triu_indices(num_counterfactuals,k=1)
      diversity_value[i] = app[app2].mean()
    return diversity_value