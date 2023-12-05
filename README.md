# Code for AISTATS submission, paper ID 1426: Human-in-the-loop Personalized Counterfactual Recourse

1. Place the data {dataset_name}.csv into the data/raw folder.

2. Execute the Train_Classifier.ipynb script located in the src folder to train the required classifier.

3. Adjust the configuration in config.yaml found in the cfg folder. Also, append a feature_types/{dataset_name}.yaml file for your dataset.

4. Run src/main.py to initiate the HIP-CORE framework with the designated configuration settings.