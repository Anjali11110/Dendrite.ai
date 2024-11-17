import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def parse_json_structure(json_input):
    """
    Extracts essential parameters and metadata from JSON input.
    """
    config_data = json_input.get("design_state_data", {})
    project_info = config_data.get("session_info", {})
    target_details = config_data.get("target", {})
    training_details = config_data.get("train", {})
    algorithms = config_data.get("algorithms", {})
    
    chosen_algorithm, algo_config = None, None
    for name, details in algorithms.items():
        if details.get("is_selected"):
            chosen_algorithm = name
            algo_config = details
            break
    
    return {
        "dataset_file": project_info.get("dataset"),
        "target_feature": target_details.get("target"),
        "data_split_policy": training_details.get("policy"),
        "training_ratio": training_details.get("train_ratio", 0.8),
        "random_seed": training_details.get("random_seed", 42),
        "algorithm_name": chosen_algorithm,
        "algorithm_details": algo_config
    }

def prepare_dataset(file_path, target_column):
    """
    Reads the dataset from the given path and separates features from the target variable.
    """
    try:
        data_frame = pd.read_csv(file_path)
        X = data_frame.drop(columns=[target_column])
        y = data_frame[target_column]
        return X, y
    except Exception as err:
        raise RuntimeError(f"Failed to load or process dataset: {err}")

def select_and_train_model(X_train, y_train, X_test, y_test, algorithm_config):
    """
    Trains a machine learning model based on the provided configuration.
    """
    algorithm_name = algorithm_config.get("algorithm_name")
    algo_params = algorithm_config.get("algorithm_details", {})
    
    if algorithm_name == "RandomForestRegressor":
        num_trees = algo_params.get("max_trees", 100)
        max_depth = algo_params.get("max_depth", None)
        model = RandomForestRegressor(
            n_estimators=num_trees,
            max_depth=max_depth,
            random_state=algo_params.get("random_state", 42)
        )
    else:
        raise ValueError(f"Unsupported algorithm specified: {algorithm_name}")
    
    # Train the model and evaluate its performance
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse_score = mean_squared_error(y_test, predictions)
    print(f"Performance: Mean Squared Error = {mse_score:.3f}")
    return model

def execute_workflow(json_path, csv_path):
    """
    Orchestrates the entire process from JSON loading to model training.
    """
    try:
        with open(json_path, 'r') as file:
            raw_json = file.read()
        json_content = json.loads(raw_json)
    except Exception as err:
        raise RuntimeError(f"Error reading or parsing JSON file: {err}")
    
    # Extract configuration from JSON
    config = parse_json_structure(json_content)
    
    # Load and process the dataset
    features, target = prepare_dataset(csv_path, config["target_feature"])
    
    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=1 - config["training_ratio"],
        random_state=config["random_seed"]
    )
    
    # Train the selected model
    trained_model = select_and_train_model(X_train, y_train, X_test, y_test, config)
    return trained_model

# Paths for JSON configuration and dataset
json_input_path = "algoparams_from_ui.json.rtf"
dataset_file_path = "iris.csv"

# Execute the entire workflow
final_model = execute_workflow(json_input_path, dataset_file_path)
