import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import logging
import itertools
import sys

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader

# Create a custom logger
custom_log_name = sys.argv[1] if len(sys.argv) > 1 else "regression"
log_filename = f"log_{custom_log_name}.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class NeuralNet(nn.Module):
    def __init__(self,
                 input_size,
                 n_neurons,
                 n_hidden_layers,
                 architecture,
                 activation,
                 ):
        """
        Initialises a custom neural network with specified configuration.

        Arguments:
            - input_size {int} -- Number of input features (e.g., 13)
            - n_neurons {int} -- Number of neurons in the first hidden layer, also base number for other layers
            - n_hidden_layers {int} -- Number of hidden layers
            - arch_type {str} -- Architecture type of neural network:
                                    - pyramid (e.g., 128->64->32)
                                    - rectangular (e.g., 64->64->64)
            - activation {str} -- Activation function to use
        """
        super().__init__()

        # Get activation function
        activation_functions = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU()
        }

        try:
            self.activation = activation_functions[activation.lower()]
        except KeyError:
            raise ValueError(f"Activation must be 'relu', 'leakyrelu' or 'sigmoid'")

        # Get neural network layers
        layers = [nn.Linear(input_size, n_neurons), self.activation]

        # Create layers based on architecture type
        if architecture.lower() == "pyramid":
            layer_size = n_neurons
            for _ in range(n_hidden_layers):
                # Decay layer size but not below a reasonable minimum (e.g., 4)
                layers.append(nn.Linear(layer_size, max(4, int(layer_size / 2))))
                layers.append(self.activation)
                layer_size = max(4, int(layer_size / 2))

        elif architecture.lower() == "rectangular":
            for _ in range(n_hidden_layers):
                layers.append(nn.Linear(n_neurons, n_neurons))
                layers.append(self.activation)
        
        # Append final output layer
        layers.append(nn.Linear(layers[-2].out_features, 1))  # Final output layer

        self.network = nn.Sequential(*layers)
        
        # Logging for info
        logger.info(f"NeuralNet initialized with architecture {[layer.out_features for layer in layers if isinstance(layer, nn.Linear)]}")
        logger.info(f"Activation function: {activation}")

    def forward(self, x):
        return self.network(x)

class Regressor:
    def __init__(self,
                 x,
                 nb_epoch=1000,
                 learning_rate=0.001,
                 weight_decay=0.0,
                 batch_size=32,
                 n_neurons=64,
                 n_hidden_layers=4,
                 architecture="pyramid",
                 activation="relu",
                 training=True,
                 early_stopping=True,
                 ):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.
            - learning_rate {float} -- Learning rate for the optimizer.
            - weight_decay {float} -- Weight decay (L2 penalty) for the optimizer.
            - batch_size {int} -- Batch size for training.
            - n_neurons {int} -- Number of neurons in the first hidden layer, also base number for other layers.
            - n_hidden_layers {int} -- Number of hidden layers.
            - architecture {str} -- Architecture type of neural network:
                                    - pyramid (e.g., 128->64->32)
                                    - rectangular (e.g., 64->64->64)
            - activation {str} -- Activation function to use
            - training {bool} -- Boolean indicating if model training or inference.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Initialize model and training parameters
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.early_stopping = early_stopping

        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.architecture = architecture
        self.activation = activation

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize neural network
        if training:
            X, _ = self._preprocessor(x, training=training)
            input_size = X.shape[1]

            self.network = NeuralNet(input_size=input_size,
                                     n_hidden_layers=self.n_hidden_layers,
                                     n_neurons=self.n_neurons,
                                     architecture=self.architecture,
                                     activation=self.activation).to(self.device)

            self.optimizer = torch.optim.Adam(self.network.parameters(),
                                              lr=self.learning_rate,
                                              weight_decay=self.weight_decay)

        self.loss_layer = nn.MSELoss()

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self,
                      x,
                      y=None,
                      training=False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {bool} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        X = x.copy()
        if y is not None:
            Y = y.copy()
            non_nan_indices = Y.iloc[:, 0].notna() # Sanity check: Drop rows with NaN target values
            X = X[non_nan_indices]
            Y = Y[non_nan_indices]
        
        # Fit preprocesser during training mode
        if training:
            numeric_features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
                                "population", "households", "median_income"]
            categorical_features = ["ocean_proximity"]

            numeric_transformer = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=5, weights="uniform")),
                ("scaler", StandardScaler())])
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))])
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", numeric_transformer, numeric_features),
                    ("categorical", categorical_transformer, categorical_features)],
                remainder="passthrough")

            self.preprocessor.fit(x)

        # Apply transformations
        X = self.preprocessor.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            Y = torch.tensor(Y.values, dtype=torch.float32)

        # Return preprocessed x and y, return None for y if it was None
        return X, (Y if y is not None else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self,
            x_train,
            y_train,
            x_val=None,
            y_val=None,
            patience=25,
            ):
        """
        Regressor training function

        Arguments:
            - x_train {pd.DataFrame} -- Raw input training array of shape 
                (batch_size, input_size).
            - y_train {pd.DataFrame} -- Raw output training array of shape (batch_size, 1).
            - x_val {pd.DataFrame} -- Raw input validation array of shape 
                (batch_size, input_size).
            - y_val {pd.DataFrame} -- Raw output validation array of shape (batch_size, 1).
            - patience {int} -- Patience for early stopping.

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X_train, Y_train = self._preprocessor(x=x_train, y=y_train, training=True) # Do not forget

        # Load training data by batches
        generator = torch.Generator().manual_seed(42) # Seed for reproducibility
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=generator)

        # Training and validation tracking
        self.train_losses = []
        self.val_losses = []

        # Early stopping parameters
        best_val_loss = float("inf")
        epochs_no_improve = 0

        # Training loop
        for epoch in range(self.nb_epoch):
            self.network.train()
            epoch_train_loss = 0.
            num_batches = 0

            for X_train_batch, Y_train_batch in train_loader:
                X_train_batch, Y_train_batch = X_train_batch.to(self.device), Y_train_batch.to(self.device)

                self.optimizer.zero_grad()
                Y_train_prediction = self.network(X_train_batch)
                train_loss = self.loss_layer(Y_train_prediction, Y_train_batch)
                train_loss.backward()
                self.optimizer.step()

                # Accumulate training loss
                epoch_train_loss += train_loss.item()
                num_batches += 1

            # Average training loss for the epoch
            avg_epoch_train_loss = epoch_train_loss / num_batches
            self.train_losses.append(avg_epoch_train_loss)

            # Validation step
            if x_val is not None and y_val is not None:
                self.network.eval()

                with torch.no_grad():
                    X_val, Y_val = self._preprocessor(x_val, y_val, training=False)
                    X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
                    Y_val_prediction = self.network(X_val)
                    val_loss = self.loss_layer(Y_val_prediction, Y_val)
                    epoch_val_loss = val_loss.item()
                    self.val_losses.append(epoch_val_loss)

                # Log progress every 100 epochs
                if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == self.nb_epoch - 1:
                    logger.info(f"Epoch {epoch + 1}/{self.nb_epoch}, Avg. Train Loss: {avg_epoch_train_loss:.4f}, Val. Loss: {epoch_val_loss:.4f}")

                # Early stopping check
                if self.early_stopping:
                    if epoch_val_loss < best_val_loss:
                        best_val_loss = epoch_val_loss
                        epochs_no_improve = 0

                        best_model_state = {
                            "epoch": epoch,
                            "model_state_dict": self.network.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": epoch_val_loss,
                        }
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        self.network.load_state_dict(best_model_state["model_state_dict"])
                        self.optimizer.load_state_dict(best_model_state["optimizer_state_dict"])
                        break

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        X = X.to(self.device)

        self.network.eval()
        with torch.no_grad():
            Y_prediction = self.network(X)

        return Y_prediction.cpu().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y, return_metrics=False):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
            - return_metrics {bool} -- If true, return detailed metrics.

        Returns:
            {float} -- Quantification of the efficiency of the model.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        Y_prediction = self.predict(x)
        Y_true = y.values

        # NaN handling in true and predicted values
        valid_mask = ~np.isnan(Y_true.flatten()) & ~np.isnan(Y_prediction.flatten())
        if not np.any(valid_mask):
            logger.warning("Warning: No valid data to score. Model may have exploded (all NaN).")
            return float("inf")  # Return infinite error for bad models

        # Filter for valid rows only
        Y_true = Y_true[valid_mask]
        Y_prediction = Y_prediction[valid_mask] 

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(Y_true, Y_prediction))
        mae = mean_absolute_error(Y_true, Y_prediction)
        r2 = r2_score(Y_true, Y_prediction)

        # Log metrics
        if return_metrics:
            logger.info(f"=== Regression Model Performance ===")
            logger.info(f"RMSE: {rmse:.2f}")
            logger.info(f"MAE: {mae:.2f}")
            logger.info(f"R2: {r2:.2f}")
            logger.info(f"===================================")

        return float(rmse) # Replace this code with your own

def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open("part2_model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")

def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open("part2_model.pickle", "rb") as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

def perform_hyperparameter_search(x, y): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        x {pd.DataFrame} -- Raw input array of shape 
            (batch_size, input_size).
        y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

    Returns:
        best_params {dict} -- Dictionary of the best hyper-parameters found.
        results {list} -- List of all hyper-parameter combinations and their scores. 
    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # Parameter search grid, modify accordingly
    param_grid = {
        "learning_rate": [0.001, 0.01],
        "weight_decay" : [0.0, 0.001],
        "n_hidden_layers" : [2, 4, 6],
        "n_neurons" : [32, 64, 128, 256],
        "batch_size" : [32, 64, 128],
        "architecture" : ["pyramid", "rectangular"],
        "activation": ["relu", "sigmoid", "leakyrelu"]
    }

    # Generate all combinations of hyper-parameters
    keys = list(param_grid.keys())
    all_combinations = list(itertools.product(*[param_grid[key] for key in keys]))

    # Parameter search parameters
    best_score = float("inf")
    best_params = {}
    results = []  # To store results for analysis

    k_folds = 10
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Paramter grid search
    logger.info(f"Starting {k_folds}-Fold Hyperparameter search: {len(all_combinations)} Combinations")
    logger.info(f"Total Runs: {len(all_combinations) * k_folds}")

    for idx, combination in enumerate(all_combinations):
        params = dict(zip(keys, combination))
        fold_scores = []
        logger.info(f"=== Testing Combination {idx + 1}/{len(all_combinations)}: {params} ===")

        for fold, (train_ids, val_ids) in enumerate(kfold.split(x)):
            logger.info(f"=== Fold {fold + 1}/{k_folds} ===")
            
            # Split data into training and validation sets
            x_train_fold, y_train_fold = x.iloc[train_ids], y.iloc[train_ids]
            x_val_fold, y_val_fold = x.iloc[val_ids], y.iloc[val_ids]

            # Initialize and train model with current hyper-parameters
            model = Regressor(x_train_fold,
                              nb_epoch=500,
                              learning_rate=params["learning_rate"],
                              weight_decay=params["weight_decay"],
                              batch_size=params["batch_size"],
                              n_neurons=params["n_neurons"],
                              n_hidden_layers=params["n_hidden_layers"],
                              architecture=params["architecture"],
                              activation=params["activation"],
                              training=True,
                              early_stopping=True,
                              )
            model.fit(x_train_fold, y_train_fold, x_val_fold, y_val_fold, patience=15)
            score = model.score(x_val_fold, y_val_fold, return_metrics=False)
            fold_scores.append(score)

        # Average score across folds
        avg_score = np.mean(fold_scores)
        params["score"] = avg_score
        results.append((params, avg_score))
        logger.info(f"=== Average RMSE: {avg_score:.2f} ===")

        # Track best parameters
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            logger.info(f"=== New Best Hyperarameters: {best_params} ===")
            logger.info(f"=== New Best RMSE: {best_score:.2f} ===")

    # Return best and top N hyperparameter combinations
    N = 10
    results.sort(key=lambda x: x[1])  # Sort by RMSE

    logger.info(f"=== Top {N} Hyperparameter Combinations ===")
    for idx, result in enumerate(results[:N]):
        param = result[0]
        logger.info(f"{idx + 1}. Avg Val Loss: {result[1]:.2f}")
        logger.info(f"|Rank {idx + 1}| LR: {param["learning_rate"]}, WD: {param["weight_decay"]}, Layers: {param["n_hidden_layers"]}, Neurons: {param["n_neurons"]}, Batch Size: {param["batch_size"]}, Arch: {param["architecture"]}, Act: {param["activation"]}")

    return best_params, result # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def train_val_test_split(x,
                         y,
                         val_size=0.2,
                         test_size=0.2,
                         random_state=None):
    """
    Splits data into train, validation, and test sets.

    This function first splits the data into a (train+val) set and a test set. Then, it splits the (train+val) set into
    the final train and validation sets.

    Arguments:
        - x {pd.DataFrame} -- Input features
        - y {pd.DataFrame} -- Target values
        - val_size {float} -- Proportion of the (train+val) dataset to include in the val split
        - test_size {float} -- Proportion of dataset to include in the test split. Defaults to 0.2
        - random_state {int} -- Random seed for reproducibility.
    Returns:
        - {tuple} -- (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    # Split intro train and test split first
    x_train_full, x_test, y_train_full, y_test = train_test_split(x,
                                                                  y,
                                                                  test_size=test_size,
                                                                  random_state=random_state,
                                                                  shuffle=True)
    
    # Further split train into train and val
    val_relative_size = val_size / (1 - test_size)  # Adjust val size
    x_train, x_val, y_train, y_val = train_test_split(x_train_full,
                                                      y_train_full,
                                                      test_size=val_relative_size,
                                                      random_state=random_state,
                                                      shuffle=True)
    
    # Return train, val, test splits (plus train-val full set for KFold param search)
    return x_train_full, x_train, x_val, x_test, y_train_full, y_train, y_val, y_test

def set_seed(seed):
    """
    Sets the random seed for NumPy and PyTorch for reproducibility.
    """
    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed for CPU
    torch.manual_seed(seed)

    # Set PyTorch's random seed for GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    return None

def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Example main function to demonstrate usage
    output_label = "median_house_value"

    # Load dataset
    data = pd.read_csv("housing.csv") 

    # Split input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Split into train, val, test sets
    x_train_full, x_train, x_val, x_test, y_train_full, y_train, y_val, y_test = train_val_test_split(x, y, random_state=42)

    # Perform hyperparameter search
    #best_params, results = perform_hyperparameter_search(x_train_full, y_train_full)

    best_params = {
         "learning_rate": 0.01,
         "weight_decay": 0.0,
         "n_hidden_layers": 4,
         "n_neurons": 256,
         "batch_size": 32,
         "architecture": "rectangular",
         "activation": "relu"
     }

    # Train final model with best hyperparameters
    final_model = Regressor(x_train,
                            nb_epoch=1000,
                            learning_rate=best_params["learning_rate"],
                            weight_decay=best_params["weight_decay"],
                            batch_size=best_params["batch_size"],
                            n_neurons=best_params["n_neurons"],
                            n_hidden_layers=best_params["n_hidden_layers"],
                            architecture=best_params["architecture"],
                            activation=best_params["activation"],
                            training=True,
                            early_stopping=True,
                            )
    final_model.fit(x_train, y_train, x_val, y_val, patience=25)

    # Evaluate on test set
    test_error = final_model.score(x_test, y_test, return_metrics=True)
    logger.info(f"\nFinal model test RMSE: {test_error}\n")

    # Save final model
    save_regressor(final_model)
    logger.info("Final model saved successfully.")

if __name__ == "__main__":
    main()
