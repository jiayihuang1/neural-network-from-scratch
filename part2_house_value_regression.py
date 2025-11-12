from email import generator
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import sklearn
import torch.utils.data as data_utils
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import itertools
import random


class NeuralNet(nn.Module):

    # --- MODIFIED ---
    def __init__(self, input_size, n_hidden_layers, neurons, arch_type="pyramid", activation='relu'):
        """
        Initialises a flexible network.

        Arguments:
            - input_size {int} -- Number of input features (e.g., 13)
            - n_hidden_layers {int} -- Number of *hidden* layers
            - neurons {int} -- Number of neurons in the *first* hidden layer
            - arch_type {str} -- 'pyramid' (e.g., 128->64->32) or 'rectangular' (e.g., 64->64->64)
        """
        super().__init__()

        # Set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'lrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Activation must be 'relu', 'lrelu' or 'sigmoid'")

        layer_sizes = [input_size]
        current_neurons = neurons

        if arch_type == "pyramid":
            for _ in range(n_hidden_layers):
                layer_sizes.append(current_neurons)
                # Decay neurons, but never go below a reasonable minimum (e.g., 4)
                current_neurons = max(4, int(current_neurons / 2))
        elif arch_type == "rectangular":  # Rectangular
            for _ in range(n_hidden_layers):
                layer_sizes.append(neurons)

        layer_sizes.append(1)  # Final output layer

        print(f"Building network with architecture: {layer_sizes}")

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # Add ReLU *except* for the final output layer
            if i < len(layer_sizes) - 2:
                layers.append(self.activation)

        self.linear_relu_stack = nn.Sequential(*layers)


    def forward(self, x):
        return self.linear_relu_stack(x)


class Regressor:

    def __init__(self, x,
                 learning_rate=0.01,
                 weight_decay=0.0,
                 n_hidden_layers=2,
                 first_layer_neurons=64,
                 nb_epoch = 1000,
                 device=None,
                 batch_size=64,
                 architecture_type="pyramid",
                 activation="relu",
                 training=True
                 ):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- Number of epochs to train the network.
            - learning_rate {float} -- Learning rate for the optimizer.
            - weight_decay {float} -- L2 regularization strength.
            - layer_sizes {list} -- List of layer sizes.
            - loss_fun {str} -- "mse" or "cross_entropy".

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Storing parameters
        self.nb_epoch = nb_epoch
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_hidden_layers = n_hidden_layers
        self.first_layer_neurons = first_layer_neurons
        self.batch_size = batch_size
        self.architecture_type = architecture_type

        if device is None:
            self.device = torch.device("cpu")
            print("Warning: No device specified, using CPU.")
        else:
            self.device = device

        # Initialise storage for loss history
        self.train_loss_history = []
        self.val_loss_history = []
        self.early_stopping = True

        # To store loss function
        self._loss_layer = torch.nn.MSELoss()

        if training:
            X, _ = self._preprocessor(x, training=training)
            input_size = X.shape[1]

            self.network = NeuralNet(input_size,
                                     self.n_hidden_layers,
                                     self.first_layer_neurons,
                                     self.architecture_type,
                                     activation=self.activation).to(self.device)

            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """ ### STANDARDSCALER, ADAM OPTIMIZER,
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Replace this code with your own
        # Make copies to avoid changing original data
        x = x.copy()
        if y is not None:
            y = y.copy()

        # Delete rows where target value is NaN, must be done before processing X
        if y is not None:
            target_present_mask = y.iloc[:, 0].notna()
            x = x[target_present_mask]
            y = y[target_present_mask]

        # Fit preprocessor only during training
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
        x = self.preprocessor.transform(x)
        x = torch.tensor(x, dtype=torch.float32)
        if y is not None:
            y = torch.tensor(y.values, dtype=torch.float32)

        return x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def fit(self, x_train, y_train, x_val=None, y_val=None, patience=25):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X_train_processed, Y_train_processed = self._preprocessor(x_train, y = y_train, training = True) # Do not forget

        # Load training data in by batches
        generator = torch.Generator()
        generator.manual_seed(42)
        train_data = TensorDataset(X_train_processed, Y_train_processed)
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        # Make sure history is empty before a new fit
        self.train_loss_history = []
        self.val_loss_history = []

        # Initialise parameters to implement early stopping if validation loss does not improve after 'patience' epochs
        best_val_loss = float('inf')
        epochs_no_improve = 0

        self.network.train()

        # Repeat for n_epoch times
        for n in range(self.nb_epoch):
            # --- Training Loop ---
            self.network.train()

            # To track loss per epoch
            train_epoch_loss = 0.0
            num_batches = 0

            for X_batch, Y_batch in train_dataloader:
                # Move data to same device as model
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                # Forward pass through the network
                Y_train_prediction = self.network.forward(X_batch)
                train_loss = self._loss_layer(Y_train_prediction, Y_batch)

                # Zero the gradients before running backward pass
                self.optimizer.zero_grad()

                # Backward pass and take gradient descent step
                train_loss.backward()
                self.optimizer.step()

                # Accumulate loss
                train_epoch_loss += train_loss.item()
                num_batches += 1

            # At the end of an epoch, save the average loss
            avg_epoch_loss = train_epoch_loss / num_batches
            self.train_loss_history.append(avg_epoch_loss)

            # --- Validation Loop --- (if validation data is provided)
            epoch_val_loss = float('inf')
            if x_val is not None and y_val is not None:
                # Set model to evaluation mode
                self.network.eval()

                # Disable gradients for validation
                with torch.no_grad():
                    # Preprocess validation data
                    X_val_processed, Y_val_processed = self._preprocessor(x_val, y_val, training=False)

                    # Move validation data to device
                    X_val_processed, Y_val_processed = X_val_processed.to(self.device), Y_val_processed.to(self.device)

                    # Get predictions
                    Y_val_prediction = self.network(X_val_processed)

                    # Calculate loss
                    val_loss = self._loss_layer(Y_val_prediction, Y_val_processed)
                    epoch_val_loss = val_loss.item()

                    # Save validation loss
                    self.val_loss_history.append(val_loss.item())
                    epoch_val_loss_str = f"{val_loss.item():4f}"

                if (n + 1) % 100 == 0 or n == 0 or n == self.nb_epoch - 1:
                    print(
                        f"Epoch {n + 1}/{self.nb_epoch}, Avg. Train Loss: {avg_epoch_loss:.4f}, Val. Loss: {epoch_val_loss_str}")

                # --- Early Stopping Check ---
                if self.early_stopping:
                    if epoch_val_loss < best_val_loss:
                        best_val_loss = epoch_val_loss
                        epochs_no_improve = 0
                        # Save the best model state
                        best_model_state = {
                            "epoch": n,
                            "model_state_dict": self.network.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": epoch_val_loss
                        }
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered at epoch {n+1}")
                        # Restore the best model weights
                        self.network.load_state_dict(best_model_state["model_state_dict"])
                        self.optimizer.load_state_dict(best_model_state["optimizer_state_dict"])
                        break
                    #

        # If training completed without early stopping, restore best model if available
        if self.early_stopping and best_model_state is not None and epochs_no_improve < patience:
            print(f"Training completed!")
            print(f"Best validation loss: {best_val_loss:.4f}")
            self.network.load_state_dict(best_model_state["model_state_dict"])
            self.optimizer.load_state_dict(best_model_state["optimizer_state_dict"])

        return self.network

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

        X_processed, _ = self._preprocessor(x, training = False) # Do not forget
        X_processed = X_processed.to(self.device)

        self.network.eval()
        with torch.no_grad():
            Y_prediction = self.network.forward(X_processed)

        return Y_prediction.cpu().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y, print_metrics=True):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        y_pred = self.predict(x)

        # Get true values from the original dataframe
        y_true = y.values

        # Handle NaNs in y_true and in y_pred
        valid_mask = ~np.isnan(y_true.flatten()) & ~np.isnan(y_pred.flatten())

        if not np.any(valid_mask):
            print("Warning: No valid data to score. Model may have exploded (all NaN).")
            return float('inf')  # Return infinite error for bad models

        # 4. Filter both arrays to only the valid rows
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        # Calculate relevant indicators
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        r2 = r2_score(y_true_valid, y_pred_valid)

        # Print metrics
        if print_metrics:
            print("--- Regression Model Performance ---")
            print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
            print(f"  Mean Absolute Error (MAE):    {mae:.2f}")
            print(f"  R-squared (R²):               {r2:.3f}")
            print("--------------------------------------")

        return float(rmse) # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def plot_loss_history(self, save_path=None):
        """
        Plots the training and validation loss curves recorded during fit().

        Arguments:
            - save_path {str} -- Filepath to save the plot image.
                                If None, displays the plot instead.
        """
        print("Plotting training and validation loss...")
        plt.figure(figsize=(10, 6))

        plt.plot(self.train_loss_history, label='Training Loss')

        # Only plot validation loss if it was actually recorded
        if self.val_loss_history:
            plt.plot(self.val_loss_history, label='Validation Loss')

        plt.yscale('log')
        plt.title("Training & Validation Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss (MSE)")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"Saved loss curve to {save_path}")
        else:
            plt.show()  # Show the plot interactively


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def perform_hyperparameter_search(x_train_full, y_train_full):
    """
    Performs K-Fold cross-validation hyperparameter search.
    """
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    print("\n--- Starting K-Fold Hyperparameter Search ---")

    # --- Define search space ---
    param_grid = {
        "lr": [0.001, 0.01],
        "wd" : [0.0, 0.001],
        "n_layers" : [2, 4, 6],
        "first_neurons" : [32, 64, 128, 256],
        "bs" : [32, 64, 128],
        "arch_type" : ["pyramid", "rectangular"],
        "activation": ["relu", "sigmoid", "lrelu"]
    }

    # Generate all possible combinations
    keys = list(param_grid.keys())
    all_combinations = list(itertools.product(*[param_grid[key] for key in keys]))

    # --- End search space ---

    best_score = float('inf')
    best_params = {}
    results = []  # To store results for analysis

    # --- K-Fold setup ---
    k_folds = 10  # Increase this for proper training later on
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Simple grid search
    for idx, combination in enumerate(all_combinations):
        params = dict(zip(keys, combination))

        fold_scores = []
        # --- K-Fold Loop ---
        for fold, (train_ids, val_ids) in enumerate(kfold.split(x_train_full)):
            print(f"  --- Fold {fold + 1}/{k_folds} ---")
            # Get data for this fold
            x_train_fold, y_train_fold = x_train_full.iloc[train_ids], y_train_full.iloc[train_ids]
            x_val_fold, y_val_fold = x_train_full.iloc[val_ids], y_train_full.iloc[val_ids]

            # Create a NEW regressor for each fold
            regressor = Regressor(
                x_train_fold,
                learning_rate=params["lr"],
                weight_decay=params["wd"],
                n_hidden_layers=params["n_layers"],
                first_layer_neurons=params["first_neurons"],
                nb_epoch=500,  # Epoch cap for search
                device=set_device(),
                batch_size=params["bs"],
                architecture_type=params["arch_type"],
                training=True
            )

            regressor.fit(x_train_fold, y_train_fold, x_val_fold, y_val_fold, patience=15)
            score = regressor.score(x_val_fold, y_val_fold, print_metrics=False)
            fold_scores.append(score)

        # --- End K-Fold Loop ---
        avg_score = np.mean(fold_scores)
        print(f"--- Avg. K-Fold RMSE: {avg_score:.2f} ---")

        params['score'] = avg_score
        results.append((params, avg_score))

        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    # Sort results by average validation loss
    results.sort(key=lambda x: x[1])

    print("\n--- Hyperparameter Search Complete ---")
    print(f"Best K-Fold Validation RMSE: {best_score:.2f}")
    print(f"Best Hyperparameters: {best_params}")

    print(f"\nTop 10 Parameter Combinations:")
    print("-" * 70)
    for i, result in enumerate(results[:10]):
        p = result[0]
        print(f"{i + 1}. Avg Val Loss: {result[1]:.4f}")
        print(f"   LR={p['lr']:.4f}, WD={p['wd']:.4f}, Batch={p['bs']}, "
              f"Layers={p['n_layers']}, Size={p['first_neurons']}, "
              f"Act={p['activation']}")
    print("=" * 70 + "\n")

    # Return best params AND the full results for analysis
    return best_params, results
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def analyze_hp_search(results):
    """
    Analyzes and plots the results of the hyperparameter search.
    """
    if not results:
        print("No results to analyze.")
        return

    result_list = []
    for result in results:
        result_list.append(result[0])

    # Convert results list to a DataFrame for easy analysis
    results_df = pd.DataFrame(result_list)

    # Get all hyperparameter columns (exclude 'score')
    hp_cols = list(set(results_df.columns) - {'score'})

    print("\n--- Hyperparameter Performance Analysis (Average RMSE) ---")

    for hp in hp_cols:
        # Group by the hyperparameter and get the mean score
        hp_performance = results_df.groupby(hp)['score'].mean().sort_values()

        print(f"\n--- Performance by {hp} ---")
        print(hp_performance)

        # Plot and save
        plt.figure(figsize=(8, 4))
        hp_performance.plot(kind='bar')
        plt.title(f"Average RMSE by {hp}")
        plt.ylabel("Average RMSE")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"hp_analysis_{hp}.png")
        print(f"Saved plot to hp_analysis_{hp}.png")

    # # --- Heatmap for 2 most important parameters (e.g., lr vs. n_layers) ---
    # print("\n--- Generating Heatmap (lr vs. first_neurons) ---")
    # try:
    #     heatmap_data = results_df.pivot_table(
    #         index='lr',
    #         columns='first_neurons',
    #         values='score'
    #     )
    #     plt.figure(figsize=(10, 7))
    #     sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis_r")
    #     plt.title("Heatmap of Avg. RMSE (lr vs. first_neurons)")
    #     plt.savefig("hp_heatmap_lr_vs_neurons.png")
    #     print("Saved heatmap to hp_heatmap_lr_vs_neurons.png")
    # except Exception as e:
    #     print(f"Could not generate heatmap: {e}")



def train_val_test_split(x, y, test_size=0.2, val_size=0.2, random_state=42, stratify=False):
    """
    Splits data into train, validation, and test sets.

    This function first splits the data into a (train+val) set and a test set. Then, it splits the (train+val) set into
    the final train and validation sets.

    Arguments:
        x (pd.DataFrame): Input features
        y (pd.Series): Target values
        test_size (float, optional): Proportion of dataset to include in the test split. Defaults to 0.2.
        val_size (float, optional): Proportion of the (train+val) dataset to include in the val split. Defaults to 0.2.
        random_state (int, optional): Controls shuffling for reproducible results.
        stratify (bool, option): If True, data is split in a stratified fashion using y. Defaults to False.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    # Determine if stratification is needed
    stratify_array_1 = y if stratify else None

    # Split into (Train + Val) and Test split
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_array_1
    )

    # Determine stratification for the second split
    stratify_array_2 = y_train_full if stratify else None

    # Split (Train + Val) into Train and Val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_array_2
    )

    return x_train_full, x_train, x_val, x_test, y_train_full, y_train, y_val, y_test


def set_device():
    # 1. Check for NVIDIA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")

    # 2. Check for Apple Silicon GPU (local M1/M2/M3 Macs)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")

    # 3. Fallback to CPU
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")

    return device


def set_seed(seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch for reproducibility.
    """
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed for CPU
    torch.manual_seed(seed)

    # Set PyTorch's random seed for GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


def example_main():

    set_seed(42)
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Splitting into train, validation and test dataset
    x_train_full, x_train, x_val, x_test, y_train_full, y_train, y_val, y_test = train_val_test_split(
        x, y, test_size=0.2, val_size=0.2, random_state=42, stratify=False
    )

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting

    # Perform hyperparameter search to look for the best parameters
    best_params, all_results = perform_hyperparameter_search(x_train_full, y_train_full)

    # Plot results
    analyze_hp_search(all_results)

    # Build regressor model with best parameters from hyperparameter search
    regressor = Regressor(x_train,
                          learning_rate=best_params['lr'],
                          weight_decay=best_params['wd'],
                          n_hidden_layers=best_params['n_layers'],
                          first_layer_neurons=best_params['first_neurons'],
                          batch_size=best_params['bs'],
                          architecture_type=best_params['arch_type'],
                          nb_epoch=5000,
                          device=set_device(),
                          training=True
    )

    # Train the model
    regressor.fit(x_train, y_train, x_val, y_val, patience=25)
    save_regressor(regressor)

    # Print train/loss error
    regressor.plot_loss_history(save_path="training_validation_loss_curve.png")

    # Error
    error = regressor.score(x_test, y_test)
    print(f"\nRegressor error: {error}\n")


if __name__ == "__main__":
    example_main()



