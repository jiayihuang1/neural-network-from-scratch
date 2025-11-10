import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


class NeuralNet(nn.Module):

    def __init__(self, input_layer_neurons, n_hidden_layers=2, first_layer_neurons=64):
        """ Initialises a flexible network.

            Args:
                layer_sizes (list): A list of layer sizes, starting with the
                                    input size and ending with the output size.
                                    e.g., [9, 64, 32, 1]
        """ ### FULLY CONNECTED LAYER ARCHITECTURE

        super().__init__()

        # Input layer is the size of the input features
        layer_sizes = [input_layer_neurons]

        # Add first hidden layer
        current_neurons = first_layer_neurons
        layer_sizes.append(current_neurons)

        # Add the rest of the hidden layers, decaying each time
        for _ in range(n_hidden_layers - 1):
            current_neurons = max(1, int(current_neurons / 2))
            layer_sizes.append(current_neurons)

        layer_sizes.append(1)

        print(f"Building network with architecture: {layer_sizes}")
        layers = []
        # Loop until the second-to-last item
        for i in range(len(layer_sizes) - 2):
            # Add a Linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # Add a ReLU
            layers.append(nn.ReLU())

        # Add the final Linear layer (without ReLU)
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.linear_relu_stack = nn.Sequential(*layers)


    def forward(self, x):
        return self.linear_relu_stack(x)


class Regressor:

    def __init__(self, x,
                 loss_fun="mse",
                 learning_rate=0.01,
                 weight_decay=0.0,
                 n_hidden_layers=2,
                 first_layer_neurons=64,
                 nb_epoch = 1000,
                 device=None,
                 batch_size=64,
                 lr_decay_step=100,
                 lr_decay_gamma=1.0
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
        # To store parameters for preprocessing generated from training data
        column_headers = x.columns
        row_labels = ['mean', 'median', 'mode']
        self.preprocessing_params = pd.DataFrame(columns=column_headers, index=row_labels)
        self.lb = preprocessing.LabelBinarizer()

        self.nb_epoch = nb_epoch
        self.loss_fun = loss_fun
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.network = None
        self.optimizer = None
        self.n_hidden_layers = n_hidden_layers
        self.first_layer_neurons = first_layer_neurons
        self.batch_size = batch_size
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma
        self.scheduler = None
        if device is None:
            self.device = torch.device("cpu")
            print("Warning: No device specified, using CPU.")
        else:
            self.device = device

        # Initialise storage for loss history
        self.train_loss_history = []
        self.val_loss_history = []

        # To store loss function
        if self.loss_fun == "mse":
            self._loss_layer = torch.nn.MSELoss()
        elif self.loss_fun == "cross_entropy":
            self._loss_layer = torch.nn.CrossEntropyLoss()
        else:
            print(f"Invalid loss function input, please provide either {'mse'} or {'cross_entropy'} as loss function inputs")

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
            x  = x[target_present_mask]
            y = y[target_present_mask]

        cat_col = "ocean_proximity"
        # If we are training the model, generate pre-processing parameters from training data
        if training:

            # Handle categorical features
            # Store the node and fit the binarizer
            self.preprocessing_params.loc["mode", cat_col] = x[cat_col].mode()[0]
            x[cat_col] = x[cat_col].fillna(value=self.preprocessing_params.loc["mode", cat_col])
            self.lb.fit(x["ocean_proximity"])


            # Handle numerical features
            for column in x.columns:
                # Skip the categorical columns
                if column == cat_col:
                    continue

                # Generate and store mean and median values generated from training data
                self.preprocessing_params.loc["mean", column] = np.mean(x[column])
                self.preprocessing_params.loc["median", column] = np.median(x[column])

        # Process categorical features with parameters from training data
        # Fill NA values with the stored node
        x[cat_col] = x[cat_col].fillna(value=self.preprocessing_params.loc["mode", cat_col])

        # Transform the column with stored fit for one-hot encoding
        one_hot_cols = self.lb.transform(x[cat_col])
        one_hot_df = pd.DataFrame(one_hot_cols, columns=self.lb.classes_, index=x.index)
        x = x.drop(columns=cat_col)
        x = x.join(one_hot_df)

        # Process numerical features with parameters from training data
        for column in x.columns:
            # Skip the categorical columns that were just added
            if column in self.lb.classes_:
                continue

            # Fill NaN values with "mean"
            x[column] = x[column].fillna(value=self.preprocessing_params.loc["mean", column])

            # Normalise by dividing all values with "mean"
            x[column] = x[column] / self.preprocessing_params.loc["mean", column]

        # Convert preprocessed input features to tensors
        x_tensor = self.df_to_tensor(x)

        # # Preprocess target values
        # y_tensor = None
        # if y is not None:
        #     # Apply pre-processing to targets
        #     y = np.log1p(y)  # Log-transform, hence no parameter needed from training datass@

        y_tensor = self.df_to_tensor(y)

        # Return preprocessed x and y, return None for y if it was None
        return x_tensor, (y_tensor if isinstance(y, pd.DataFrame) else None)

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

        if self.network is None:

            # Get the input shape after processing features
            processed_input_shape = X_train_processed.shape[1]

            # Create network and optimizer
            self.network = NeuralNet(processed_input_shape, self.n_hidden_layers, self.first_layer_neurons).to(self.device)
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.lr_decay_step,
                gamma=self.lr_decay_gamma
            )

        # Load training data in by batches
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
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

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
                    X_val_processed = X_val_processed.to(self.device)
                    Y_val_processed = Y_val_processed.to(self.device)

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
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    epochs_no_improve = 0
                    torch.save(self.network.state_dict(), 'best_model.pth')
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {n+1}")
                    break

                self.scheduler.step()

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

        # Invert the log-transform, np.expm1 is the inverse of np.log1p
        # Y_prediction_unscaled = torch.expm1(Y_prediction_log)

        return Y_prediction

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
        y_pred = self.predict(x).detach().cpu().numpy()

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


    def df_to_tensor(self, df):
        return torch.tensor(df.values).float()


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


def perform_hyperparameter_search(x_train, y_train, x_val, y_val):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """ ### ADD K-FOLD, PRINT TOP 5 PARAM COMBOS, BATCH_SIZE // 2

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    print("\n--- Starting Hyperparameter Search ---")

    # Define search space for hyperparameters
    learning_rates = [0.0001, 0.001, 0.01]
    weight_decays = [0.0, 0.001]
    n_hidden_layers = [2, 3]
    first_layer_neurons = [128, 64, 32]
    batch_sizes = [50, 100]
    lr_gammas = [0.9, 0.5]

    best_score = float('inf')
    best_params = {}
    results = []

    # Simple grid search
    for lr in learning_rates:
        for wd in weight_decays:
            for n_layers in n_hidden_layers:
                for first_neurons in first_layer_neurons:  # New loop
                    for bs in batch_sizes:
                        for gamma in lr_gammas:

                            print(f"Testing params: lr={lr}, wd={wd}, layers={n_layers}, first_neurons={first_neurons}, batch={bs}, gamma={gamma}")

                            # Create model
                            regressor = Regressor(
                                x_train,
                                loss_fun="mse",
                                learning_rate=lr,
                                weight_decay=wd,
                                n_hidden_layers=n_layers,
                                first_layer_neurons=first_neurons,
                                nb_epoch=200,
                                device=set_device(),
                                batch_size=bs,
                                lr_decay_gamma=gamma,
                                lr_decay_step=100
                            )

                            # Train model
                            regressor.fit(x_train, y_train, x_val, y_val, patience=10)

                            # Score on validation set
                            score = regressor.score(x_val, y_val, print_metrics=False)

                            print(f"Validation RMSE: {score:.2f}")
                            results.append({
                                 'lr': lr,
                                 'wd': wd,
                                 'n_layers': n_layers,
                                 'first_neurons': first_neurons,
                                 'bs': bs,
                                 'gamma': gamma,
                                 'score': score
                            })

                            if score < best_score:
                                best_score = score
                                best_params = {
                                    'lr': lr,
                                    'wd': wd,
                                    'n_layers': n_layers,
                                    'first_neurons': first_neurons,
                                    'bs': bs,
                                    'gamma': gamma,
                                    'score': score,
                                    'regressor': regressor
                                }

    print("\n--- Hyperparameter Search Complete ---")
    print(f"Best Validation RMSE: {best_score:.2f}")
    print(f"Best Hyperparameters: {best_params}")

    # Plot results
    plt.figure(figsize=(12, 6))
    param_labels = [
        f"lr={r['lr']}, L={r['n_layers']}, N={r['first_neurons']}, bs={r['bs']}"
        for r in results
    ]
    scores = [r['score'] for r in results]

    plt.bar(param_labels, scores)
    plt.xticks(rotation=90)
    plt.title("Hyperparameter Search Results (Validation RMSE)")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig("hyperparameter_search_results.png")
    print("Saved hyperparameter search results plot.")


    # Return the best params
    return best_params # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


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
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_array_1
    )

    # Determine stratification for the second split
    stratify_array_2 = y_train_val if stratify else None

    # Split (Train + Val) into Train and Val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_array_2
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def set_device():
    # setting the device to mps for mac OS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU via MPS")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")

    return device


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Splitting into train, validation and test dataset
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        x, y, test_size=0.2, val_size=0.2, random_state=42, stratify=False
    )

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting

    # Perform hyperparameter search to look for the best parameters
    best_params = perform_hyperparameter_search(x_train, y_train, x_val, y_val)

    # Build regressor model with best parameters from hyperparameter search
    regressor = Regressor(x_train,
                          loss_fun = "mse",
                          learning_rate=best_params['lr'],
                          weight_decay=best_params['wd'],
                          n_hidden_layers=best_params['n_layers'],
                          first_layer_neurons=best_params['first_neurons'],
                          batch_size=best_params['bs'],
                          lr_decay_gamma=best_params['gamma'],
                          nb_epoch=5000,
                          device=set_device()
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



