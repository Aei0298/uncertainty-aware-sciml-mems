"""
Reusable PIML functions
"""

import torch
import torch.nn as nn


class AccelerationNetwork(nn.Module):
    """
    Feedforward neural network for predicting acceleration.

    Inputs
    ------
    displacement
    velocity
    excitation

    Output
    ------
    predicted acceleration
    """

    def __init__(
        self,
        input_size=3,
        hidden_size=32,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_size,
                1,
            ),
        )

    def forward(
        self,
        input_data,
    ):
        """
        Perform the forward prediction.
        """

        return self.network(
            input_data
        )


def train_data_driven_model(
    model,
    X_train,
    y_train,
    number_of_epochs=2000,
    learning_rate=1e-3,
    print_interval=400,
):
    """
    Train the neural network using only data-fitting loss.

    No physical equation is included in this loss.
    """

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_history = []


    for epoch in range(
        number_of_epochs
    ):

        # Set the model to training mode.
        model.train()


        # Remove gradients from the previous step.
        optimizer.zero_grad()


        # Predict acceleration.
        prediction = model(
            X_train
        )


        # Calculate data-fitting error.
        data_loss = loss_function(
            prediction,
            y_train,
        )


        # Calculate gradients.
        data_loss.backward()


        # Update model parameters.
        optimizer.step()


        # Store the loss for later analysis.
        loss_history.append(
            data_loss.item()
        )


        # Print occasional progress information.
        if (
            epoch == 0
            or (epoch + 1) % print_interval == 0
        ):
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Data loss = "
                f"{data_loss.item():.8e}"
            )


    return (
        model,
        loss_history,
    )

def calculate_physics_residual(
    predicted_acceleration,
    displacement,
    velocity,
    excitation,
    mass=1.0,
    damping=0.08,
    stiffness=1000.0,
):
    """
    Calculate the residual of the linear MEMS
    governing equation.

    The governing equation is

        m * acceleration
        + c * velocity
        + k * displacement
        - excitation
        = 0

    A physically consistent prediction should therefore
    produce a residual close to zero.
    """

    physics_residual = (
        mass * predicted_acceleration
        + damping * velocity
        + stiffness * displacement
        - excitation
    )

    return physics_residual


def train_physics_informed_model(
    model,
    X_train,
    y_train,
    X_collocation,
    collocation_physical,
    target_mean,
    target_scale,
    number_of_epochs=2000,
    learning_rate=1e-3,
    physics_weight=1.0,
    mass=1.0,
    damping=0.08,
    stiffness=1000.0,
    print_interval=400,
):
    """
    Train a neural network using both data loss and
    physics-consistency loss.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network that predicts standardized acceleration.

    X_train : torch.Tensor
        Standardized labelled training inputs.

    y_train : torch.Tensor
        Standardized labelled acceleration targets.

    X_collocation : torch.Tensor
        Standardized collocation-point inputs.

    collocation_physical : torch.Tensor
        Physical collocation values with columns

        [displacement, velocity, excitation].

    target_mean : float
        Mean used to standardize acceleration.

    target_scale : float
        Standard deviation used to standardize acceleration.

    physics_weight : float
        Relative weight of the physics residual in the
        total training loss.
    """

    data_loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )


    total_loss_history = []
    data_loss_history = []
    physics_loss_history = []


    for epoch in range(
        number_of_epochs
    ):

        model.train()

        optimizer.zero_grad()


        
        # DATA LOSS
        labelled_prediction_scaled = model(
            X_train
        )

        data_loss = data_loss_function(
            labelled_prediction_scaled,
            y_train,
        )


        # PHYSICS LOSS
        
        # Predict acceleration at the collocation points.
        collocation_prediction_scaled = model(
            X_collocation
        )


        # Convert standardized acceleration prediction back
        # into physical acceleration units.
        collocation_prediction_physical = (
            collocation_prediction_scaled
            * target_scale
            + target_mean
        )


        # Extract physical state variables.
        displacement = (
            collocation_physical[:, 0:1]
        )

        velocity = (
            collocation_physical[:, 1:2]
        )

        excitation = (
            collocation_physical[:, 2:3]
        )


        # Governing-equation residual:
        #
        # m*a + c*v + k*x - F = 0
        physics_residual = (
            mass
            * collocation_prediction_physical

            + damping
            * velocity

            + stiffness
            * displacement

            - excitation
        )


        # Mean squared physical residual.
        physics_loss = torch.mean(
            physics_residual ** 2
        )


        # TOTAL LOSS

        total_loss = (
            data_loss
            + physics_weight
            * physics_loss
        )


        total_loss.backward()

        optimizer.step()


        total_loss_history.append(
            total_loss.item()
        )

        data_loss_history.append(
            data_loss.item()
        )

        physics_loss_history.append(
            physics_loss.item()
        )


        if (
            epoch == 0
            or (epoch + 1) % print_interval == 0
        ):
            print(
                f"Epoch {epoch + 1:4d} | "
                f"Total = {total_loss.item():.8e} | "
                f"Data = {data_loss.item():.8e} | "
                f"Physics = {physics_loss.item():.8e}"
            )


    return (
        model,
        total_loss_history,
        data_loss_history,
        physics_loss_history,
    )