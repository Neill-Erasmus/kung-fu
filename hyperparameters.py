class Hyperparameters:
    def __init__(self) -> None:
        self.learning_rate       : float = 1e-4
        self.gamma               : float = 0.99
        self.number_environments : int   = 10