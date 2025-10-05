"""Constants for PINN solver module."""

# Default network parameters
DEFAULT_EPOCHS = 10000
DEFAULT_BATCH_SIZE = 1000
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DISPLAY_INTERVAL = 100

# Default loss weights
DEFAULT_WEIGHT_IC = 1000  # Initial conditions weight
DEFAULT_WEIGHT_BC = 10000  # Boundary conditions weight
DEFAULT_WEIGHT_EQ = 1000   # Equation weight

# Default regularization parameters
DEFAULT_LAMBDA_REG = 0.001
DEFAULT_L1_RATIO = 0.5  # For elastic regularization

# Default RAR parameters
DEFAULT_RAR_EPSILON = 0.05
DEFAULT_RAR_NUM_POINTS = 10
DEFAULT_RAR_INTERVAL = 50

# Default early stopping parameters
DEFAULT_EARLY_STOPPING_PATIENCE = 100

# Default SIREN parameters
DEFAULT_FIRST_OMEGA_0 = 30.0
DEFAULT_HIDDEN_OMEGA_0 = 30.0

# Device settings
DEFAULT_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"

# File extensions
MODEL_EXTENSION = ".pth"
CSV_EXTENSION = ".csv"
PNG_EXTENSION = ".png"
GIF_EXTENSION = ".gif"

# Plot settings
DEFAULT_FONT_SIZE = 14
DEFAULT_FIGURE_SIZE = (8, 8)
DEFAULT_DPI = 100

# Animation settings
DEFAULT_FPS = 10
DEFAULT_FRAMES = 50
