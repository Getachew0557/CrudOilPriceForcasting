import os
import logging

# Ensure the 'logs' directory exists
log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)

# Set up logging
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

# Create loggers
info_logger = logging.getLogger('info_logger')
error_logger = logging.getLogger('error_logger')

# Handlers for logging to files
info_handler = logging.FileHandler(log_file_info)
error_handler = logging.FileHandler(log_file_error)

# Handlers for logging to console (for Jupyter output)
console_handler = logging.StreamHandler()

# Set log levels
info_logger.setLevel(logging.INFO)
error_logger.setLevel(logging.ERROR)

# Define a common formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to loggers
info_logger.addHandler(info_handler)
info_logger.addHandler(console_handler)  # Logs INFO to console as well
error_logger.addHandler(error_handler)
error_logger.addHandler(console_handler)  # Logs ERROR to console as well
