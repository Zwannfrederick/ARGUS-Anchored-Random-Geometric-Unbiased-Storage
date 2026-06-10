import os
import logging
import sys
import inspect

# Get log level from environment variable (default: INFO)
env_log_level = os.environ.get("ARGUS_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, env_log_level, logging.INFO)

class ArgusFormatter(logging.Formatter):
    """Custom logging formatter that matches vLLM style format and supports overrides."""
    def format(self, record):
        filename = getattr(record, "override_filename", record.filename)
        lineno = getattr(record, "override_lineno", record.lineno)
        level = getattr(record, "override_level", record.levelname)
        
        # Format time to MM-DD HH:MM:SS
        asctime = self.formatTime(record, "%m-%d %H:%M:%S")
        
        # Color codes matching standard logging colors
        if level == "INFO":
            color_level = "\033[1;32mINFO\033[0m"
        elif level == "WARNING":
            color_level = "\033[1;33mWARNING\033[0m"
        elif level == "ERROR":
            color_level = "\033[1;31mERROR\033[0m"
        else:
            color_level = f"\033[1;34m{level}\033[0m"
            
        file_line = f"{filename}:{lineno}"
        argus_prefix = "\033[1;36m[ARGUS]\033[0m"
        message = record.getMessage()
        
        return f"{color_level} {asctime} {file_line:<22}] {argus_prefix} {message}"

# Create and configure the logger
logger = logging.getLogger("argus")
logger.setLevel(log_level)
logger.propagate = True

# Console handler redirecting to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(log_level)
console_handler.setFormatter(ArgusFormatter())
logger.addHandler(console_handler)

def argus_log(level: str, message: str, line_no: int = None):
    """
    Main logging function matching the signature expected by ARGUS.
    Uses inspect to capture caller context.
    """
    level_upper = level.upper()
    numeric_level = getattr(logging, level_upper, None)
    if numeric_level is None:
        # Custom levels fallback to INFO for standard processing
        numeric_level = logging.INFO
    
    # Inspect stack to find the actual caller frame
    frame = inspect.currentframe()
    caller_filename = "memory_manager.py"
    caller_lineno = line_no
    
    try:
        # Move up one frame to the caller
        caller_frame = frame.f_back
        if caller_frame:
            caller_filename = os.path.basename(caller_frame.f_code.co_filename)
            if caller_lineno is None:
                caller_lineno = caller_frame.f_lineno
    finally:
        del frame
        
    extra = {
        "override_filename": caller_filename,
        "override_lineno": caller_lineno,
        "override_level": level_upper
    }
    
    logger.log(numeric_level, message, extra=extra)
