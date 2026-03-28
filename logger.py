# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

import logging
import sys

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for better readability"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname:8}{reset}"
        record.name = f"\033[34m{record.name:20}\033[0m"
        return super().format(record)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the orchestrator"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(
        '%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    root_logger = logging.getLogger("INFRA-SKILL")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    root_logger.propagate = False
    
    return root_logger