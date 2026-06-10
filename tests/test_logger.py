import os
import sys
import logging
import pytest
from io import StringIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argus_cache.core.logger import argus_log, logger

def test_logger_format_and_levels(caplog):
    # Enable info level explicitly
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.setLevel(logging.INFO)
        
    with caplog.at_level(logging.INFO, logger="argus"):
        argus_log("INFO", "Testing logger info message")
        argus_log("RESEARCH", "Testing logger custom level message")
        
    assert len(caplog.records) == 2
    
    # Verify info message
    record_info = caplog.records[0]
    assert record_info.levelname == "INFO"
    assert record_info.message == "Testing logger info message"
    
    from argus_cache.core.logger import ArgusFormatter
    formatter = ArgusFormatter()
    formatted_info = formatter.format(record_info)
    assert "[ARGUS]" in formatted_info
    assert "INFO" in formatted_info
    assert "Testing logger info message" in formatted_info
    assert "test_logger.py" in formatted_info
    
    # Verify custom level message (e.g., RESEARCH)
    record_custom = caplog.records[1]
    # Check that custom level maps to INFO level internally
    assert record_custom.levelname == "INFO"
    assert record_custom.message == "Testing logger custom level message"
    
    formatted_custom = formatter.format(record_custom)
    assert "[ARGUS]" in formatted_custom
    # Formatter should override level output to RESEARCH
    assert "RESEARCH" in formatted_custom
    assert "Testing logger custom level message" in formatted_custom
    assert "test_logger.py" in formatted_custom

def test_logger_env_leveling(monkeypatch, capsys):
    # Set logging level to WARNING using monkeypatch on env
    monkeypatch.setenv("ARGUS_LOG_LEVEL", "WARNING")
    
    # Re-import or re-initialize logger settings from environment
    import importlib
    import argus_cache.core.logger
    importlib.reload(argus_cache.core.logger)
    
    # Try logging INFO (should be ignored) and WARNING (should be printed)
    argus_cache.core.logger.argus_log("INFO", "This info message should be ignored")
    captured = capsys.readouterr()
    assert "This info message should be ignored" not in captured.out
    
    argus_cache.core.logger.argus_log("WARNING", "This warning message should be printed")
    captured = capsys.readouterr()
    assert "This warning message should be printed" in captured.out
    assert "WARNING" in captured.out

    # Clean up environment reload to default INFO
    monkeypatch.delenv("ARGUS_LOG_LEVEL", raising=False)
    importlib.reload(argus_cache.core.logger)
