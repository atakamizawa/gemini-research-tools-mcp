"""Dify Plugin Entry Point for Gemini Research Tools."""

import logging
import os

from dify_plugin import DifyPluginEnv, Plugin
from dify_plugin.config.logger_format import plugin_logger_handler

# Configure logging
log_level = os.environ.get("DIFY_PLUGIN_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

# Set up root logger with Dify plugin handler
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, log_level, logging.INFO))
root_logger.addHandler(plugin_logger_handler)

# Create module logger
logger = logging.getLogger(__name__)
logger.info(f"Dify Plugin starting with log level: {log_level}")

# Deep Research can take several minutes, so we set a longer timeout
plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=3600))


if __name__ == "__main__":
    logger.info("Starting Gemini Research Tools plugin...")
    plugin.run()
