import logging
from typing import Literal


def configure_logger(
    name,
    log_file="example.log",
    log_level=Literal["debug", "info", "error", "warning", "critical"],
):
    """
    Configures a logger with the specified name, log file, and log level.

    This function sets up a basic configuration for a logger, which includes setting the log level,
    creating a file handler, and defining a formatter for the log messages.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): The path to the log file. Defaults to "example.log".
        log_level (Literal["debug", "info", "error", "warning", "critical"], optional): 
            The minimum severity level of messages to be logged. Defaults to "info".

    Returns:
        logger: A configured logger object.

    Raises:
        NotImplementedError: If an unsupported log level is provided.
    """
    level = logging.DEBUG
    assert log_level in ["debug", "info", "error", "warning", "critical"]
    match log_level:
        case "debug":
            level = logging.DEBUG
        case "info":
            level = logging.INFO
        case "warning":
            level = logging.WARNING
        case "error":
            level = logging.ERROR
        case "critical":
            level = logging.CRITICAL
        case _:
            raise NotImplementedError(
                "not supported log level: {info}}".format(info=log_level)
            )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)

    logger.info("Logging Module config finished.")
    return logger
