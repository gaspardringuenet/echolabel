import logging


def setUpLogging(loggerName, level=logging.INFO):
    """
    loggerName: __name__
    """
    logger = logging.getLogger(loggerName)
    logger.setLevel(level)
    formatter = logging.Formatter("%(levelname)s : %(name)s : %(message)s")

    # Sets up file handler
    fileHandler = logging.FileHandler(loggerName + ".log")
    fileHandler.setFormatter(formatter)

    # Sets up handler to std out
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(logging.DEBUG)

    # Adds handler to logger
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger
