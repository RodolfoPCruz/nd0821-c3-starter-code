import logging


# Configure main application logger
app_logger = logging.getLogger("app_logger")
app_logger.setLevel(logging.INFO)
app_handler = logging.FileHandler("../logs/app.log")
app_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

# Configure test logger
test_logger = logging.getLogger("test_logger")
test_logger.setLevel(logging.DEBUG)
test_handler = logging.FileHandler("../logs/test.log")
test_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
test_handler.setFormatter(test_formatter)
test_logger.addHandler(test_handler)
