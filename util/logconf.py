import logging
import logging.handlers

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Some libraries attempt to add their own root logger handlers. This is
# annoying and so we get rid of them.
# I commented this part. Is better to not modify the iterable object
# of a for loop inside it, the code below should do the trick.
# for handler in list(root_logger.handlers):
#     root_logger.removeHandler(handler)
root_logger.handlers = []

logfmt_str = ("%(asctime)s %(levelname)-8s pid:%(process)d %(name)s"
              ":%(lineno)03d:%(funcName)s %(message)s")
formatter = logging.Formatter(logfmt_str)

# StreamHandler is located in the core logging package, sends logging
# output to streams
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.DEBUG)

root_logger.addHandler(streamHandler)
