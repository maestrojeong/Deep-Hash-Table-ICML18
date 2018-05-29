import logging
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S" )

class LoggerManager(object):
    def __init__(self, filepath, name="logger"):
        self.logger = logging.getLogger(name)
        self.logger.addHandler(logging.FileHandler(filepath))
    
    def info(self, string):
        self.logger.info(string)

    def remove(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        del self.logger
