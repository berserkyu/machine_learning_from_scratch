import os

class Logger:
    def __init__(self, verbose=True, log_file=None):
        self._verbose = verbose
        self._log_file = log_file
        # create log file if not exists
        if log_file is not None:
            if not os.path.exists(log_file):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
        self._log_file_writer = open(log_file, 'a')

    def __del__(self): 
        self._log_file_writer.close()

    def set_log_file(seflf, log_file):
        self._log_file = log_file
        if log_file is not None:
            if not os.path.exists(log_file):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self._log_file_writer = open(log_file, 'a')

    def log(self, message):
        if self._verbose: 
            print(message)
        if self._log_file is not None:
            self._log_file_writer.write(message + '\n')
    
    def flush(self):
        if self._log_file is not None:
            self._log_file_writer.flush()

    def close(self):
        if self._log_file is not None:
            self._log_file_writer.close()

    
