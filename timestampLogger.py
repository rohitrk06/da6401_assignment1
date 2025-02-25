import datetime

class TimeStampLogger:
    '''
    A simple TimeStampLogger class that can be used to log messages with a timestamp.
    '''
    def __init__(self, stream):
        self.stream = stream
        
    def write(self, message):
        if message.strip():
            self.stream.write(f"[{datetime.datetime.now()}] {message}")
        else:
            self.stream.write(message)

    def flush(self):
        self.stream.flush()