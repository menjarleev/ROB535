import os

class Visualizer:
    log_path = "./"

    @staticmethod
    def log_print(log):
        print(log)
        with open(Visualizer.log_path, "a") as log_file:
            log_file.write('%s\n' % log)