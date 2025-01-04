from queue import Queue
from demo import myDetect
from inform import inform_you
import time
import numpy as np

if __name__ == '__main__':
    print("Welcome!")
    queue = Queue()
    detection = myDetect("detection", queue)
    inform = inform_you("inform", queue)
    detection.start()
    time.sleep(15)
    inform.start()
    detection.join()
    inform.join()

