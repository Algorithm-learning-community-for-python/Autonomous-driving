#NB: NOT WORKING
from subprocess import Popen, PIPE
import os
FNULL = open(os.devnull, "w")
import time
for i in range(1):
    print("Starting Carla...")
    carla = Popen(["DISPLAY= ../CarlaUE4.sh /Game/Carla/Maps/Town01"], stdout=FNULL, shell=True)
    #(out1,err1) = carla.communicate()
    print("Waiting for carla to get ready...")

    time.sleep(6)
    print("Starting recorder")

    recorder = Popen(["python", "recorder.py"], stdout=PIPE)
    print("Recorder started. waiting for it to finish")

    recorder.wait()
    print("Recorder finished. Killing subprocesses...")

    carla.kill()
#    runpy.run_module('recorder', run_name='__main__')