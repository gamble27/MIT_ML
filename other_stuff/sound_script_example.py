import os
import time

time.sleep(5)

# sudo apt install sox
duration = 1  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

time.sleep(5)

# sudo apt install speech-dispatcher
os.system('spd-say "your program has finished"')
