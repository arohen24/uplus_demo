import time
import datetime


def log(logstr, console_also=False):
    time_str = datetime.datetime.fromtimestamp(int(time.time())).strftime('[%d/%b/%Y %H:%M:%S]')
    with open('log.txt', 'a') as f:
        print(f"{time_str} {logstr}", file=f)
        if console_also:
            print(f"{time_str} {logstr}")
