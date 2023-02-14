import psutil
import telnetlib as tel
import sysfs_paths as sysfs
import time


def getTelnetPower(telnet_connection, last_power):
    """
    read power values using telnet.
    """
    # Get the latest data available from the telnet connection without blocking
    tel_dat = str(telnet_connection.read_very_eager())
    print('telnet reading:', tel_dat)
    # find latest power measurement in the data
    idx = tel_dat.rfind('\n')
    idx2 = tel_dat[:idx].rfind('\n')
    idx2 = idx2 if idx2 != -1 else 0
    ln = tel_dat[idx2:idx].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power


def getCpuLoad():
    """
    Returns the cpu load as a value from the interval [0.0, 1.0]
    """
    loads = [x / 100 for x in psutil.cpu_percent(interval=None, percpu=True)]
    return loads




def getTemps():
    """
    obtain the temp values from sysfs_paths.py
    """
    templ = []
    # get temp from temp zones 0-3 (the big cores)
    for i in range(4):
        temp = float(open(sysfs.fn_thermal_sensor.format(i), 'r').readline().strip()) / 1000
        templ.append(temp)
    # Note: on the 5422, cpu temperatures 5 and 7 (big cores 1 and 3, counting from 0) appear to be swapped. Therefore, swap them back.
    t1 = templ[1]
    templ[1] = templ[3]
    templ[3] = t1
    return templ


def getCoreFreq(core_num):
    with open(sysfs.fn_cluster_freq_read.format(core_num), 'r') as f:
        return int(f.read().strip())


# create a text file to log the results
out_fname = 'log.txt'
header = "time W usage_c0 usage_c1 usage_c2 usage_c3 usage_c4 usage_c5 usage_c6 usage_c7 temp4 temp5 temp6 temp7"
header = "\t".join(header.split(' '))
out_file = open(out_fname, 'w')
out_file.write(header)
out_file.write("\n")

# measurement
telnet_connection = tel.Telnet("192.168.4.1")
total_power = 0.0

start_time = time.time()
print("Waiting for cpu to start at time:", start_time)

count = 0
while(True):
    usages = getCpuLoad()
    if usages[4] > .1 or usages[5] > .1 or usages[6] > .1 or usages[7] > .1:
        count +=1
    else:
        count = 0
    if count > 1:
        start_time = time.time()
        print("CPU ACTIVE... start time:", start_time)
        break
    time.sleep(5)


count = 0

while count < 10:
    last_time = time.time()  # time_stamp
    # system power
    total_power = getTelnetPower(telnet_connection, total_power)
    print('telnet power:', total_power)
    # cpu load
    usages = getCpuLoad()
    print('cpu usage:', usages)
    # temp for big cores
    temps = getTemps()
    print('temp of big cores:', temps)

    #big cluster core frequencies
    freq = getCoreFreq(4)
    time_stamp = last_time


    # Data writeout:
    fmt_str = "{}\t" * 14
    out_ln = fmt_str.format(time_stamp, total_power, \
                            usages[0], usages[1], usages[2], usages[3], \
                            usages[4], usages[5], usages[6], usages[7], \
                            temps[0], temps[1], temps[2], temps[3])

    out_file.write(out_ln)
    out_file.write("\n")

    if usages[4] < .15 and usages[5] < .15 and usages[6] <.15 and usages[7] < .15:
        count += 1
    else:
        count = 0

    elapsed = time.time() - last_time
    DELAY = 0.2
    time.sleep(max(0, DELAY - elapsed))
end_time = time.time()
print("CPU INACTIVE... end time:", end_time)
print("elapsed time:", end_time - start_time)
fmt_str = "{}\t" * 2
out_ln = fmt_str.format(start_time, end_time)
out_file.write(out_ln)
out_file.write("\n")

