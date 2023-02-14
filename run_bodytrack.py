import sysfs_paths as sysfs
import subprocess
import telnetlib as tel
import time
from timeit import default_timer as timer


def get_available_freqs(cluster):
    """
    obtain the available frequency for a cpu. return unit in khz by default
    return cpu from sysfs_paths.py
    """
    freqs = open(sysfs.fn_cluster_freq_range.format(cluster)).read().strip().split(' ')
    return [int(f.strip()) for f in freqs]



def get_cluster_freqs(core_num):
    """
    read the current frequency for a cpu. Return unit in khz by default!
    read cpu freq from sysfs_paths.py
    """
    with open(sysfs.fn_cluster_freq_read.format(core_num), 'r') as f:
        return int(f.read().strip())


def set_user_space(clusters=None):

    print("Setting userspace")
    clusters = [0, 4]
    for i in clusters:
        with open(sysfs.fn_cluster_gov.format(i), 'w') as f:
            f.write('userspace')


def set_cluster_freq(cluster_num, frequency):
    """
    set customized freq for a cluster. Accepts frequency in khz as int or string
    """
    with open(sysfs.fn_cluster_freq_set.format(cluster_num), 'w') as f:
        f.write(str(frequency))


print('available freq for little cluster:', get_cluster_freqs(0))
print('available freq for big cluster:', get_cluster_freqs(4))
set_user_space()
set_cluster_freq(4, 2000000)  # big cluster
# print current freq for the big cluster
print('current freq for big cluster:', get_cluster_freqs(4))

# execution of your benchmark
start = time.time()

# run the benchmark
command = "taskset --all-tasks 0xF0 /home/student/HW2_files/parsec_files/bodytrack /home/student/HW2_files/parsec_files/sequenceB_261 4 260 3000 8 3 4 0"  #0x20: core 5

proc_ben = subprocess.call(command.split())
total_time = time.time() - start
print("Benchmark runtime: ", total_time)
