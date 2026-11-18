"""Keep the MineRL Minecraft clients (JVMs) of a user off a set of CPUs, so the training processes always get CPU.

Malmo's synchronous tick waits by yielding in a loop, so every client keeps its server and client threads
runnable even with no work (~750k context switches per second per JVM). With 8 runs of 16 clients on 128 CPUs
this crowds out the dreamer.py / R2I processes that drive the GPUs. This sets the CPU affinity of every thread of
every MalmoMod JVM of the user to --java_cpus; new threads inherit it, new JVMs are caught on the next pass.
Only placement changes, not the game. Stdlib only.

usage: python3 scripts/pin_minerl_cpus.py [--user $USER] [--java_cpus 8-31,40-63,72-95,104-127] [--every 30]
"""
import argparse
import os
import pwd
import time


def parse(spec):
    cpus = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def pin(uid, cpus):
    jvms = threads = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            if b"MalmoMod" not in open(f"/proc/{pid}/cmdline", "rb").read():
                continue
            jvms += 1
            for tid in os.listdir(f"/proc/{pid}/task"):
                if os.sched_getaffinity(int(tid)) != cpus:
                    os.sched_setaffinity(int(tid), cpus)
                    threads += 1
        except (OSError, ValueError):
            pass
    return jvms, threads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=pwd.getpwuid(os.getuid()).pw_name)
    parser.add_argument("--java_cpus", default="8-31,40-63,72-95,104-127")  # 8 cores per socket left to Python
    parser.add_argument("--every", type=int, default=0, help="seconds between passes; 0 = one pass")
    args = parser.parse_args()
    uid = pwd.getpwnam(args.user).pw_uid
    cpus = parse(args.java_cpus)
    while True:
        jvms, threads = pin(uid, cpus)
        if threads:
            print(f"{time.strftime('%F %T')} pinned {threads} threads of {jvms} JVMs to {len(cpus)} CPUs", flush=True)
        if not args.every:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
