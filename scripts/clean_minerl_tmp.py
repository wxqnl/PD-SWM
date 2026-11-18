"""Delete the copies of Minecraft that MineRL clients left in /tmp after a crash.

Each MineRL 0.4.4 client copies its Minecraft directory (~0.7 GB) to /tmp/tmp*/Minecraft and runs from there; a
client that dies with its run (a socket timeout, a relaunch) leaves the copy behind, 16 per crash of a run. This
removes the copies of the given user that no running client uses (its cwd) and that were not touched for
--min_age seconds (a client being started is copying into its directory). Stdlib only; run as root or the user.

usage: python3 scripts/clean_minerl_tmp.py [--user $USER] [--min_age 600] [--every 600]
"""
import argparse
import os
import pwd
import re
import shutil
import time


def in_use():
    dirs = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if b"MalmoMod" in open(f"/proc/{pid}/cmdline", "rb").read():
                dirs.add(os.readlink(f"/proc/{pid}/cwd").split("/Minecraft")[0])
        except OSError:
            pass
    return dirs


def clean(user, min_age):
    used = in_use()
    now = time.time()
    removed = 0
    for name in os.listdir("/tmp"):
        d = "/tmp/" + name
        if not re.fullmatch(r"/tmp/tmp[A-Za-z0-9_]+", d) or os.path.islink(d) or not os.path.isdir(d + "/Minecraft"):
            continue
        if pwd.getpwuid(os.stat(d).st_uid).pw_name != user or d in used:
            continue
        if now - max(os.stat(d).st_mtime, os.stat(d + "/Minecraft").st_mtime) < min_age:
            continue
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
    return len(used), removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=pwd.getpwuid(os.getuid()).pw_name)
    parser.add_argument("--min_age", type=int, default=600)
    parser.add_argument("--every", type=int, default=0, help="seconds between passes; 0 = one pass")
    args = parser.parse_args()
    while True:
        used, removed = clean(args.user, args.min_age)
        if removed:
            print(f"{time.strftime('%F %T')} removed {removed} stale Minecraft copies ({used} in use)", flush=True)
        if not args.every:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
