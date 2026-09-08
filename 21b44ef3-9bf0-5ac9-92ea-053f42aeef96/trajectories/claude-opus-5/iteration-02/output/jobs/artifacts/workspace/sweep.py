"""Run many training configs in parallel subprocesses and tabulate results."""
import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def param_count(d, blocks, seq=10):
    n = 10 * d + 1  # embedding (tied to unembedding) + logit scale
    for (h, k, m) in blocks:
        if h > 0:
            n += 4 * d * h * k + h * k + h * seq
        if m > 0:
            n += 2 * d * m + m
    return n


def launch(job):
    tag = job["tag"]
    cmd = [sys.executable, os.path.join(HERE, "train.py"), "--tag", tag]
    for k, v in job.items():
        if k == "tag":
            continue
        cmd += ["--" + k, str(v)]
    log = os.path.join(RUNS, tag + ".log")
    os.makedirs(RUNS, exist_ok=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return tag, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="json file with a list of jobs")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    jobs = json.load(open(a.spec))
    print("%d jobs, %d workers" % (len(jobs), a.workers), flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for tag, rc in ex.map(launch, jobs):
            print("done", tag, "rc", rc, flush=True)


if __name__ == "__main__":
    main()
