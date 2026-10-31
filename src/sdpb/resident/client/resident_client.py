"""Drop-in replacement for `srun sdpb <args>` that keeps one resident SDPB per checkpoint directory.

Lab bootstrap-speed #197 (first version), #238 (this version: exact file-route semantics, see "Exactness").

hyperion runs `srun_sdpb.sh <sdpb args>` once per SDPB call, after building the call's SDP (with #181's
`srun_pmp2sdp_reuse.sh`, pmp2sdp --reuseSdp, or a full pmp2sdp). With this client as srun_sdpb.sh:
  - the first call for a checkpoint directory starts `sdpb <args> --controlDir=<ck>.resident` in the background
    (detached; `srun --overlap`, so that later `srun -n 1 pmp2sdp` steps can start) and solves;
  - later calls send `update_sdp` (the blocks whose files changed, read from the call's SDP directory) and `solve`
    with the call's own arguments, then wait for the answer;
  - every call writes the final checkpoint, as sdpb does (unless --noFinalCheckpoint), so the checkpoint directory
    and the out directory end each call exactly as in the file route;
  - the resident process exits after RESIDENT_IDLE seconds without a call (default 600), when its job ends, or when
    the job starts a resident process for another checkpoint directory (RESIDENT_REGISTRY, default
    /tmp/sdpb-resident-$SLURM_JOB_ID, lists them): one resident process per job at a time.
The exit status is 0 if the solve ran, else 1 (hyperion's script retries, as with srun failures).

Exactness (#238; #211 found that the #197 client let later calls inherit the first call's switches):
  - Start-up: the resident starts with the first call's full arguments, so its start-up is sdpb's on that call: the
    block distribution (block_timings of the checkpoint it loads), the timing run if there is none, and the initial
    checkpoint (--initialCheckpointDir or --checkpointDir) or the initial point.
  - Options: every solve, the first included, is sent with "inherit": false (needs an sdpb whose ready.json lists the
    feature "solve_inherit", lab/238-integrated-v2 or later). A solve's options are then the call's own arguments
    only, as for an sdpb run; nothing carries over from the first call.
  - Blocks: before a solve the client re-reads the blocks whose block_data file in the call's SDP directory is not the
    file the resident holds (same device, inode, size and mtime). pmp2sdp --reuseSdp hard-links the unchanged blocks,
    so only the rewritten ones are re-read; a full pmp2sdp writes every file anew, so every block is re-read (a part
    changed under an unchanged path is not missed); a changed normalization.json re-reads all blocks.
  - Iterate: the resident solves from its iterate in memory only if that iterate is what sdpb would load: the call
    reads its checkpoint from this checkpoint directory (no --initialCheckpointDir elsewhere), the previous solve
    wrote its final checkpoint there, and checkpoint.json is still the file it wrote. Otherwise, or if an option fixed
    at start-up differs (--precision, --procGranularity, --maxSharedMemory, --sparseB), or the update is refused,
    the client quits the resident (without a checkpoint) and starts a new one with this call's arguments, as a new
    sdpb process would.
  - RESIDENT_CHECKPOINT_EVERY=N > 1 (not exact): checkpoint every N calls only; the iterate in memory is then ahead
    of the checkpoint directory, and the client keeps solving from it.

Environment: RESIDENT_LAUNCH (default "srun -v --overlap"), RESIDENT_SDPB (the sdpb binary), RESIDENT_IDLE,
RESIDENT_CHECKPOINT_EVERY, RESIDENT_REGISTRY. Python 3.6.
"""
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

FIXED = ["--precision", "--procGranularity", "--maxSharedMemory"]
FIXED_SWITCHES = ["--sparseB"]
BLOCK_FILE = re.compile(r"^block_data_(\d+)\.(bin|json)$")


def option(args, name, short=None):
    """Value of --name=value or --name value (or the short form) in args; the last one wins, as in sdpb."""
    value = None
    names = [name] + ([short] if short else [])
    for i, a in enumerate(args):
        for n in names:
            if a == n and i + 1 < len(args):
                value = args[i + 1]
            elif a.startswith(n + "="):
                value = a[len(n) + 1:]
    return value


def without(args, names):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in names:
            skip = True
            continue
        if any(a.startswith(n + "=") for n in names):
            continue
        out.append(a)
    return out


def file_id(path):
    st = os.stat(str(path))
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sdp_ids(sdp):
    """Identity of each block_data file of an SDP directory, and the normalization's hash."""
    blocks = {}
    for p in Path(sdp).iterdir():
        m = BLOCK_FILE.match(p.name)
        if m:
            blocks[m.group(1)] = file_id(p)
    norm = Path(sdp) / "normalization.json"
    return blocks, (sha256(norm) if norm.exists() else None)


def ck_id(ck):
    p = Path(ck) / "checkpoint.json"
    if not p.exists():
        return None
    return file_id(p) + [sha256(p)]


def fixed_values(args):
    values = {n: option(args, n) for n in FIXED}
    values.update({n: n in args for n in FIXED_SWITCHES})
    return values


def wait_file(path, proc_alive, timeout):
    start = time.time()
    sleep = 0.0005
    while not path.exists():
        if not proc_alive():
            raise RuntimeError("resident sdpb is not running (waiting for {})".format(path))
        if time.time() - start > timeout:
            raise RuntimeError("timeout waiting for {}".format(path))
        time.sleep(sleep)
        sleep = min(sleep * 2, 0.02)
    with open(str(path)) as f:
        return json.load(f)


class Client:
    def __init__(self, args):
        self.args = args
        self.ck = option(args, "--checkpointDir", "-c")
        self.sdp = option(args, "--sdpDir", "-s")
        if self.ck is None or self.sdp is None:
            raise SystemExit("resident_client: needs --checkpointDir and --sdpDir")
        self.ctl = Path(self.ck.rstrip("/") + ".resident")
        self.state_file = self.ctl / "client.json"
        self.launch = shlex.split(os.environ.get("RESIDENT_LAUNCH", "srun -v --overlap"))
        self.sdpb = os.environ["RESIDENT_SDPB"]
        self.idle = os.environ.get("RESIDENT_IDLE", "600")
        self.every = max(1, int(os.environ.get("RESIDENT_CHECKPOINT_EVERY", "1")))
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else None
        self.child = None  # the resident process, if this client started it

    def alive(self):
        if self.state is None:
            return False
        if self.child is not None:
            # our own child: a zombie still answers kill(pid, 0)
            if self.child.poll() is not None:
                return False
        else:
            try:
                os.kill(self.state["pid"], 0)
            except OSError:
                return False
        return not (self.ctl / "exit.json").exists()

    def save(self):
        self.state_file.write_text(json.dumps(self.state))

    def send(self, commands, timeout=10 * 86400):
        seq = self.state["seq"] + 1
        tmp = self.ctl / ".tmp.cmd.{}".format(seq)
        tmp.write_text(json.dumps({"commands": commands}))
        os.rename(str(tmp), str(self.ctl / "cmd.{}.json".format(seq)))
        res = wait_file(self.ctl / "res.{}.json".format(seq), self.alive, timeout)
        self.state["seq"] = seq
        self.save()
        return res

    def quit(self, why):
        """Stop this checkpoint directory's resident process, without a checkpoint (exact mode): the checkpoint
        directory keeps what the file route would have there."""
        print("resident_client: restarting the resident process: {}".format(why), flush=True)
        if self.alive():
            self.send([{"cmd": "quit", "checkpoint": self.every > 1}], timeout=1800)
            deadline = time.time() + 1800
            while not (self.ctl / "exit.json").exists() and time.time() < deadline:
                time.sleep(0.2)
        self.state = None

    def stop_others(self):
        """One resident process per job at a time: stop the others first, so that two SDPs and iterates never sit
        in memory together."""
        registry = Path(os.environ.get(
            "RESIDENT_REGISTRY",
            "/tmp/sdpb-resident-{}".format(os.environ.get("SLURM_JOB_ID", "local"))))
        registry.mkdir(parents=True, exist_ok=True)
        for entry in registry.iterdir():
            other = Path(entry.read_text().strip())
            if other == self.ctl or (other / "exit.json").exists():
                continue
            (other / "stop").write_text("")
            print("resident_client: stopping the resident process of {}".format(other), flush=True)
            deadline = time.time() + 1800
            while not (other / "exit.json").exists() and time.time() < deadline:
                time.sleep(0.5)
        (registry / hashlib.sha1(str(self.ctl).encode()).hexdigest()).write_text(str(self.ctl))

    def start(self):
        t0 = time.time()
        self.stop_others()
        self.ctl.mkdir(parents=True, exist_ok=True)
        for p in self.ctl.iterdir():
            if p.name.startswith(("cmd.", "res.", ".tmp")) or p.name in ("ready.json", "exit.json", "stop", "alive"):
                p.unlink()
        log = open(str(self.ctl / "sdpb.log"), "a")
        cmd = self.launch + [self.sdpb] + self.args + [
            "--controlDir={}".format(self.ctl), "--controlIdleTimeout={}".format(self.idle)]
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        self.child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        blocks, norm = sdp_ids(self.sdp)
        self.state = {"pid": self.child.pid, "seq": 0, "started": time.time(), "sdp": self.sdp, "blocks": blocks,
                      "norm": norm, "fixed": fixed_values(self.args), "ck": None, "ahead": False}
        self.save()
        ready = wait_file(self.ctl / "ready.json", self.alive, 7200)
        if "solve_inherit" not in ready.get("features", []):
            self.quit("this sdpb has no solve 'inherit' (needs lab/238-integrated-v2 or later)")
            raise SystemExit("resident_client: {} cannot solve without inheriting start-up options".format(self.sdpb))
        print("resident_client: started resident sdpb pid {} in {:.1f}s: {}".format(
            self.child.pid, time.time() - t0, ready), flush=True)

    def reuse_problem(self):
        """Why the resident's iterate and SDP structure can't be used for this call (None if they can)."""
        st = self.state
        if st["fixed"] != fixed_values(self.args):
            return "an option fixed at start-up changed: {} -> {}".format(st["fixed"], fixed_values(self.args))
        init = option(self.args, "--initialCheckpointDir", "-i")
        if init is not None and os.path.realpath(init) != os.path.realpath(self.ck):
            return "the call loads its checkpoint from {}".format(init)
        if st["ahead"]:
            return None
        if st["ck"] is None:
            return "the previous solve wrote no final checkpoint"
        if ck_id(self.ck) != st["ck"]:
            return "checkpoint.json changed since the previous solve"
        return None

    def run(self):
        if self.alive():
            why = self.reuse_problem()
            if why is not None:
                self.quit(why)
        commands = []
        if not self.alive():
            self.start()
            new_blocks, new_norm = self.state["blocks"], self.state["norm"]
        else:
            new_blocks, new_norm = sdp_ids(self.sdp)
            if new_norm != self.state["norm"] or set(new_blocks) != set(self.state["blocks"]):
                blocks = "all"
            else:
                blocks = sorted(int(b) for b in new_blocks if new_blocks[b] != self.state["blocks"][b])
            commands.append({"cmd": "update_sdp", "sdp": self.sdp, "blocks": blocks})
        no_final = "--noFinalCheckpoint" in self.args
        seq = self.state["seq"] + 1
        n_call = self.state.get("calls", 0) + 1
        checkpoint = (not no_final) and n_call % self.every == 0
        commands.append({"cmd": "solve", "args": without(self.args, ["--controlDir", "--controlIdleTimeout"]),
                         "inherit": False, "checkpoint": checkpoint})
        t1 = time.time()
        res = self.send(commands)
        results = res.get("results", [])
        errors = [r["error"] for r in results if "error" in r]
        if "error" in res and not errors:
            errors = [res["error"]]
        update_refused = commands[0]["cmd"] == "update_sdp" and (not results or "error" in results[0])
        if errors and update_refused and self.alive():
            # e.g. another block structure: a new sdpb process would read this SDP from scratch
            self.quit("update_sdp refused: {}".format(errors))
            return self.run()
        for r in results:
            if r.get("cmd") == "update_sdp" and "error" not in r:
                print("resident_client: call {} update_sdp blocks {}".format(seq, commands[0]["blocks"]), flush=True)
            if r.get("cmd") == "solve":
                print("resident_client: call {} {} steps {} in {:.1f}s (solve {:.1f}s)".format(
                    seq, r["terminateReason"], r["steps"], time.time() - t1, r.get("seconds", float("nan"))), flush=True)
        if errors:
            print("resident_client: error: {}".format(errors), flush=True)
            return 1
        self.state["blocks"], self.state["norm"], self.state["sdp"] = new_blocks, new_norm, self.sdp
        self.state["calls"] = n_call
        self.state["ck"] = ck_id(self.ck) if checkpoint else None
        self.state["ahead"] = (not checkpoint) and (not no_final)
        self.save()
        return 0


def main():
    sys.exit(Client(sys.argv[1:]).run())


if __name__ == "__main__":
    main()
