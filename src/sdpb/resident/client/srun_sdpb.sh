#!/bin/bash
# lab (agent) bootstrap-speed #197, #238: hyperion's srun_sdpb.sh (stress-tensors-3d hyperion-config/expanse-dsd) as a
# client of a resident SDPB, one per checkpoint directory (resident_client.py next to this script). Same arguments as
# `srun sdpb`. RESIDENT_SDPB: an sdpb whose resident mode has solve "inherit" (lab/238-integrated-v2 or later). Opt-in:
# put this script and resident_client.py in the bound config's scriptsDir instead of the usual srun_sdpb.sh.
module purge
module load cpu/0.15.4 gcc/10.2.0 openmpi/4.0.4 gmp/6.1.2 mpfr/4.0.2 cmake/3.18.2 openblas/dynamic/0.3.7 boost/1.74.0 slurm
here=$(dirname "$(readlink -f "$0")")
export RESIDENT_SDPB=${RESIDENT_SDPB:?set RESIDENT_SDPB to the resident-capable sdpb}
echo "resident srun_sdpb.sh $@"
/usr/bin/python3 "$here/resident_client.py" "$@" || \
    (echo "Retrying in 60s" && sleep 60s && /usr/bin/python3 "$here/resident_client.py" "$@")
