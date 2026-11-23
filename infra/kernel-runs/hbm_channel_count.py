#!/usr/bin/env python3
"""Count HBM requests and bytes per channel from a GVSoC trace of the HBM controllers.

    gvsoc ... --trace=hbm_ctrl | hbm_channel_count.py <arch.py> [t_from_ns t_to_ns] > hbm_channels.txt

The optional window keeps only requests inside [t_from, t_to] ns (e.g. the timed region read off the
Perfetto trace), leaving out the preload and the output check.

The controller trace (soft_hier/flex_cluster/hbm_ctrl.cpp, req_muxed) prints the offset, the
input id and the size of each request, but not the channel. The channel is recomputed with the
controller's own rule: offset += (id / aliase) * node_size * aliase; bank = (offset >> ilv) & mask,
then XOR scrambling over the higher bits. Red scrambling is not modelled (asserted off).
"""
import collections
import importlib.util
import math
import re
import sys

spec = importlib.util.spec_from_file_location("arch", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
arch = mod.FlexClusterArch()
assert getattr(arch, "hbm_ctrl_red_scrambling", 0) == 0, "red scrambling not modelled"

ilv = int(math.log2(arch.noc_link_width / 8))
aliase = getattr(arch, "hbm_node_aliase", 1)
node_size = arch.hbm_node_addr_space
xor = getattr(arch, "hbm_ctrl_xor_scrambling", 0)
edges = ("west", "north", "east", "south")
nodes_per_edge = {"west": arch.num_cluster_y, "east": arch.num_cluster_y,
                  "north": arch.num_cluster_x, "south": arch.num_cluster_x}

pat = re.compile(r"/chip/(\w+)_hbm_ctrl_(\d+)/trace.*?Received IO req \(offset: 0x([0-9a-f]+), id: (\d+), "
                 r"size: 0x([0-9a-f]+), is_write: (\d)\)")
reqs = collections.Counter()
nbytes = collections.Counter()
t_from = float(sys.argv[2]) if len(sys.argv) > 3 else -1.0
t_to = float(sys.argv[3]) if len(sys.argv) > 3 else float("inf")
for line in sys.stdin:
    m = pat.search(line)
    if not m:
        continue
    t_ns = int(line.split(":", 1)[0]) / 1000.0  # first field: time in ps
    if not t_from <= t_ns <= t_to:
        continue
    edge, ctrl, off, mux, size, wr = m.group(1), int(m.group(2)), int(m.group(3), 16), int(m.group(4)), int(m.group(5), 16), m.group(6)
    n_ctrl = nodes_per_edge[edge] // arch.num_node_per_ctrl
    nb_slaves = arch.hbm_chan_placement[edges.index(edge)] // n_ctrl
    stage = int(math.log2(nb_slaves)) if nb_slaves > 1 else 0
    mask = (1 << stage) - 1
    # mux id is already divided by aliase in the trace (printed after the translation)
    offset = off + mux * node_size * (aliase if aliase > 1 else 1)
    bank = (offset >> ilv) & mask
    if xor and stage > 0:
        tmp = offset >> (stage + ilv)
        for _ in range((32 - ilv - 1) // stage):
            bank ^= tmp
            tmp >>= stage
        bank &= mask
    key = (edge, ctrl, bank, "W" if wr == "1" else "R")
    reqs[key] += 1
    nbytes[key] += size

total = sum(nbytes.values())
print(f"{'edge':<6} {'ctrl':>4} {'chan':>4} {'R/W':>3} {'requests':>10} {'bytes':>12} {'share':>6}")
for key in sorted(nbytes):
    print(f"{key[0]:<6} {key[1]:>4} {key[2]:>4} {key[3]:>3} {reqs[key]:>10} {nbytes[key]:>12} {100 * nbytes[key] / max(total, 1):>5.1f}%")
chans = {(e, c, b) for e, c, b, _ in nbytes}
print(f"channels used: {len(chans)}; total bytes: {total}")
