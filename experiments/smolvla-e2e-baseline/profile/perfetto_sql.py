#!/usr/bin/env python3
"""Query a perfetto_ops trace with Perfetto's own trace_processor (step PROF.3).

Usage: python perfetto_sql.py <name>.perfetto_ops.json.gz [layer]
Needs `pip install perfetto` (downloads trace_processor_shell on first use). Prints the slice count per
process/track, then, for every op of the given layer (default 6) inside the trace, RedMulE and iDMA busy
as % of the op window averaged over 16 clusters, computed in Perfetto SQL from the official events only.
These numbers are the cross-check against parse_profile.py's ops.csv (PROFILE.md, expert_layer.csv).
"""
import sys
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace=sys.argv[1])
def show(it):
    for r in it: print(r.__dict__)

show(tp.query("""
select p.name as process, t.name as thread, count(*) n
from slice s join thread_track tt on s.track_id = tt.id join thread t using(utid) join process p using(upid)
group by 1, 2 order by 1, 2 limit 60"""))
# Expert layer 6 ops of this slice, busy % per engine averaged over 16 clusters, from Perfetto SQL alone
q = """
with ops as (
  select s.name op, s.ts t0, s.ts + s.dur t1, s.dur d, extract_arg(s.arg_set_id, 'args.layer') layer
  from slice s join thread_track tt on s.track_id = tt.id join thread t using(utid) join process p using(upid)
  where p.name like 'ops%' and t.name = 'op'),
ev as (
  select s.name agent, s.ts a0, s.ts + s.dur a1
  from slice s join thread_track tt on s.track_id = tt.id join thread t using(utid) join process p using(upid)
  where p.name like 'cluster_%' and t.name in ('redmule', 'idma'))
select op, d / 1000.0 as us,
  round(100.0 * sum(case when agent = 'redmule' then max(0, min(a1, t1) - max(a0, t0)) else 0 end) / 16 / d, 1) redmule_pct,
  round(100.0 * sum(case when agent = 'idma' then max(0, min(a1, t1) - max(a0, t0)) else 0 end) / 16 / d, 1) idma_pct
from ops join ev on a1 > t0 and a0 < t1
where layer = ?LAYER?
group by op, t0 order by t0
"""
show(tp.query(q.replace('?LAYER?', repr(sys.argv[2] if len(sys.argv) > 2 else '6'))))
tp.close()
