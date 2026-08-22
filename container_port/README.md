# container_port

A discrete-event simulation (DES) of a two-slip Panamax container terminal,
built with [SimPy](https://simpy.readthedocs.io/). Companion starter project
to the [planet_sim](../README.md) ocean/atmosphere model, same "build the
thing, watch it in a browser" approach, much smaller scope.

**[Live visualization](../docs/container_port/index.html)** (or, once pushed,
`https://<user>.github.io/planet_sim/container_port/`).

## The model

```
                     +------------------+
   Anchorage  ---->  |  Slip 1 (2 cranes)|  ---->  departs
   (queue for a      +------------------+
    free slip)       |  Slip 2 (2 cranes)|
                      +------------------+
                             |  ^
                             v  |
                     [ shared truck pool ]
                             |  ^
                             v  |
                +-------------------------+
                |  Stacking yard (import) |
                |  Export-ready pool      |
                +-------------------------+
```

- **Ships** (Panamax-class) arrive as a Poisson process (mean interarrival
  26h) each carrying a random discharge and load count (TEU, triangular
  300-900-600).
- **2 slips**, modeled as an interchangeable pool of berth tokens — a ship
  grabs whichever is free, or queues at anchorage.
- **2 dedicated quay cranes per slip** (4 total). Cranes don't roam between
  slips (they're rail-bound, like the real thing). Both of a ship's cranes
  pull moves off the same shared discharge-then-load worklist, so they
  naturally split the work.
- **A shared truck pool** (12 trucks by default) ferries every single move
  between the quay and the yard/gate. This is deliberately the tightest
  resource in the default parameters (~75-90% utilized) — it's usually the
  first bottleneck worth finding, same as it often is at a real terminal.
- **Stacking yard**: discharge moves add to it; a background "customer
  pickup" process drains it. If the yard is full, discharge blocks — cranes
  idle waiting on yard space, not on trucks.
- **Export-ready pool**: a background "gate-in" process (inland trucks
  dropping export boxes ahead of a vessel call) fills it; load moves drain
  it. If it's empty, load moves block — a crane with nothing ready to load.

Everything stochastic here is exponential/triangular (classic M/M/c-style
DES), not calibrated to a real terminal. It's a playground: change
`PortParams` and watch what breaks.

## Run it

```bash
pip install -r requirements.txt   # just simpy

python port_model.py                        # text summary, default params
python port_model.py --days 30 --trucks 8   # starve the truck pool on purpose
```

Sample output (21 days, defaults):

```
ships_completed                          13
avg_turnaround_hours                  61.27
avg_anchorage_wait_hours              16.63
crane_utilization_pct                 34.07
truck_utilization_pct                 72.61
```

Cranes sit well under 50% utilized by design (real terminals oversize crane
capacity for peak vessel calls) while the shared truck pool runs hot — that
gap is the point: turnaround time here is gated by trucks and yard flow, not
by crane speed. Try `--trucks 20` and watch turnaround drop even though
nothing about the ships or cranes changed; try `--trucks 6` and watch the
anchorage queue back up.

## Rebuild the web visualization

```bash
python export_web_data.py --days 14        # runs the sim, writes port_data.json
python build_port_page.py                  # inlines it into ../docs/container_port/index.html
```

`port_data.json` is a timeline sampled every 15 simulated minutes (state of
every resource, every ship's progress) — the page just scrubs/plays through
those snapshots on an HTML5 canvas, no server, no dependencies. Commit the
regenerated `docs/container_port/index.html` to publish an update.

## A real modeling bug, for the DES-curious

The first working version of the crane/truck logic had one coroutine do
*both* the crane lift *and* wait out the truck's full round trip before
grabbing the next container. That accidentally serialized all 4 cranes onto
the truck cycle time — effective terminal throughput collapsed to roughly
`(number of active crane workers) / (truck round-trip time)`, independent of
how many trucks were actually in the pool. Symptom: ships piled up in the
anchorage queue for days, crane and truck utilization both showed as *low*
(because most work was stuck waiting on yard/export capacity upstream, not
because the resources were actually busy), and 15 of 21 arriving ships never
finished their port call in a 21-day run.

Fix: once the crane finishes its lift, it's free — the truck's return trip
now runs as its own independent process, so a crane can pick up its next
box immediately using a *different* free truck rather than waiting on its
own truck to come back. Classic DES lesson: a resource is only really free
when the code says it's free, and it's easy to accidentally chain two
resources' hold times together without meaning to. Worth staring at
`_do_move` / `_truck_return` in `port_model.py` if you want to see the
before/after shape of that fix.

## Layout

| File | Purpose |
|---|---|
| `port_model.py` | SimPy model: `PortParams`, `PortSim`, `run_simulation()`, `summary_stats()`; also a CLI (`python port_model.py`) |
| `export_web_data.py` | Runs the sim, samples a timeline, writes `port_data.json` |
| `port_template.html` | Visualization shell (vanilla Canvas 2D, no dependencies) |
| `build_port_page.py` | Inlines `port_data.json` into `port_template.html` → `../docs/container_port/index.html` |
