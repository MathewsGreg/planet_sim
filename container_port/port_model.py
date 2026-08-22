"""
container_port/port_model.py

A discrete-event simulation of a two-slip container terminal:

  anchorage -> slip (berth) -> 2 dedicated quay cranes per slip -> yard

Panamax-class containerships arrive, queue for one of two slips if both
are occupied, then are worked by that slip's two cranes in parallel.
Every crane move (discharge or load) needs a drayage truck to carry the
box to/from the stacking yard, and a shared truck pool feeds both slips
-- that shared pool is usually the tightest resource in the system, on
purpose, so it's the first bottleneck worth finding.

Discharge moves need room in the yard; load moves need an already
gated-in export box waiting in the yard. Two background processes model
the "far side" of the yard: customers picking up import boxes, and
inland trucks gating in export boxes ahead of a vessel's arrival. Both
are simple memoryless (Poisson) processes, same as everything else here
-- this is a classic M/M/c-flavored model, not a calibrated real
terminal, and every parameter below is meant to be poked at.

Run directly (`python port_model.py`) for a text summary. Import
`run_simulation()` from `export_web_data.py` to also get a sampled
timeline for the web visualization.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import simpy

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class PortParams:
    # -- overall run --
    sim_days: float = 21.0
    random_seed: int = 42
    sample_interval_hours: float = 0.25  # snapshot cadence for the web viz

    # -- ship arrivals (Poisson process) --
    ship_interarrival_mean_hours: float = 26.0

    # -- cargo per port call (triangular: min, mode, max) --
    discharge_teu_min: float = 300.0
    discharge_teu_mode: float = 600.0
    discharge_teu_max: float = 900.0
    load_teu_min: float = 300.0
    load_teu_mode: float = 600.0
    load_teu_max: float = 900.0

    # -- terminal layout --
    n_slips: int = 2
    cranes_per_slip: int = 2
    n_trucks: int = 12

    # -- service times (memoryless / exponential means) --
    crane_move_minutes: float = 2.2       # ~27 moves/hr per crane
    truck_round_trip_minutes: float = 12.0  # quay <-> yard/gate and back

    # -- yard --
    yard_capacity_teu: float = 2500.0
    export_pool_capacity_teu: float = 1500.0
    yard_pickup_mean_minutes: float = 2.0   # customer collects an import box
    gate_in_mean_minutes: float = 2.0       # inland truck drops an export box


SHIP_NAMES = [
    "MV Pacific Vanguard", "MV Meridian Star", "MV Cape Falcon", "MV Nordic Trader",
    "MV Golden Horizon", "MV Blue Ensign", "MV Southern Cross", "MV Coral Voyager",
    "MV Atlas Bridge", "MV Harbor King", "MV Silver Current", "MV Amber Tide",
    "MV Iron Compass", "MV Northern Wake", "MV Crimson Pilot", "MV Sea Aster",
]


@dataclass
class ShipRecord:
    id: int
    name: str
    discharge_teu: int
    load_teu: int
    arrival_time: float
    slip_assigned_time: Optional[float] = None
    slip_id: Optional[int] = None
    departure_time: Optional[float] = None

    @property
    def anchorage_wait_hours(self) -> Optional[float]:
        if self.slip_assigned_time is None:
            return None
        return self.slip_assigned_time - self.arrival_time

    @property
    def turnaround_hours(self) -> Optional[float]:
        if self.departure_time is None:
            return None
        return self.departure_time - self.arrival_time

    @property
    def total_moves(self) -> int:
        return self.discharge_teu + self.load_teu


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class PortSim:
    """Holds all SimPy resources + shared mutable state for one run."""

    def __init__(self, params: PortParams):
        self.p = params
        self.env = simpy.Environment()
        self.rng = random.Random(params.random_seed)

        # 2 slips modeled as a pool of interchangeable berth tokens (0..n-1)
        self.berths = simpy.Store(self.env, capacity=params.n_slips)
        for i in range(params.n_slips):
            self.berths.put(i)

        # one dedicated crane pool per slip
        self.cranes = [
            simpy.Resource(self.env, capacity=params.cranes_per_slip)
            for _ in range(params.n_slips)
        ]

        # shared drayage truck pool feeding both slips
        self.trucks = simpy.Resource(self.env, capacity=params.n_trucks)

        # yard: import boxes waiting for pickup
        self.yard = simpy.Container(
            self.env, capacity=params.yard_capacity_teu, init=params.yard_capacity_teu * 0.2
        )
        # export pool: gated-in boxes waiting to be loaded
        self.export_pool = simpy.Container(
            self.env, capacity=params.export_pool_capacity_teu,
            init=params.export_pool_capacity_teu * 0.2,
        )

        # live state for the sampler / web viz (SimPy is single-threaded
        # cooperative, so plain dict/list mutation here is race-free)
        self.anchorage: list[int] = []
        self.ships_at_slip: dict[int, dict] = {}
        self.completed_ships: list[ShipRecord] = []
        self.moves_completed = 0
        self.samples: list[dict] = []
        self.crane_busy_hours = 0.0
        self.truck_busy_hours = 0.0

        self._next_ship_id = 0
        self._name_cycle = list(SHIP_NAMES)
        self.rng.shuffle(self._name_cycle)

    # -- ship generation -----------------------------------------------

    def _next_name(self) -> str:
        name = self._name_cycle[self._next_ship_id % len(self._name_cycle)]
        cycle_num = self._next_ship_id // len(self._name_cycle)
        return name if cycle_num == 0 else f"{name} ({cycle_num + 1})"

    def ship_generator(self):
        p = self.p
        while True:
            yield self.env.timeout(self.rng.expovariate(1.0 / p.ship_interarrival_mean_hours))
            ship = ShipRecord(
                id=self._next_ship_id,
                name=self._next_name(),
                discharge_teu=int(self.rng.triangular(
                    p.discharge_teu_min, p.discharge_teu_max, p.discharge_teu_mode)),
                load_teu=int(self.rng.triangular(
                    p.load_teu_min, p.load_teu_max, p.load_teu_mode)),
                arrival_time=self.env.now,
            )
            self._next_ship_id += 1
            self.env.process(self.ship_process(ship))

    # -- one ship's lifecycle -------------------------------------------

    def ship_process(self, ship: ShipRecord):
        self.anchorage.append(ship.id)

        slip_id = yield self.berths.get()
        self.anchorage.remove(ship.id)
        ship.slip_assigned_time = self.env.now
        ship.slip_id = slip_id

        moves = ["discharge"] * ship.discharge_teu + ["load"] * ship.load_teu
        state = {"i": 0}  # shared move cursor, safe: no yield between read+incr
        self.ships_at_slip[slip_id] = {
            "id": ship.id, "name": ship.name,
            "discharge": ship.discharge_teu, "load": ship.load_teu,
            "moves_done": 0, "moves_total": len(moves),
        }

        def crane_worker():
            crane_pool = self.cranes[slip_id]
            while state["i"] < len(moves):
                move = moves[state["i"]]
                state["i"] += 1
                yield from self._do_move(move, crane_pool)
                self.moves_completed += 1
                self.ships_at_slip[slip_id]["moves_done"] += 1

        workers = [self.env.process(crane_worker()) for _ in range(self.p.cranes_per_slip)]
        for w in workers:
            yield w

        ship.departure_time = self.env.now
        del self.ships_at_slip[slip_id]
        self.completed_ships.append(ship)
        yield self.berths.put(slip_id)

    def _do_move(self, move: str, crane_pool: simpy.Resource):
        """One container move. Gates on yard/export space, then needs a
        truck at the quay before the crane can lift, then the crane does
        its (short) lift, then the truck drives off to the yard/gate and
        back on its own -- that return trip must NOT block the crane from
        starting its next box, or the whole terminal's throughput would be
        capped at (number of cranes) / (truck round-trip time) instead of
        each resource pulling its own weight. That was bug #1 here: the
        first version awaited the truck's full round trip inside the same
        coroutine as the crane move, which serialized all 4 cranes onto
        the truck cycle time and collapsed effective throughput ~6x below
        what 12 trucks should support.
        """
        p = self.p
        # gate the move on yard capacity / export inventory first, so a
        # truck isn't held hostage waiting on yard/pool space
        if move == "discharge":
            yield self.yard.put(1)
        else:
            yield self.export_pool.get(1)

        truck_req = self.trucks.request()
        yield truck_req
        truck_start = self.env.now

        with crane_pool.request() as crane_req:
            yield crane_req
            crane_start = self.env.now
            yield self.env.timeout(self.rng.expovariate(1.0 / p.crane_move_minutes) / 60.0)
        self.crane_busy_hours += self.env.now - crane_start

        # crane is free again for the next box; the truck's return trip to
        # the yard/gate runs independently and doesn't hold up the crane
        self.env.process(self._truck_return(truck_req, truck_start))

    def _truck_return(self, truck_req, truck_start: float):
        p = self.p
        yield self.env.timeout(self.rng.expovariate(1.0 / p.truck_round_trip_minutes) / 60.0)
        self.truck_busy_hours += self.env.now - truck_start
        self.trucks.release(truck_req)

    # -- background yard processes ---------------------------------------

    def yard_pickup_process(self):
        p = self.p
        while True:
            yield self.env.timeout(self.rng.expovariate(1.0 / p.yard_pickup_mean_minutes) / 60.0)
            if self.yard.level >= 1:
                yield self.yard.get(1)

    def gate_in_process(self):
        p = self.p
        while True:
            yield self.env.timeout(self.rng.expovariate(1.0 / p.gate_in_mean_minutes) / 60.0)
            if self.export_pool.level + 1 <= self.export_pool.capacity:
                yield self.export_pool.put(1)

    # -- sampler for the web visualization -------------------------------

    def sampler(self):
        while True:
            self.samples.append({
                "t": round(self.env.now, 3),
                "anchorage": list(self.anchorage),
                "slip_ships": [self.ships_at_slip.get(i) for i in range(self.p.n_slips)],
                "crane_busy": [c.count for c in self.cranes],
                "crane_cap": [c.capacity for c in self.cranes],
                "truck_busy": self.trucks.count,
                "truck_queue": len(self.trucks.queue),
                "truck_cap": self.trucks.capacity,
                "yard_level": round(self.yard.level, 1),
                "yard_cap": self.yard.capacity,
                "export_level": round(self.export_pool.level, 1),
                "export_cap": self.export_pool.capacity,
                "ships_completed": len(self.completed_ships),
                "moves_completed": self.moves_completed,
            })
            yield self.env.timeout(self.p.sample_interval_hours)

    # -- run ---------------------------------------------------------------

    def run(self):
        self.env.process(self.ship_generator())
        self.env.process(self.yard_pickup_process())
        self.env.process(self.gate_in_process())
        self.env.process(self.sampler())
        self.env.run(until=self.p.sim_days * 24.0)
        return self


def run_simulation(params: PortParams) -> PortSim:
    return PortSim(params).run()


def summary_stats(sim: PortSim) -> dict:
    p = sim.p
    done = sim.completed_ships
    turnarounds = [s.turnaround_hours for s in done if s.turnaround_hours is not None]
    waits = [s.anchorage_wait_hours for s in done if s.anchorage_wait_hours is not None]

    total_crane_capacity_hours = p.n_slips * p.cranes_per_slip * p.sim_days * 24.0
    total_truck_capacity_hours = p.n_trucks * p.sim_days * 24.0

    return {
        "ships_completed": len(done),
        "ships_still_in_system": sim._next_ship_id - len(done),
        "avg_turnaround_hours": _mean(turnarounds),
        "p90_turnaround_hours": _percentile(turnarounds, 0.90),
        "avg_anchorage_wait_hours": _mean(waits),
        "pct_ships_that_waited": 100.0 * sum(1 for w in waits if w > 0.01) / len(waits) if waits else 0.0,
        "total_moves_completed": sim.moves_completed,
        "moves_per_day": sim.moves_completed / p.sim_days,
        "crane_utilization_pct": 100.0 * sim.crane_busy_hours / total_crane_capacity_hours,
        "truck_utilization_pct": 100.0 * sim.truck_busy_hours / total_truck_capacity_hours,
        "final_yard_level_pct": 100.0 * sim.yard.level / sim.yard.capacity,
        "final_export_pool_pct": 100.0 * sim.export_pool.level / sim.export_pool.capacity,
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(q * len(s)))
    return s[idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Two-slip Panamax container terminal DES")
    ap.add_argument("--days", type=float, default=21.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trucks", type=int, default=12)
    ap.add_argument("--interarrival", type=float, default=26.0, help="mean ship interarrival, hours")
    args = ap.parse_args()

    params = PortParams(
        sim_days=args.days, random_seed=args.seed, n_trucks=args.trucks,
        ship_interarrival_mean_hours=args.interarrival,
    )
    sim = run_simulation(params)
    stats = summary_stats(sim)

    print(f"\n== Container port DES: {params.sim_days:.0f} simulated days ==")
    print(f"  slips={params.n_slips}  cranes/slip={params.cranes_per_slip}  trucks={params.n_trucks}")
    print(f"  ship interarrival mean = {params.ship_interarrival_mean_hours:.1f}h\n")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:32s} {v:10.2f}")
        else:
            print(f"  {k:32s} {v:10d}")


if __name__ == "__main__":
    main()
