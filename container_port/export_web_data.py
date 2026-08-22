"""
container_port/export_web_data.py

Runs the port DES and packs a sampled timeline into port_data.json, for
port_template.html (see build_port_page.py) to inline and animate.

    python export_web_data.py [--days 14] [--seed 42] [--trucks 12] ...
"""
from __future__ import annotations

import argparse
import json

from port_model import PortParams, run_simulation, summary_stats


def build_payload(params: PortParams) -> dict:
    sim = run_simulation(params)
    stats = summary_stats(sim)

    ships = []
    for s in sim.completed_ships:
        ships.append({
            "id": s.id, "name": s.name,
            "discharge": s.discharge_teu, "load": s.load_teu,
            "arrival": round(s.arrival_time, 2),
            "slip_assigned": round(s.slip_assigned_time, 2) if s.slip_assigned_time else None,
            "slip_id": s.slip_id,
            "departure": round(s.departure_time, 2) if s.departure_time else None,
            "turnaround": round(s.turnaround_hours, 2) if s.turnaround_hours else None,
            "wait": round(s.anchorage_wait_hours, 2) if s.anchorage_wait_hours else None,
        })
    ships.sort(key=lambda r: r["arrival"])

    return {
        "params": {
            "sim_days": params.sim_days,
            "n_slips": params.n_slips,
            "cranes_per_slip": params.cranes_per_slip,
            "n_trucks": params.n_trucks,
            "ship_interarrival_mean_hours": params.ship_interarrival_mean_hours,
            "crane_move_minutes": params.crane_move_minutes,
            "truck_round_trip_minutes": params.truck_round_trip_minutes,
            "yard_capacity_teu": params.yard_capacity_teu,
            "export_pool_capacity_teu": params.export_pool_capacity_teu,
            "sample_interval_hours": params.sample_interval_hours,
        },
        "stats": stats,
        "ships": ships,
        "samples": sim.samples,
    }


def main():
    ap = argparse.ArgumentParser(description="Export port DES data for the web viz")
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trucks", type=int, default=12)
    ap.add_argument("--interarrival", type=float, default=26.0)
    ap.add_argument("--out", type=str, default="port_data.json")
    args = ap.parse_args()

    params = PortParams(
        sim_days=args.days, random_seed=args.seed, n_trucks=args.trucks,
        ship_interarrival_mean_hours=args.interarrival,
    )
    payload = build_payload(params)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"wrote {args.out}: {len(payload['samples'])} samples, "
          f"{len(payload['ships'])} completed ships, "
          f"{len(json.dumps(payload)) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
