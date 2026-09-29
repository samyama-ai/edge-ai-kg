"""`EA17`'s ground truth, computed in Python from the `Fleet` (#35).

Split out of `tests/test_blast_radius.py`, which had reached 571 lines -- past
the 500 the review harness reads, so it was skipped unread, and a skipped file
is an unreviewed one.

This half is the seam: it computes what the answer *should* be from the
generator's own edges, and touches no engine, no fixture and no pytest. The
tests that compare a query against it stay next door. `test_blast_radius.py`
imports `Truth` from here for its own tests; no sibling module imports either
name, so nothing else changed. (An earlier draft re-exported both, and the
comment saying so outlived the re-export by one commit -- which is the kind
of sentence this split exists to make findable.)

Deliberately not named `test_*`: pytest collects by that prefix, and a module
of pure computation with no assertions has nothing to collect.
"""
from __future__ import annotations

import collections

# Edges EA17 walks, and the direction it walks them in. `VARIANT_OF` and
# `OF_VARIANT` point *up* (variant -> model, deployment -> variant), so
# downstream traversal follows them backwards -- getting this wrong is the
# easiest way to write a ground truth that agrees with a broken query.
DOWNSTREAM = {"FEEDS": "forward", "NEXT_STAGE": "forward", "PRECEDES": "forward",
              "VARIANT_OF": "backward", "OF_VARIANT": "backward",
              "ON_BOARD": "forward"}


class Truth:
    """The ground truth for one fleet, computed in Python, built once.

    A class rather than free functions that stamp `fleet._downstream_adjacency`
    and `fleet._reachable_by_sensor` onto the caller's object. Three things
    needed caching -- the adjacency map, the label index, and each sensor's
    reachable set -- and the private-attribute form cached two of them, missed
    the third, and mutated a `Fleet` the test did not own. Here the lifetime is
    the object's: build one per fleet, throw it away with the fleet.

    The costs are real, not hypothetical. `exclusive` walks every *other*
    sensor, so 14 sensors meant 196 passes over 76,303 edges at `--scale 1.0`
    without the per-sensor cache, and `labels` was rebuilt over all 25,150 rows
    once per sensor inside that loop.
    """

    def __init__(self, fleet):
        self.fleet = fleet
        self.adjacency = collections.defaultdict(list)
        for _sl, src, rel, _tl, tgt, _p in fleet.edges:
            direction = DOWNSTREAM.get(rel)
            if direction == "forward":
                self.adjacency[src].append(tgt)
            elif direction == "backward":
                self.adjacency[tgt].append(src)
        self.labels = {row["id"]: label
                       for label, rows in fleet.nodes.items() for row in rows}
        self.requires = collections.defaultdict(set)
        for _sl, src, rel, _tl, tgt, _p in fleet.edges:
            if rel == "REQUIRES_SENSOR":
                self.requires[src].add(tgt)
        # The one-hop invariant `by_kind` reads below, checked where the data
        # is rather than assumed where it is used: every `REQUIRES_SENSOR`
        # endpoint is a `Sensor`, so a task that requires this sensor is
        # adjacent to it and its blast-radius depth is 1. A schema change that
        # routed this through a stage would make that depth wrong -- and wrong
        # in the same direction as the query, which is the pair of errors that
        # cancel and leave a green test.
        # Both ends. Checking only the target left the other half of "one hop
        # from a ClinicalTask to a Sensor" unexamined: an edge from something
        # else *to* a Sensor would satisfy the target check and still mean the
        # depth of 1 was measuring a different path.
        off_sensor = {tgt for sensors in self.requires.values() for tgt in sensors
                      if self.labels.get(tgt) != "Sensor"}
        off_task = {src for src in self.requires
                    if self.labels.get(src) != "ClinicalTask"}
        assert not off_sensor, (
            f"REQUIRES_SENSOR points at {sorted(off_sensor)[:3]}, which is not "
            f"a Sensor, so ClinicalTask is no longer one hop from the sensor "
            f"and `by_kind`'s depth of 1 is wrong. Derive it instead.")
        assert not off_task, (
            f"REQUIRES_SENSOR starts at {sorted(off_task)[:3]}, which is not a "
            f"ClinicalTask. `by_kind` counts these as tasks at depth 1, so the "
            f"kind is as load-bearing as the hop count.")
        self.modality = {row["id"]: row.get("modality")
                         for row in fleet.nodes.get("Sensor", ())}
        self.reachable = {row["id"]: self._reachable_from(row["id"])
                          for row in fleet.nodes["Sensor"]}

    def distances(self, sensor_id: str) -> dict[str, int]:
        """Shortest hop count from a sensor to everything downstream, in Python."""
        distance = {sensor_id: 0}
        queue = collections.deque([sensor_id])
        while queue:
            node = queue.popleft()
            for neighbour in self.adjacency.get(node, ()):
                if neighbour not in distance:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)
        return distance

    def _reachable_from(self, sensor_id: str) -> set:
        return {node for node in self.distances(sensor_id) if node != sensor_id}

    def exclusive(self, sensor_id: str) -> set:
        """Everything reachable from this sensor and from **no other** sensor.

        A set difference, not a second BFS with the answer baked in: it walks
        the other sensors independently and subtracts. That is what makes it a
        check on `only_via_me` rather than a restatement of it.

        ClinicalTask is handled by its own rule -- a task stops only if this
        sensor is the *only* sensor it requires -- because it is a consumer of
        the sensor rather than something downstream of it.
        """
        others = set()
        for other_id, reached in self.reachable.items():
            if other_id != sensor_id:
                others |= reached
        exclusive = self.reachable[sensor_id] - others
        exclusive |= {task for task in self.requires
                      if self.task_stops_without(task, sensor_id)}
        return exclusive

    def task_stops_without(self, task: str, sensor_id: str) -> bool:
        """Whether losing this sensor leaves the task short of a modality.

        Not "is this the task's only sensor". `etl/generate.py` links a task
        to every sensor whose modality it requires, so other sensors are
        alternatives only when they supply the same modality: a task needing
        ECG and PPG, with two ECG sensors and one PPG, survives losing an ECG
        sensor and stops on losing the PPG one.

        **This half restates the query's rule rather than checking it
        independently.** `EA17`'s ClinicalTask leg matches
        `o.modality = s.modality` and so does this; there is no second way to
        say "the task loses a modality". What it proves is that the engine
        agrees with Python at full cardinality -- a lost join or a mis-grouped
        aggregate still shows up. The *rule* is checked in
        `tests/test_blast_radius_semantics.py`, on a fixture where the answer
        is known by construction. The other four kinds are genuinely
        independent: `exclusive` walks every other sensor and subtracts.
        """
        sensors = self.requires[task]
        if sensor_id not in sensors:
            return False
        mine = self.modality.get(sensor_id)
        return not any(other != sensor_id and self.modality.get(other) == mine
                       for other in sensors)

    def by_kind(self, sensor_id: str) -> dict[str, tuple[int, int, int]]:
        """`{kind: (affected, only_via_me, nearest_depth)}`, EA17's shape."""
        distance = self.distances(sensor_id)
        exclusive = self.exclusive(sensor_id)
        per_kind: dict[str, list[int]] = collections.defaultdict(list)
        only: dict[str, int] = collections.defaultdict(int)
        for node_id, depth in distance.items():
            if node_id == sensor_id:
                continue
            kind = self.labels.get(node_id)
            if kind in ("SignalStage", "Model", "Deployment", "Board"):
                per_kind[kind].append(depth)
                if node_id in exclusive:
                    only[kind] += 1

        # ClinicalTask points *at* the sensor (`REQUIRES_SENSOR`), so it is not
        # reached by the downstream walk above -- it is a consumer that stops
        # for the same reason. Depth 1: it is adjacent to the sensor.
        tasks = {task for task, sensors in self.requires.items()
                 if sensor_id in sensors}
        if tasks:
            # Depth 1 is the schema's one hop, and the invariant that makes
            # it one is asserted in `__init__` rather than assumed here.
            per_kind["ClinicalTask"] = [1] * len(tasks)
            only["ClinicalTask"] = len(tasks & exclusive)

        return {kind: (len(depths), only[kind], min(depths))
                for kind, depths in per_kind.items()}
