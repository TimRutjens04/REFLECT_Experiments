#!/usr/bin/env python3
"""
generate_sim_crops.py
---------------------
Generates AI2-THOR crops for full/empty state classification.

Objects : Bowl, Cup, Pot  (isFilledWithLiquid / canFillWithLiquid)
Scenes  : FloorPlan1–30
Labels  : 0 = empty, 1 = full
Target  : TARGET_PER_CLASS crops per label

Output  : data/sim_crops.pkl
  list of dicts: {image: PIL.Image, label: int, object_type: str, scene: str}

Crop extraction uses instance_detections2D bounding boxes,
matching the GroundingDINO output format used downstream.
"""

import os, sys, random, pickle, math
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

from PIL import Image
import numpy as np
import ai2thor.controller

FILLABLE_TYPES    = {"Bowl", "Cup", "Pot"}
TARGET_PER_CLASS  = 150
SCENES            = [f"FloorPlan{i}" for i in range(1, 31)]
MIN_BBOX_PX       = 30    # discard crops smaller than 30×30 px
MAX_POSITIONS     = 6     # interactable positions to try per object
OUTPUT_PATH       = DATA / "sim_crops.pkl"

# ── Helpers ────────────────────────────────────────────────────────────────
def crop_from_frame(frame_rgb, bbox):
    """
    frame_rgb : HxWx3 numpy array (RGB)
    bbox      : [x1, y1, x2, y2]
    Returns PIL.Image or None if bbox is too small.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if (x2 - x1) < MIN_BBOX_PX or (y2 - y1) < MIN_BBOX_PX:
        return None
    return Image.fromarray(frame_rgb).crop((x1, y1, x2, y2))


def rotation_to_face(agent_pos, obj_pos):
    """Y rotation (degrees) for agent at agent_pos to face obj_pos."""
    dx = obj_pos["x"] - agent_pos["x"]
    dz = obj_pos["z"] - agent_pos["z"]
    return math.degrees(math.atan2(dx, dz)) % 360


def set_fill_state(c, obj_id, fill: bool):
    """Fill or empty an object. Returns True if action succeeded."""
    action = "FillObjectWithLiquid" if fill else "EmptyLiquidFromObject"
    kwargs = {"objectId": obj_id}
    if fill:
        kwargs["fillLiquid"] = "water"
    event = c.step(action, **kwargs)
    return event.metadata["lastActionSuccess"]


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    samples = {0: [], 1: []}   # 0=empty, 1=full

    scenes = SCENES[:]
    random.shuffle(scenes)

    print(f"Target: {TARGET_PER_CLASS} crops per class  ({TARGET_PER_CLASS * 2} total)")
    print(f"Objects: {FILLABLE_TYPES}")
    print(f"Scenes: FloorPlan1–30 (shuffled)\n")

    c = ai2thor.controller.Controller(
        scene=scenes[0],
        width=400,
        height=400,
        renderInstanceSegmentation=True,
        fieldOfView=60,
    )

    try:
        for scene in scenes:
            n0, n1 = len(samples[0]), len(samples[1])
            if n0 >= TARGET_PER_CLASS and n1 >= TARGET_PER_CLASS:
                break

            print(f"{scene:>12}  empty={n0:>3}  full={n1:>3}")
            c.reset(scene)
            event = c.step("Pass")

            # Find fillable objects in this scene
            fillable = [
                obj for obj in event.metadata["objects"]
                if obj["objectType"] in FILLABLE_TYPES
                and obj.get("canFillWithLiquid", False)
            ]
            if not fillable:
                continue

            # All reachable positions in this scene (computed once)
            reach_event = c.step("GetReachablePositions")
            all_positions = reach_event.metadata.get("actionReturn") or []
            if not all_positions:
                continue

            for obj in fillable:
                obj_id   = obj["objectId"]
                obj_type = obj["objectType"]
                obj_pos  = obj["position"]

                # Filter to positions within 2 m of the object
                nearby = [
                    p for p in all_positions
                    if math.sqrt((p["x"] - obj_pos["x"])**2 + (p["z"] - obj_pos["z"])**2) < 2.0
                ]
                if not nearby:
                    continue

                # Try sampled positions; keep those where object is visible
                candidates = random.sample(nearby, min(MAX_POSITIONS * 3, len(nearby)))
                good = []
                for pos in candidates:
                    rot = rotation_to_face(pos, obj_pos)
                    c.step("TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"],
                           rotation=rot, horizon=20, standing=True)
                    ev = c.step("Pass")
                    if obj_id in ev.instance_detections2D:
                        good.append((pos, rot))
                    if len(good) >= MAX_POSITIONS:
                        break

                for pos, rot in good:
                    # Teleport to confirmed-visible position, tilt camera down
                    c.step("TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"],
                           rotation=rot, horizon=20, standing=True)

                    # Capture both states from this viewpoint
                    for label, fill in [(1, True), (0, False)]:
                        if len(samples[label]) >= TARGET_PER_CLASS:
                            continue

                        if not set_fill_state(c, obj_id, fill):
                            continue

                        event = c.step("Pass")
                        detections = event.instance_detections2D

                        if obj_id not in detections:
                            continue

                        crop = crop_from_frame(event.frame, detections[obj_id])
                        if crop is None:
                            continue

                        samples[label].append({
                            "image":       crop,
                            "label":       label,
                            "object_type": obj_type,
                            "scene":       scene,
                        })

    finally:
        c.stop()

    all_samples = samples[0] + samples[1]
    random.shuffle(all_samples)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_samples, f)

    print(f"\nDone.")
    print(f"  Empty : {len(samples[0])}")
    print(f"  Full  : {len(samples[1])}")
    print(f"  Total : {len(all_samples)}")
    print(f"  Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
