#!/usr/bin/env python3
"""
generate_compositionality_scenes.py
------------------------------------
Generates AI2-THOR full-frame scenes where both a Bowl and a Cup are
visible simultaneously, across all four fill-state combinations.

State combinations (bowl_state, cup_state):
  full  / full   →  same_state = True
  empty / empty  →  same_state = True
  full  / empty  →  same_state = False
  empty / full   →  same_state = False

Output: data/compositionality_scenes.pkl
  list of dicts:
    image        PIL.Image  — full 400×400 frame
    bowl_state   str        — 'full' or 'empty'
    cup_state    str        — 'full' or 'empty'
    same_state   bool
    scene        str
    bowl_bbox    list       — [x1, y1, x2, y2]
    cup_bbox     list       — [x1, y1, x2, y2]
"""

import os, sys, random, pickle, math
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

from PIL import Image
import ai2thor.controller

TARGET_PER_COMBO  = 50    # images per (bowl_state, cup_state) combination
MAX_DUAL          = 2     # max viewpoints per object pair — keeps dataset diverse across scenes
MIDPOINT_RADIUS   = 4.0   # pre-filter: only try positions within this distance of pair midpoint
MIN_BBOX_PX       = 40    # both objects must be at least this large in frame
SCENES            = [f"FloorPlan{i}" for i in range(1, 31)]
CUP_TYPES         = {"Cup", "Mug"}  # Mug is fillable and common in FloorPlan1-30
OUTPUT_PATH       = DATA / "compositionality_scenes.pkl"

COMBOS = [
    ("full",  "full"),
    ("empty", "empty"),
    ("full",  "empty"),
    ("empty", "full"),
]


# ── Helpers ─────────────────────────────────────────────────────────────────
def rotation_to_face(agent_pos, target_pos):
    dx = target_pos["x"] - agent_pos["x"]
    dz = target_pos["z"] - agent_pos["z"]
    return math.degrees(math.atan2(dx, dz)) % 360


def bbox_ok(bbox):
    return (bbox[2] - bbox[0]) >= MIN_BBOX_PX and (bbox[3] - bbox[1]) >= MIN_BBOX_PX


def set_fill(c, obj_id, fill: bool):
    obj = next((o for o in c.last_event.metadata["objects"] if o["objectId"] == obj_id), None)
    if obj and obj.get("isFilledWithLiquid") == fill:
        return True  # already in target state — skip action, avoid false failure
    action = "FillObjectWithLiquid" if fill else "EmptyLiquidFromObject"
    kwargs = {"objectId": obj_id}
    if fill:
        kwargs["fillLiquid"] = "water"
    return c.step(action, **kwargs).metadata["lastActionSuccess"]


def find_dual_positions(c, all_pos, bowl_id, bowl_pos, cup_id, cup_pos):
    """Return up to MAX_DUAL positions from which both objects are co-visible."""
    mid = {
        "x": (bowl_pos["x"] + cup_pos["x"]) / 2,
        "z": (bowl_pos["z"] + cup_pos["z"]) / 2,
    }
    candidates = [
        p for p in all_pos
        if math.sqrt((p["x"] - mid["x"])**2 + (p["z"] - mid["z"])**2) < MIDPOINT_RADIUS
    ]
    random.shuffle(candidates)

    good = []
    for pos in candidates:
        rot = rotation_to_face(pos, mid)
        c.step("TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"],
               rotation=rot, horizon=20, standing=True)
        dets = c.last_event.instance_detections2D
        if bowl_id in dets and cup_id in dets:
            if bbox_ok(dets[bowl_id]) and bbox_ok(dets[cup_id]):
                good.append((pos, rot))
        if len(good) >= MAX_DUAL:
            break
    return good


def teleport_close(c, all_pos, target_pos):
    """Teleport agent to the reachable position closest to target_pos, facing it."""
    closest = min(all_pos, key=lambda p:
        (p["x"] - target_pos["x"])**2 + (p["z"] - target_pos["z"])**2)
    rot = rotation_to_face(closest, target_pos)
    c.step("TeleportFull", x=closest["x"], y=closest["y"], z=closest["z"],
           rotation=rot, horizon=30, standing=True)


def set_fill_states(c, all_pos, bowl_id, bowl_pos, cup_id, cup_pos,
                    bowl_fill, cup_fill):
    """
    Set fill states while agent is close to each object.
    Returns False if either action fails.
    """
    teleport_close(c, all_pos, bowl_pos)
    if not set_fill(c, bowl_id, bowl_fill):
        return False
    teleport_close(c, all_pos, cup_pos)
    if not set_fill(c, cup_id, cup_fill):
        return False
    return True


def capture_frame(c, pos, rot, bowl_id, cup_id):
    """Teleport to capture position and return frame dict or None."""
    c.step("TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"],
           rotation=rot, horizon=20, standing=True)
    c.step("Pass")
    dets = c.last_event.instance_detections2D
    if bowl_id not in dets or cup_id not in dets:
        return None
    b_bbox  = dets[bowl_id]
    cu_bbox = dets[cup_id]
    if not (bbox_ok(b_bbox) and bbox_ok(cu_bbox)):
        return None
    img = Image.fromarray(c.last_event.frame)
    return {
        "bowl_bbox": list(b_bbox),
        "cup_bbox":  list(cu_bbox),
        "image":     img,
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    # samples[(bowl_state, cup_state)] = [dict, ...]
    samples = {combo: [] for combo in COMBOS}

    def done():
        return all(len(samples[k]) >= TARGET_PER_COMBO for k in COMBOS)

    def progress():
        return "  ".join(
            f"({b}/{c}):{len(samples[(b,c)])}" for b, c in COMBOS
        )

    scenes = SCENES[:]
    random.shuffle(scenes)

    c = ai2thor.controller.Controller(
        scene=scenes[0], width=400, height=400,
        renderInstanceSegmentation=True, fieldOfView=60,
    )

    try:
        for scene in scenes:
            if done():
                break

            print(f"\n{scene}  {progress()}")
            c.reset(scene)
            c.step("Pass")

            reach   = c.step("GetReachablePositions")
            all_pos = reach.metadata.get("actionReturn") or []
            if not all_pos:
                continue

            objs = c.last_event.metadata["objects"]  # read after GetReachablePositions

            bowls = [o for o in objs if o["objectType"] == "Bowl"
                     and o.get("canFillWithLiquid", False)
                     and o.get("pickupable", False)
                     and not o.get("isPickedUp", False)]
            cups  = [o for o in objs if o["objectType"] in CUP_TYPES
                     and o.get("canFillWithLiquid", False)
                     and o.get("pickupable", False)
                     and not o.get("isPickedUp", False)]
            print(f"  bowls={len(bowls)}  cups/mugs={len(cups)}  positions={len(all_pos)}")

            for bowl in bowls:
                for cup in cups:
                    if done():
                        break

                    dual_pos = find_dual_positions(
                        c, all_pos,
                        bowl["objectId"], bowl["position"],
                        cup["objectId"],  cup["position"],
                    )
                    if not dual_pos:
                        continue

                    for bowl_state, cup_state in COMBOS:
                        if len(samples[(bowl_state, cup_state)]) >= TARGET_PER_COMBO:
                            continue
                        # Set fill state once while agent is near each object,
                        # then capture from all dual positions for this combo.
                        ok = set_fill_states(
                            c, all_pos,
                            bowl["objectId"], bowl["position"],
                            cup["objectId"],  cup["position"],
                            bowl_state == "full", cup_state == "full",
                        )
                        if not ok:
                            continue
                        for pos, rot in dual_pos:
                            if len(samples[(bowl_state, cup_state)]) >= TARGET_PER_COMBO:
                                break
                            result = capture_frame(c, pos, rot,
                                                   bowl["objectId"], cup["objectId"])
                            if result is None:
                                continue
                            result.update({
                                "bowl_state": bowl_state,
                                "cup_state":  cup_state,
                                "same_state": bowl_state == cup_state,
                                "scene":      scene,
                            })
                            samples[(bowl_state, cup_state)].append(result)

    finally:
        c.stop()

    all_samples = []
    for combo in COMBOS:
        all_samples.extend(samples[combo])
    random.shuffle(all_samples)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_samples, f)

    print(f"\n{'='*50}")
    for b, cu in COMBOS:
        n = len(samples[(b, cu)])
        print(f"  bowl={b:<6}  cup={cu:<6}  n={n:>3}")
    print(f"  TOTAL  {len(all_samples)}")
    print(f"  Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
