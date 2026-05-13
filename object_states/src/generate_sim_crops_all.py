#!/usr/bin/env python3
"""
generate_sim_crops_all.py
-------------------------
Generates AI2-THOR crops for 7 of the 8 planned state pairs.
Skipped: held/free (first-person framing — crop domain unrealistic for GroundingDINO output).

Implemented pairs
  0  full / empty      Bowl, Cup, Pot
  1  open / closed     Cabinet, Fridge, Microwave
  2  on / off          Faucet, StoveBurner, CoffeeMachine
  3  cooked / raw      Apple, Bread, Potato, Egg
  4  dirty / clean     Apple, Bread, Potato, Plate
  5  broken / intact   Bottle, Egg, Plate
  6  sliced / whole    Apple, Bread, Potato

Output: data/sim_crops_all.pkl
  list of dicts: {image, label, pair_idx, pair_name, object_type, scene}
"""

import os, sys, random, pickle, math
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

from PIL import Image
import ai2thor.controller

TARGET_PER_CLASS = 300      # per state (pos + neg) per pair → 160 per pair
MIN_BBOX_PX      = 25
MAX_NEARBY       = 20     # confirmed-visible positions to collect per object
NEARBY_RADIUS    = 3.0      # wider radius to find more viewpoints per object
SCENES           = [f"FloorPlan{i}" for i in range(1, 31)]
OUTPUT_PATH      = DATA / "sim_crops_all.pkl"

PAIR_OBJECTS = {
    "full_empty":    ["Bowl", "Cup", "Pot"],
    "open_closed":   ["Cabinet", "Fridge", "Microwave"],
    "on_off":        ["Faucet", "CoffeeMachine"],
    "cooked_raw":    ["Potato", "Bread", "Egg"],
    "dirty_clean":   ["Apple", "Bread", "Potato", "Plate"],
    "broken_intact": ["Bottle", "Egg", "Plate"],
    "sliced_whole":  ["Apple", "Bread", "Potato"],
}
PAIR_NAMES = list(PAIR_OBJECTS.keys())    # fixed order → pair_idx

# ── Helpers ────────────────────────────────────────────────────────────────
def crop_from_frame(frame_rgb, bbox):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if (x2 - x1) < MIN_BBOX_PX or (y2 - y1) < MIN_BBOX_PX:
        return None
    return Image.fromarray(frame_rgb).crop((x1, y1, x2, y2))

def rotation_to_face(agent_pos, obj_pos):
    dx = obj_pos["x"] - agent_pos["x"]
    dz = obj_pos["z"] - agent_pos["z"]
    return math.degrees(math.atan2(dx, dz)) % 360

def nearby_visible_positions(c, all_positions, obj_id, obj_pos, n=MAX_NEARBY, radius=NEARBY_RADIUS):
    """Return up to n positions within radius from which obj_id is visible."""
    close = [p for p in all_positions
             if math.sqrt((p["x"]-obj_pos["x"])**2 + (p["z"]-obj_pos["z"])**2) < radius]
    random.shuffle(close)
    good = []
    for pos in close:
        rot = rotation_to_face(pos, obj_pos)
        c.step("TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"],
               rotation=rot, horizon=20, standing=True)
        if obj_id in c.last_event.instance_detections2D:
            good.append((pos, rot))
        if len(good) >= n:
            break
    return good

def get_crop(c, obj_id, pos, rot):
    c.step("TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"],
           rotation=rot, horizon=20, standing=True)
    c.step("Pass")
    bbox = c.last_event.instance_detections2D.get(obj_id)
    if bbox is None:
        return None
    return crop_from_frame(c.last_event.frame, bbox)

def ok(event):
    return event.metadata["lastActionSuccess"]

# ── Per-pair collectors ─────────────────────────────────────────────────────
def collect_full_empty(c, obj, positions, all_pos, samples, scene):
    oid      = obj["objectId"]
    obj_type = obj["objectType"]
    if not obj.get("canFillWithLiquid", False):
        return
    good = nearby_visible_positions(c, all_pos, oid, obj["position"])
    for pos, rot in good:
        if len(samples[0]) < TARGET_PER_CLASS:
            ok(c.step("EmptyLiquidFromObject", objectId=oid))
            crop = get_crop(c, oid, pos, rot)
            if crop: samples[0].append((crop, obj_type, scene))
        if len(samples[1]) < TARGET_PER_CLASS:
            ok(c.step("FillObjectWithLiquid", objectId=oid, fillLiquid="water"))
            crop = get_crop(c, oid, pos, rot)
            if crop: samples[1].append((crop, obj_type, scene))

def collect_open_closed(c, obj, all_pos, samples, scene):
    oid      = obj["objectId"]
    obj_type = obj["objectType"]
    if not obj.get("openable", False):
        return
    good = nearby_visible_positions(c, all_pos, oid, obj["position"])
    for pos, rot in good:
        if len(samples[0]) < TARGET_PER_CLASS:
            ok(c.step("CloseObject", objectId=oid))
            crop = get_crop(c, oid, pos, rot)
            if crop: samples[0].append((crop, obj_type, scene))
        if len(samples[1]) < TARGET_PER_CLASS:
            ok(c.step("OpenObject", objectId=oid))
            crop = get_crop(c, oid, pos, rot)
            if crop: samples[1].append((crop, obj_type, scene))

def collect_on_off(c, obj, all_pos, samples, scene):
    oid      = obj["objectId"]
    obj_type = obj["objectType"]
    if not obj.get("toggleable", False):
        return
    good = nearby_visible_positions(c, all_pos, oid, obj["position"])
    for pos, rot in good:
        if len(samples[0]) < TARGET_PER_CLASS:
            ok(c.step("ToggleObjectOff", objectId=oid))
            crop = get_crop(c, oid, pos, rot)
            if crop: samples[0].append((crop, obj_type, scene))
        if len(samples[1]) < TARGET_PER_CLASS:
            ok(c.step("ToggleObjectOn", objectId=oid))
            crop = get_crop(c, oid, pos, rot)
            if crop: samples[1].append((crop, obj_type, scene))

def collect_cooked_raw(c, obj, all_pos, samples, scene):
    """Two-phase: collect crops from all positions first, then cook once.
    If cook succeeds: all pre-cook crops are raw, collect post-cook crops too.
    If 'already' error: all crops are cooked (pre-cooked object).
    """
    oid      = obj["objectId"]
    obj_type = obj["objectType"]
    good = nearby_visible_positions(c, all_pos, oid, obj["position"])
    if not good:
        return
    crops = []
    for pos, rot in good:
        crop = get_crop(c, oid, pos, rot)
        if crop:
            crops.append((crop, pos, rot))
    if not crops:
        return
    # Agent is at last position; try CookObject to determine initial state
    ev = c.step("CookObject", objectId=oid, forceAction=True)
    if ev.metadata["lastActionSuccess"]:
        # Object was raw → all collected crops are raw
        for crop, _, _ in crops:
            if len(samples[0]) < TARGET_PER_CLASS:
                samples[0].append((crop, obj_type, scene))
        # Collect cooked crops from same positions
        for _, pos, rot in crops:
            if len(samples[1]) < TARGET_PER_CLASS:
                crop_after = get_crop(c, oid, pos, rot)
                if crop_after: samples[1].append((crop_after, obj_type, scene))
    elif "already" in ev.metadata.get("errorMessage", "").lower():
        # Object was pre-cooked → all collected crops are cooked
        for crop, _, _ in crops:
            if len(samples[1]) < TARGET_PER_CLASS:
                samples[1].append((crop, obj_type, scene))

def collect_dirty_clean(c, obj, all_pos, samples, scene):
    oid      = obj["objectId"]
    obj_type = obj["objectType"]
    if not obj.get("dirtyable", False):
        return
    good = nearby_visible_positions(c, all_pos, oid, obj["position"])
    for pos, rot in good:
        if len(samples[0]) < TARGET_PER_CLASS:
            ok(c.step("CleanObject", objectId=oid))
            crop = get_crop(c, oid, pos, rot)
            if crop: samples[0].append((crop, obj_type, scene))
        if len(samples[1]) < TARGET_PER_CLASS:
            ok(c.step("DirtyObject", objectId=oid))
            crop = get_crop(c, oid, pos, rot)
            if crop: samples[1].append((crop, obj_type, scene))

def collect_broken_intact(c, scene, obj, all_pos, samples):
    """Broken is irreversible; collect intact crops from all positions first,
    then break once and collect broken crops from the same positions."""
    oid = obj["objectId"]
    obj_type = obj["objectType"]
    if not obj.get("breakable", False):
        return
    good = nearby_visible_positions(c, all_pos, oid, obj["position"])
    if not good:
        return
    # Collect intact crops from every position
    good_positions = []  # positions that actually yielded a crop
    for pos, rot in good:
        get_crop(c, oid, pos, rot)
        bbox = c.last_event.instance_detections2D.get(oid)
        if bbox is not None:
            crop = crop_from_frame(c.last_event.frame, bbox)
            if crop:
                if len(samples[0]) < TARGET_PER_CLASS:
                    samples[0].append((crop, obj_type, scene))
                good_positions.append((pos, rot))
    if not good_positions:
        return
    ev = c.step("BreakObject", objectId=oid, forceAction=True)
    if not ev.metadata["lastActionSuccess"]:
        return
    # After breaking, AI2-THOR may replace the object with shards that have
    # new objectIds. Find the broken version by scanning metadata.
    broken_id = oid
    for o in ev.metadata["objects"]:
        if o.get("isBroken") and o["objectType"] == obj_type:
            broken_id = o["objectId"]
            break
    # Broken pieces fall to the floor — sweep horizons to find them.
    # Also use a smaller bbox threshold since shards are smaller than whole objects.
    for pos, rot in good_positions:
        if len(samples[1]) >= TARGET_PER_CLASS:
            break
        for horizon in [20, 35, 50]:
            c.step("TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"],
                   rotation=rot, horizon=horizon, standing=True)
            c.step("Pass")
            dets = c.last_event.instance_detections2D
            bbox = dets.get(broken_id)
            if bbox is None:
                for det_id, det_bbox in dets.items():
                    if obj_type in det_id:
                        bbox = det_bbox
                        break
            if bbox is not None:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                    crop = Image.fromarray(c.last_event.frame).crop((x1, y1, x2, y2))
                    samples[1].append((crop, obj_type, scene))
                    break  # found at this horizon, move to next position

def collect_sliced_whole(c, scene, obj, all_pos, samples):
    """Slicing is irreversible: collect whole crops first, then slice once and collect sliced crops.
    Requires a Knife in the scene. Returns True if a slice was successfully performed."""
    oid      = obj["objectId"]
    obj_type = obj["objectType"]
    if not obj.get("sliceable", False):
        return False

    knife = next(
        (o for o in c.last_event.metadata["objects"]
         if o["objectType"] == "Knife" and not o.get("isPickedUp", False)),
        None
    )
    if knife is None:
        return False

    good = nearby_visible_positions(c, all_pos, oid, obj["position"])
    if not good:
        return False

    # Collect whole crops from all visible positions
    whole_crops = []
    for pos, rot in good:
        crop = get_crop(c, oid, pos, rot)
        if crop:
            whole_crops.append((crop, pos, rot))
            if len(samples[0]) < TARGET_PER_CLASS:
                samples[0].append((crop, obj_type, scene))

    if not whole_crops:
        return False

    # Pick up the knife from a nearby position
    knife_good = nearby_visible_positions(
        c, all_pos, knife["objectId"], knife["position"], n=5, radius=2.0)
    picked_up = False
    for kpos, krot in knife_good:
        c.step("TeleportFull", x=kpos["x"], y=kpos["y"], z=kpos["z"],
               rotation=krot, horizon=20, standing=True)
        if ok(c.step("PickupObject", objectId=knife["objectId"])):
            picked_up = True
            break

    if not picked_up:
        return False

    # Slice from a position near the object
    sliced = False
    for _, pos, rot in whole_crops[:5]:
        c.step("TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"],
               rotation=rot, horizon=20, standing=True)
        if ok(c.step("SliceObject", objectId=oid)):
            sliced = True
            break

    c.step("DropHandObject")

    if not sliced:
        return False

    # Collect sliced crops — pieces appear as e.g. "AppleSliced|..."
    sliced_type = obj_type + "Sliced"
    for _, pos, rot in whole_crops:
        if len(samples[1]) >= TARGET_PER_CLASS:
            break
        c.step("TeleportFull", x=pos["x"], y=pos["y"], z=pos["z"],
               rotation=rot, horizon=20, standing=True)
        c.step("Pass")
        dets = c.last_event.instance_detections2D
        for det_id, det_bbox in dets.items():
            if sliced_type in det_id or ("Slice" in det_id and obj_type in det_id):
                x1, y1, x2, y2 = int(det_bbox[0]), int(det_bbox[1]), int(det_bbox[2]), int(det_bbox[3])
                if (x2 - x1) >= MIN_BBOX_PX and (y2 - y1) >= MIN_BBOX_PX:
                    crop = Image.fromarray(c.last_event.frame).crop((x1, y1, x2, y2))
                    samples[1].append((crop, obj_type, scene))
                    break

    return True


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    all_samples = []   # flat list of dicts

    # samples[pair_name][label] = [(PIL.Image, obj_type, scene), ...]
    samples = {p: {0: [], 1: []} for p in PAIR_NAMES}

    scenes = SCENES[:]
    random.shuffle(scenes)

    c = ai2thor.controller.Controller(
        scene=scenes[0], width=400, height=400,
        renderInstanceSegmentation=True, fieldOfView=60,
    )

    def done():
        return all(
            len(samples[p][0]) >= TARGET_PER_CLASS and len(samples[p][1]) >= TARGET_PER_CLASS
            for p in PAIR_NAMES
        )

    def progress():
        return "  ".join(f"{p[:8]}:{len(samples[p][0])}/{len(samples[p][1])}" for p in PAIR_NAMES)

    try:
        for scene in scenes:
            if done():
                break

            print(f"\n{scene}  {progress()}")
            c.reset(scene)
            event = c.step("Pass")

            reach = c.step("GetReachablePositions")
            all_pos = reach.metadata.get("actionReturn") or []
            if not all_pos:
                continue

            objs_by_type = {}
            for o in event.metadata["objects"]:
                objs_by_type.setdefault(o["objectType"], []).append(o)

            # ── full / empty ──────────────────────────────────────────────
            if not (len(samples["full_empty"][0]) >= TARGET_PER_CLASS and
                    len(samples["full_empty"][1]) >= TARGET_PER_CLASS):
                for typ in PAIR_OBJECTS["full_empty"]:
                    for obj in objs_by_type.get(typ, []):
                        collect_full_empty(c, obj, None, all_pos, samples["full_empty"], scene)

            # ── open / closed ─────────────────────────────────────────────
            if not (len(samples["open_closed"][0]) >= TARGET_PER_CLASS and
                    len(samples["open_closed"][1]) >= TARGET_PER_CLASS):
                for typ in PAIR_OBJECTS["open_closed"]:
                    for obj in objs_by_type.get(typ, []):
                        collect_open_closed(c, obj, all_pos, samples["open_closed"], scene)

            # ── on / off ──────────────────────────────────────────────────
            if not (len(samples["on_off"][0]) >= TARGET_PER_CLASS and
                    len(samples["on_off"][1]) >= TARGET_PER_CLASS):
                for typ in PAIR_OBJECTS["on_off"]:
                    for obj in objs_by_type.get(typ, []):
                        collect_on_off(c, obj, all_pos, samples["on_off"], scene)

            # ── cooked / raw ──────────────────────────────────────────────
            if not (len(samples["cooked_raw"][0]) >= TARGET_PER_CLASS and
                    len(samples["cooked_raw"][1]) >= TARGET_PER_CLASS):
                for typ in PAIR_OBJECTS["cooked_raw"]:
                    for obj in objs_by_type.get(typ, []):
                        collect_cooked_raw(c, obj, all_pos, samples["cooked_raw"], scene)

            # ── dirty / clean ─────────────────────────────────────────────
            if not (len(samples["dirty_clean"][0]) >= TARGET_PER_CLASS and
                    len(samples["dirty_clean"][1]) >= TARGET_PER_CLASS):
                for typ in PAIR_OBJECTS["dirty_clean"]:
                    for obj in objs_by_type.get(typ, []):
                        collect_dirty_clean(c, obj, all_pos, samples["dirty_clean"], scene)

            # ── broken / intact ───────────────────────────────────────────
            # irreversible: collect one per object, then reset scene
            if not (len(samples["broken_intact"][0]) >= TARGET_PER_CLASS and
                    len(samples["broken_intact"][1]) >= TARGET_PER_CLASS):
                for typ in PAIR_OBJECTS["broken_intact"]:
                    for obj in objs_by_type.get(typ, []):
                        before_n = len(samples["broken_intact"][1])
                        collect_broken_intact(c, scene, obj, all_pos, samples["broken_intact"])
                        if len(samples["broken_intact"][1]) > before_n:
                            # reset scene after breaking so other objects stay intact
                            c.reset(scene)
                            c.step("Pass")
                            reach = c.step("GetReachablePositions")
                            all_pos = reach.metadata.get("actionReturn") or []

            # ── sliced / whole ────────────────────────────────────────────
            # irreversible: collect whole crops first, then slice and collect sliced crops
            if not (len(samples["sliced_whole"][0]) >= TARGET_PER_CLASS and
                    len(samples["sliced_whole"][1]) >= TARGET_PER_CLASS):
                for typ in PAIR_OBJECTS["sliced_whole"]:
                    for obj in objs_by_type.get(typ, []):
                        before_n = len(samples["sliced_whole"][1])
                        collect_sliced_whole(c, scene, obj, all_pos, samples["sliced_whole"])
                        if len(samples["sliced_whole"][1]) > before_n:
                            # reset scene after slicing so other objects stay whole
                            c.reset(scene)
                            c.step("Pass")
                            reach = c.step("GetReachablePositions")
                            all_pos = reach.metadata.get("actionReturn") or []

    finally:
        c.stop()

    # Flatten into list of dicts
    for pair_name in PAIR_NAMES:
        pair_idx = PAIR_NAMES.index(pair_name)
        for label in [0, 1]:
            for img, obj_type, scene in samples[pair_name][label]:
                all_samples.append({
                    "image":       img,
                    "label":       label,
                    "pair_idx":    pair_idx,
                    "pair_name":   pair_name,
                    "object_type": obj_type,
                    "scene":       scene,
                })
    random.shuffle(all_samples)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(all_samples, f)

    print(f"\n{'='*55}")
    for pair_name in PAIR_NAMES:
        n0, n1 = len(samples[pair_name][0]), len(samples[pair_name][1])
        print(f"  {pair_name:<16}  neg={n0:>3}  pos={n1:>3}  total={n0+n1:>3}")
    print(f"  {'TOTAL':<16}  {len(all_samples):>3} samples")
    print(f"  Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
