"""The evaluated entry: an annotator-conditioned detector whose masks are redrawn by class-balanced
pixel consensus over four other networks.

A manifest (`weights/vote.json`) names the base, its voters and the shared refiner:

  {"refiner": "refiner.pt",
   "base": {"name": "acond", "file": "acond.pt", "film_ids": 37, "annotator": 28},
   "voters": [{"name": "allrec", "file": "allrec.pt"},
              {"name": "m2f", "file": "m2f.pt", "kind": "mask2former", "config": "mask2former_config.json"},
              ...]}

Every Mask R-CNN goes through the same path as `filseg.cascade`: translate TTA, mask NMS, ownership,
refiner, then score >= 0.80, refiner probability >= 0.40 and area >= 250 px. `filseg.vote.cb_vote`
then reshapes the base's objects; the vote adds and drops none. A Mask2Former voter reads its own
queries (`filseg.mask2former`). Fragments the vote leaves beside an object are dropped last. The annotator row of the base is
fixed in the manifest, so inference reads nothing but the H-alpha image.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


from . import mask2former
from .cascade import image_candidates, member_masks
from .infer import Config
from .model import load_detector, load_refiner
from .vote import cb_vote, drop_satellites


@dataclass
class Member:
    name: str
    file: str
    backbone: str = "resnet50"
    cascade: bool = False
    film_ids: int = 0
    annotator: int = -1
    voter_class: str | None = None
    kind: str = "maskrcnn"
    config: str | None = None


def read_manifest(path: str | Path) -> tuple[Member, list[Member], str]:
    """(base, voters, refiner file) from a vote manifest."""
    spec = json.loads(Path(path).read_text())

    def member(entry: dict) -> Member:
        entry = dict(entry)
        entry["voter_class"] = entry.pop("class", None)
        return Member(**entry)

    return member(spec["base"]), [member(v) for v in spec["voters"]], spec["refiner"]


def load_member(member: Member, weights: Path, device):
    if member.kind == "mask2former":
        return mask2former.load_mask2former(weights / member.file, weights / member.config, device)
    return load_detector(
        weights / member.file,
        device,
        backbone=member.backbone,
        cascade=member.cascade,
        film_ids=member.film_ids,
        annotator=member.annotator,
    )


def vote_image(path: Path, base, voters, refiner, cfg: Config, device) -> list[dict]:
    """The final disjoint masks of one image. `base` and `voters` are (Member, network) pairs."""

    def masks(member: Member, network) -> list[dict]:
        if member.kind == "mask2former":
            return mask2former.member_masks(mask2former.query_masks(network, path, device))
        entry = image_candidates(path, network, refiner, cfg, device)
        return member_masks(entry, cfg)

    classes = {m.name: m.voter_class for m, _ in voters if m.voter_class}
    votes = {m.name: masks(m, net) for m, net in voters}
    return drop_satellites(cb_vote(masks(*base), votes, classes))


def vote_all(paths, manifest, weights, cfg: Config, device, log_every: int = 25):
    """Yield (stem, masks, seconds) per image, with every network loaded once."""
    weights = Path(weights)
    base_spec, voter_specs, refiner_file = read_manifest(manifest)
    refiner = load_refiner(weights / refiner_file, device)
    base = (base_spec, load_member(base_spec, weights, device))
    voters = [(m, load_member(m, weights, device)) for m in voter_specs]
    for i, path in enumerate(paths, 1):
        started = time.perf_counter()
        masks = vote_image(path, base, voters, refiner, cfg, device)
        seconds = time.perf_counter() - started
        if i % log_every == 0:
            print(f"[{i}/{len(paths)}] {path.stem}: {len(masks)} masks, {seconds:.2f} s", flush=True)
        yield path.stem, masks, seconds


def counts_of(masks: list[dict]) -> list[str]:
    return [m["counts"].decode("ascii") if isinstance(m["counts"], bytes) else m["counts"] for m in masks]


__all__ = ["Member", "counts_of", "read_manifest", "vote_all", "vote_image"]
