"""Visual domain randomization of the arms' appearance: random flat colours and random image textures.

Every link visual matched by ``patterns`` gets its own OmniPBR material. Each :meth:`ArmVisualRandomizer.randomize`
call then resamples, per link, either a random colour or a random image from ``texture_dir`` (plus a random tint,
roughness and metallic). The STL meshes have no UVs, so textures are projected in object space (OmniPBR
``project_uvw``) and stay glued to the link as it moves. :meth:`restore` puts the PART_COLORS look back.

This is the standalone-scene version of Isaac Lab's ``mdp.randomize_visual_color`` /
``mdp.randomize_visual_texture_material`` event terms, which only work inside a ManagerBasedEnv.

Import this only after the Isaac Sim app is running (``AppLauncher``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import isaaclab.sim as sim_utils

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class VisualDRCfg:
    patterns: tuple[str, ...] = ("/World/Arm.*/lbr_link_.*", "/World/Arm.*/gripper_.*")
    """Prim-path regexes of the links to randomize (their ``/visuals`` children get the material)."""
    texture_dir: Path | None = None
    """Folder of images (searched recursively) to use as textures; None = colours only."""
    texture_prob: float = 0.5
    """Chance that a link gets an image texture instead of a flat colour (ignored without textures)."""
    per_link: bool = True
    """True: every link samples independently. False: all links of one arm share one sample."""
    color_range: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    """Uniform (low, high) linear RGB for flat colours."""
    tint_range: tuple[float, float] = (0.6, 1.0)
    """Per-channel multiplier applied on top of a texture, so the same image comes out in different hues."""
    texture_scale_range: tuple[float, float] = (2.0, 10.0)
    """Texture repeats per metre of link (larger = finer pattern)."""
    roughness_range: tuple[float, float] = (0.2, 0.9)
    metallic_range: tuple[float, float] = (0.0, 0.4)
    seed: int | None = None
    _textures: list[str] = field(default_factory=list, init=False, repr=False)


class ArmVisualRandomizer:
    def __init__(self, cfg: VisualDRCfg = VisualDRCfg()):
        from pxr import UsdShade

        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.textures = []
        if cfg.texture_dir is not None:
            self.textures = sorted(str(p) for p in Path(cfg.texture_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
            if not self.textures:
                raise FileNotFoundError(f"no images ({', '.join(IMAGE_EXTS)}) under {cfg.texture_dir}")

        stage = sim_utils.get_current_stage()
        visuals = sorted({p for pat in cfg.patterns for p in sim_utils.find_matching_prim_paths(pat + "/visuals")})
        if not visuals:
            raise RuntimeError(f"no link visuals match {cfg.patterns}; build the scene first")
        self.visuals = visuals
        self.original = {}  # visuals path -> material it had (e.g. from apply_part_colors), for restore()
        self.shaders = []
        for k, path in enumerate(visuals):
            prim = stage.GetPrimAtPath(path)
            prim.SetInstanceable(False)  # materials can't be bound to instanced prims
            self.original[path] = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath()
            mat_path = f"/World/Looks/visual_dr_{k}"
            mdl = sim_utils.MdlFileCfg(mdl_path="OmniPBR.mdl", project_uvw=True)
            mdl.func(mat_path, mdl)
            self.shaders.append(UsdShade.Shader(stage.GetPrimAtPath(f"{mat_path}/Shader")))
            self._set(self.shaders[-1], "world_or_object", False)  # object space: the texture moves with the link
        self.materials = [s.GetPrim().GetParent().GetPath() for s in self.shaders]
        # which sample each link uses: its own, or one shared per arm (/World/Arm1, /World/Arm2)
        groups = visuals if cfg.per_link else ["/".join(p.split("/")[:3]) for p in visuals]
        keys = sorted(set(groups))
        self.group_of = [keys.index(g) for g in groups]
        self.n_groups = len(keys)

    @staticmethod
    def _set(shader, name: str, value):
        from pxr import Gf, Sdf

        types = {bool: Sdf.ValueTypeNames.Bool, float: Sdf.ValueTypeNames.Float}
        if isinstance(value, str):
            shader.CreateInput(name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(value))
        elif isinstance(value, tuple) and len(value) == 3:
            shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
        elif isinstance(value, tuple) and len(value) == 2:
            shader.CreateInput(name, Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*value))
        else:
            shader.CreateInput(name, types[type(value)]).Set(value)

    def _sample(self) -> dict:
        c, r = self.cfg, self.rng
        s = {
            "reflection_roughness_constant": float(r.uniform(*c.roughness_range)),
            "metallic_constant": float(r.uniform(*c.metallic_range)),
        }
        if self.textures and r.random() < c.texture_prob:
            scale = float(r.uniform(*c.texture_scale_range))
            s |= {
                "diffuse_texture": str(r.choice(self.textures)),
                "diffuse_tint": tuple(float(v) for v in r.uniform(*c.tint_range, size=3)),
                "texture_scale": (scale, scale),
                "texture_rotate": float(r.uniform(0.0, 360.0)),
                "texture_translate": tuple(float(v) for v in r.uniform(0.0, 1.0, size=2)),
            }
        else:
            s |= {
                "diffuse_texture": "",  # empty asset -> OmniPBR falls back to diffuse_color_constant
                "diffuse_color_constant": tuple(float(v) for v in r.uniform(*c.color_range)),
                "diffuse_tint": (1.0, 1.0, 1.0),
            }
        return s

    def randomize(self) -> list[dict]:
        """Resample and bind random appearances. Returns the per-link samples (for logging)."""
        samples = [self._sample() for _ in range(self.n_groups)]
        stage = sim_utils.get_current_stage()
        out = []
        for path, shader, mat, g in zip(self.visuals, self.shaders, self.materials, self.group_of):
            for name, value in samples[g].items():
                self._set(shader, name, value)
            sim_utils.bind_visual_material(path, str(mat), stage=stage)
            out.append({"prim": path, **samples[g]})
        return out

    def restore(self):
        """Rebind the materials the links had before randomization (the PART_COLORS look)."""
        for path, mat in self.original.items():
            if mat:
                sim_utils.bind_visual_material(path, str(mat))


def make_procedural_textures(out_dir: Path, n: int = 64, size: int = 512, seed: int = 0) -> Path:
    """Write `n` random textures (noise, stripes, checkers, blobs, gradients) to `out_dir`, for when no image
    dataset is at hand. A real image set (e.g. DTD, https://www.robots.ox.ac.uk/~vgg/data/dtd/) is more varied."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size

    def palette(k):
        return rng.uniform(0, 255, size=(k, 3)).astype(np.float32)

    def colorize(t, k=2):  # map a 0..1 field through a random k-colour ramp
        cols = palette(k)
        x = np.clip(t, 0, 1)[..., None] * (k - 1)
        i = np.clip(np.floor(x).astype(int), 0, k - 2)
        f = x - i
        return cols[i[..., 0]] * (1 - f) + cols[i[..., 0] + 1] * f

    for idx in range(n):
        kind = idx % 5
        if kind == 0:  # multi-octave value noise
            t = np.zeros((size, size), np.float32)
            for octave in range(5):
                res = 4 * 2**octave
                t += cv2.resize(rng.random((res, res)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC) / 2**octave
            img = colorize((t - t.min()) / (np.ptp(t) + 1e-6), k=rng.integers(2, 5))
        elif kind == 1:  # stripes at a random angle and frequency
            a, f = rng.uniform(0, np.pi), rng.uniform(4, 40)
            t = 0.5 + 0.5 * np.sin(2 * np.pi * f * (xx * np.cos(a) + yy * np.sin(a)))
            img = colorize(t if rng.random() < 0.5 else (t > 0.5).astype(np.float32))
        elif kind == 2:  # checkerboard
            f = rng.integers(2, 24)
            img = colorize(((np.floor(xx * f) + np.floor(yy * f)) % 2).astype(np.float32))
        elif kind == 3:  # random filled circles
            img = np.empty((size, size, 3), np.float32)
            img[:] = palette(1)
            for c in palette(int(rng.integers(10, 80))):
                cv2.circle(img, tuple(int(v) for v in rng.integers(0, size, 2)), int(rng.integers(5, size // 6)), c.tolist(), -1)
        else:  # smooth gradient
            a = rng.uniform(0, 2 * np.pi)
            t = xx * np.cos(a) + yy * np.sin(a)
            img = colorize((t - t.min()) / np.ptp(t), k=rng.integers(2, 4))
        cv2.imwrite(str(out_dir / f"tex_{idx:03d}.png"), np.clip(img, 0, 255).astype(np.uint8)[..., ::-1])
    return out_dir
