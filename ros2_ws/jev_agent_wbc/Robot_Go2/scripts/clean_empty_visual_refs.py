"""Remove only proven dangling visual references on the two visual-less heads."""
from pathlib import Path
from pxr import Sdf, Usd


def clean_empty_head_visual_references(base_layer_path):
    layer = Sdf.Layer.FindOrOpen(str(base_layer_path))
    root_asset = Path(base_layer_path).parent.parent / "go2_d1.usd"
    stage = Usd.Stage.Open(str(root_asset.resolve()))
    removed = []
    for head in ["Head_upper", "Head_lower"]:
        path = f"/go2_d1/{head}/visuals"
        spec = layer.GetPrimAtPath(path)
        if spec is None:
            continue
        for reference in list(spec.referenceList.GetAddedOrExplicitItems()):
            if not reference.primPath:
                continue
            if reference.assetPath:
                target_path = (Path(layer.realPath).parent / reference.assetPath).resolve()
                target = Sdf.Layer.FindOrOpen(str(target_path))
                if target is None or target.GetPrimAtPath(reference.primPath) is not None:
                    continue
            else:
                # The importer authors internal references for these visual-less
                # URDF links. Prove the exact target is absent from the full stage.
                if str(reference.primPath) != f"/visuals/{head}" or stage.GetPrimAtPath(reference.primPath):
                    continue
                target_path = root_asset
            spec.referenceList.Remove(reference)
            removed.append({"source_layer": str(base_layer_path), "prim": path,
                            "target_layer": str(target_path), "missing_prim": str(reference.primPath)})
    if removed:
        layer.Save()
    return removed
