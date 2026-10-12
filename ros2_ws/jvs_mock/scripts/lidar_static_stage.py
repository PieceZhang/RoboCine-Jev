#!/usr/bin/env python3
"""Author a temporary USD override for static optical coverage tests.

Visual geometry and materials stay referenced from the supplied scene. Physics
APIs are removed before Kit loads it so no environment convex cooking starts.
No referenced source layer is modified.
"""
import argparse
from pathlib import Path
from pxr import Sdf, Usd, UsdGeom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.stage).resolve()
    output = Path(args.output).resolve()
    if output.exists() or output == source:
        raise ValueError('output must be a new temporary layer')
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = [str(source)]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(stage))
    count = 0
    for prim in stage.Traverse():
        # Raw metadata also covers vendor schemas not registered by USD alone.
        schemas = prim.GetMetadata('apiSchemas')
        if schemas is None:
            continue
        names = list(schemas.GetAppliedItems())
        kept = [n for n in names if not n.lower().startswith(('physics', 'physx'))]
        if names != kept:
            prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(kept))
            count += 1
    stage.GetRootLayer().Save()
    print(f'Static optical layer {output}: removed physics schemas on {count} prims')


if __name__ == '__main__':
    main()
