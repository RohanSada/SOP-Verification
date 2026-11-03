"""Write a procedure package's vocab.yaml from IMPACT label files (PSR procedure info + TAS-S mapping).

Usage: python scripts/build_vocab.py <impact annotations root> <out vocab.yaml>
"""

import sys

from sopverify.data.impact.adapters import read_mapping, read_psr_vocabulary
from sopverify.data.impact.discover import AnnotationIndex
from sopverify.utils import read_json, write_yaml


def main(root: str, out: str) -> None:
    idx = AnnotationIndex.build(root)
    vocab = read_psr_vocabulary(idx.singletons["psr_procedure_info"])
    comps = read_json(idx.singletons["psr_component_names"]) if "psr_component_names" in idx.singletons else []
    tas = read_mapping(idx.singletons["mapping_tas_s"])
    data = {
        "source": "IMPACT v1.1: PSR/labels/procedure_info_IMPACT.json, component_names.json, TAS/mapping_TAS-S.txt",
        "components": comps,
        "psr_events": [{"id": v["id"], "kind": v["kind"], "component": v["component"], "impact_id": v["impact_id"]}
                       for v in vocab],
        "tas_s_labels": [tas[i] for i in sorted(tas)],
    }
    write_yaml(out, data)
    print(f"wrote {out}: {len(vocab)} PSR events, {len(comps)} components, {len(tas)} TAS-S labels")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
