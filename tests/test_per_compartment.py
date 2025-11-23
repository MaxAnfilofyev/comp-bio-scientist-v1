import numpy as np

from ai_scientist.utils.per_compartment import (
    derive_per_compartment_from_arrays,
    derive_per_compartment_from_files,
)
from ai_scientist.tools.per_compartment_validator import validate_per_compartment_outputs


def test_derive_per_compartment_from_arrays_writes_outputs(tmp_path):
    failure = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    time = np.array([0.0, 1.0])
    nodes = ["a", "b"]

    res = derive_per_compartment_from_arrays(
        failure_matrix=failure,
        time_vector=time,
        nodes_order=nodes,
        output_dir=tmp_path,
        provenance="test",
    )

    assert res["status"] == "ok"
    assert (tmp_path / "per_compartment.npz").exists()
    validation = validate_per_compartment_outputs(tmp_path)
    assert validation.get("status") == "ok"


def test_derive_per_compartment_detects_mismatch(tmp_path):
    failure = np.zeros((2, 3))
    time = np.array([0.0])
    nodes = ["a", "b", "c"]

    res = derive_per_compartment_from_arrays(
        failure_matrix=failure,
        time_vector=time,
        nodes_order=nodes,
        output_dir=tmp_path,
    )
    assert res["status"] == "error"
    assert "mismatch" in res["reason"]


def test_derive_per_compartment_from_files(tmp_path):
    failure = np.array([[0.2, 0.8]], dtype=float)
    time = np.array([0.0])
    nodes = ["n0", "n1"]
    fm_path = tmp_path / "fm.npy"
    tv_path = tmp_path / "tv.npy"
    nodes_path = tmp_path / "nodes.txt"
    np.save(fm_path, failure)
    np.save(tv_path, time)
    nodes_path.write_text("\n".join(nodes))

    res = derive_per_compartment_from_files(
        failure_matrix_path=fm_path,
        time_vector_path=tv_path,
        nodes_order_path=nodes_path,
        output_dir=tmp_path,
        binary_threshold=0.5,
    )
    assert res["status"] == "ok"
    validation = validate_per_compartment_outputs(tmp_path)
    assert validation.get("status") == "ok"
