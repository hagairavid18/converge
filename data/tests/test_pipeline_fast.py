"""Fast (no ML, no network) tests for the metadata/splitting stage of the
pre-processing pipeline, plus an end-to-end integration check that only runs
when the raw SKEMPI CSV and structures are already present locally.
"""

from __future__ import annotations

import pytest

from data.affinity import compute_ddg_label, parse_affinity_value, parse_temperature_kelvin
from data.chain_roles import (
    antibody_side_is_protein_1,
    classify_heavy_light_by_sequence,
    is_antibody_protein_name,
    parse_pdb_field,
)
from data.homology_dedup import homology_sibling_sample_ids, normalize_protein_name
from data.mutation_tokens import parse_mutation_string, parse_mutation_tokens
from shared.constants import (
    SKEMPI_RAW_CSV,
    SKEMPI_RAW_STRUCTURES_DIR,
    ChainRole,
    LabelType,
    MutationRecord,
    PointMutation,
)


def test_parse_mutation_string_without_insertion_code():
    wt, chain, position, insertion_code, mutant = parse_mutation_string("TC121A")
    assert (wt, chain, position, insertion_code, mutant) == ("T", "C", 121, "", "A")


def test_parse_mutation_string_with_insertion_code():
    wt, chain, position, insertion_code, mutant = parse_mutation_string("DL27aN")
    assert (wt, chain, position, insertion_code, mutant) == ("D", "L", 27, "a", "N")


def test_parse_mutation_tokens_multi_mutant_aligned_with_locations():
    tokens = parse_mutation_tokens("TC121A,KC122A", "TC110A,KC111A", "SUR,RIM")
    assert [t.residue_position for t in tokens] == [121, 122]
    assert [t.aligned_interface_position for t in tokens] == [110, 111]
    assert [t.interface_region.value for t in tokens] == ["surface", "rim"]


def test_parse_pdb_field():
    spec = parse_pdb_field("1AHW_AB_C")
    assert spec.pdb_id == "1AHW"
    assert spec.protein_1_chains == "AB"
    assert spec.protein_2_chains == "C"


def test_is_antibody_protein_name_keyword_and_override():
    assert is_antibody_protein_name("HyHEL-63 fab")
    assert is_antibody_protein_name("Herceptin")
    assert not is_antibody_protein_name("Tissue factor")


def test_antibody_side_is_protein_1_uses_keyword_then_default():
    assert antibody_side_is_protein_1("Immunoglobulin fab 5G9", "Tissue factor")
    assert not antibody_side_is_protein_1("Subtype N9 neuraminidase", "Antibody NC41 scFv")


def test_classify_heavy_light_by_sequence_handles_full_fab_not_just_fv():
    heavy_seq = (
        "EIQLQQSGAELVRPGALVKLSCKASGFNIKDYYMHWVKQRPEQGLEWIGWIDPENGDTEYAPKFQGKATMTADTSSNTAYLQLSSLTSEDTAVYYCNAGHDYDRGRFPYWGQGTLVTVSAAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKI"
    )
    light_seq = (
        "DIKMTQSPSSMYASLGERVTITCKASQDIRKYLNWYQQKPGKAPKLLIYAASILESGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQANSFPYTFGGGTKLEIKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC"
    )
    roles = classify_heavy_light_by_sequence({"A": light_seq, "B": heavy_seq})
    assert roles["A"] == ChainRole.LIGHT
    assert roles["B"] == ChainRole.HEAVY


def test_parse_affinity_value_variants():
    assert parse_affinity_value("5.26E-11").value_molar == pytest.approx(5.26e-11)
    assert parse_affinity_value("n.b.").is_no_binding
    bound = parse_affinity_value(">2E-05")
    assert bound.bound_direction == ">"
    assert bound.value_molar == pytest.approx(2e-5)


def test_parse_temperature_kelvin_strips_assumed_suffix():
    assert parse_temperature_kelvin("298(assumed)") == pytest.approx(298.0)
    assert parse_temperature_kelvin("294") == pytest.approx(294.0)


def test_compute_ddg_label_bounded_destabilizing_when_mutant_binds_weaker():
    label = compute_ddg_label("1E-08", "3.4E-09", "298(assumed)")
    assert label.label_type == LabelType.BOUNDED
    assert label.ddg_kcal_mol > 0
    assert label.ddg_bin is not None


def test_compute_ddg_label_no_binding():
    label = compute_ddg_label("n.b.", "3.4E-09", "298")
    assert label.label_type == LabelType.NB
    assert label.ddg_kcal_mol is None


def test_compute_ddg_label_inequality_direction():
    label = compute_ddg_label(">2E-05", "3.4E-09", "298")
    assert label.label_type == LabelType.INEQ
    assert label.ineq_direction == ">"


def _make_point_mutation(position, aligned_position):
    return PointMutation(
        chain_id="A",
        chain_role=ChainRole.HEAVY,
        wt_residue="A",
        mutant_residue="G",
        residue_position=position,
        aligned_interface_position=aligned_position,
    )


def _make_record(sample_id, pdb_id, complex_name, *position_and_aligned_pairs):
    return MutationRecord(
        sample_id=sample_id,
        pdb_id=pdb_id,
        complex_name=complex_name,
        mutations=[_make_point_mutation(position, aligned) for position, aligned in position_and_aligned_pairs],
        label_type=LabelType.BOUNDED,
        ddg_kcal_mol=1.0,
        ddg_bin=5,
    )


def test_homology_sibling_sample_ids_flags_all_but_the_first_in_a_group():
    records = [
        _make_record("s1", "2NZ9", "AR2 mab :: BoNT/A1", (874, 861)),
        _make_record("s2", "2NYY", "AR2 mab :: BoNT/A1", (874, 861)),
        _make_record("s3", "2NZ9", "AR2 mab :: BoNT/A1", (875, 862)),
    ]
    siblings = homology_sibling_sample_ids(records)
    assert siblings == {"s2"}
    assert normalize_protein_name("  AR2 MAB  ") == "ar2 mab"


def test_homology_sibling_sample_ids_matches_whole_mutation_set_not_subset():
    records = [
        _make_record("s1", "2NZ9", "AR2 mab :: BoNT/A1", (874, 861)),
        _make_record("s2", "2NYY", "AR2 mab :: BoNT/A1", (874, 861), (875, 862)),
    ]
    assert homology_sibling_sample_ids(records) == set()


def test_discard_homology_siblings_drops_all_but_representative():
    from data.homology_dedup import discard_homology_siblings

    records = [
        _make_record("s1", "2NZ9", "AR2 mab :: BoNT/A1", (874, 861)),
        _make_record("s2", "2NYY", "AR2 mab :: BoNT/A1", (874, 861)),
        _make_record("s3", "2NZ9", "AR2 mab :: BoNT/A1", (875, 862)),
    ]
    kept = discard_homology_siblings(records)

    assert {r.sample_id for r in kept} == {"s1", "s3"}


def test_assign_held_out_pdb_split_keeps_same_complex_together_across_pdb_codes():
    """`1KIQ` and `1VFB` hash to opposite sides of `val_fraction=0.1` under
    the old (buggy) per-`pdb_id` grouping -- this is exactly the leakage this
    fix addresses -- but both records share one `complex_name`, so grouping
    by `_record_complex_key` must place them on the same side regardless.
    """
    from data.splitting import assign_held_out_pdb_split

    records = [
        _make_record("s1", "1VFB", "IgG1-kappa D1.3 Fv :: HEW lysozyme", (874, 861)),
        _make_record("s2", "1KIQ", "IgG1-kappa D1.3 Fv :: HEW lysozyme", (875, 862)),
        _make_record("s3", "2NZ9", "AR2 mab :: BoNT/A1", (874, 861)),
    ]
    assignment = assign_held_out_pdb_split(records, val_fraction=0.1)

    assert assignment["s1"] == assignment["s2"] == "val"
    assert assignment["s3"] == "train"


def test_assign_same_pdb_allowed_split_holds_out_some_complexes_entirely_and_mixes_others():
    """Fixture chosen so the outcome is deterministic under the real
    `_stable_unit_interval_hash`/`RANDOM_SEED`: "IgG1-kappa ... " sorts before
    "AR2 mab ..." in hash order and is the only complex small enough (1
    record) to fit under `new_complex_fraction=0.1`'s target (0.5 of 5
    records), so it alone is held out entirely; the remaining 4 "AR2 mab"
    records' per-sample hashes split 2-and-2 across train/val at
    `shared_sample_fraction=0.4`.
    """
    from data.splitting import assign_same_pdb_allowed_split

    records = [
        _make_record("s1", "1VFB", "IgG1-kappa D1.3 Fv :: HEW lysozyme", (1, 1)),
        _make_record("s2", "2NZ9", "AR2 mab :: BoNT/A1", (2, 2)),
        _make_record("s3", "2NZ9", "AR2 mab :: BoNT/A1", (3, 3)),
        _make_record("s4", "2NZ9", "AR2 mab :: BoNT/A1", (4, 4)),
        _make_record("s5", "2NZ9", "AR2 mab :: BoNT/A1", (5, 5)),
    ]
    assignment = assign_same_pdb_allowed_split(records, new_complex_fraction=0.1, shared_sample_fraction=0.4)

    assert assignment["s1"] == "val"
    ar2_assignments = {assignment[sample_id] for sample_id in ("s2", "s3", "s4", "s5")}
    assert ar2_assignments == {"train", "val"}


def test_mutation_record_csv_row_json_encodes_mutations_list():
    import json

    from data.splitting import mutation_record_to_csv_row

    record = _make_record("s1", "2NZ9", "AR2 mab :: BoNT/A1", (874, 861), (875, 862))
    row = mutation_record_to_csv_row(record)

    assert isinstance(row["mutations"], str)
    decoded = json.loads(row["mutations"])
    assert len(decoded) == 2
    assert decoded[0]["residue_position"] == 874
    assert "split_membership" not in row


def test_embedding_source_mode_gates_which_backend_actually_runs(monkeypatch):
    from data.embedding_pipeline import entry_embeddings
    from shared.constants import EmbeddingSourceMode

    monkeypatch.setattr(entry_embeddings, "ACTIVE_EMBEDDING_SOURCE_MODE", EmbeddingSourceMode.SEQUENCE_ONLY)
    assert entry_embeddings.should_compute_sequence_embeddings() is True
    assert entry_embeddings.should_compute_structure_embeddings() is False

    monkeypatch.setattr(entry_embeddings, "ACTIVE_EMBEDDING_SOURCE_MODE", EmbeddingSourceMode.STRUCTURE_ONLY)
    assert entry_embeddings.should_compute_sequence_embeddings() is False
    assert entry_embeddings.should_compute_structure_embeddings() is True

    monkeypatch.setattr(entry_embeddings, "ACTIVE_EMBEDDING_SOURCE_MODE", EmbeddingSourceMode.STRUCTURE_AND_SEQUENCE)
    assert entry_embeddings.should_compute_sequence_embeddings() is True
    assert entry_embeddings.should_compute_structure_embeddings() is True


def test_compute_wt_structure_bundle_omits_embedding_under_sequence_only(monkeypatch):
    import numpy as np

    from data.embedding_pipeline import entry_embeddings
    from shared.constants import ChainRole, EmbeddingSourceMode

    monkeypatch.setattr(entry_embeddings, "ACTIVE_EMBEDDING_SOURCE_MODE", EmbeddingSourceMode.SEQUENCE_ONLY)
    monkeypatch.setattr(entry_embeddings, "get_chain", lambda structure, chain_id: chain_id)
    monkeypatch.setattr(entry_embeddings, "chain_ca_coordinates", lambda chain: np.zeros((5, 3), dtype=np.float32))
    monkeypatch.setattr(
        entry_embeddings, "compute_real_structure_features", lambda structure, chain_id: np.zeros((5, 8), dtype=np.float32)
    )

    bundle = entry_embeddings.compute_wt_structure_bundle(
        structure=None, ordered_chain_ids=["A"], chain_role_by_id={"A": ChainRole.ANTIGEN}
    )

    assert "residue_coordinates" in bundle
    assert "wt_real_structure_features" in bundle
    assert "wt_structure_embedding" not in bundle
    assert "wt_structure_plddt_diagnostic" not in bundle


def test_compute_mut_structure_bundle_omits_embedding_under_sequence_only(monkeypatch):
    import numpy as np

    from data.embedding_pipeline import entry_embeddings
    from shared.constants import ChainRole, EmbeddingSourceMode

    monkeypatch.setattr(entry_embeddings, "ACTIVE_EMBEDDING_SOURCE_MODE", EmbeddingSourceMode.SEQUENCE_ONLY)
    mutations = [
        PointMutation(
            chain_id="A", chain_role=ChainRole.HEAVY, wt_residue="A", mutant_residue="G",
            residue_position=1, flat_residue_index=2,
        )
    ]

    bundle = entry_embeddings.compute_mut_structure_bundle(
        structure=None,
        ordered_chain_ids=["A"],
        chain_role_by_id={"A": ChainRole.HEAVY},
        mutations_by_chain_id={},
        mutations=mutations,
        residue_coordinates=np.random.randn(5, 3).astype(np.float32),
    )

    assert "mutation_distances" in bundle
    assert "mut_structure_embedding" not in bundle
    assert "mut_structure_confidence" not in bundle


def test_structure_embeddings_by_chain_routes_antibody_to_igfold_and_antigen_to_zero(monkeypatch):
    import numpy as np

    from data.embedding_pipeline import entry_embeddings
    from data.embedding_pipeline.structure_backend import STRUCTURE_EMBEDDING_COMMON_DIM

    def fake_antibody_embed(chain_sequences):
        return {
            chain_id: (np.full((len(seq), STRUCTURE_EMBEDDING_COMMON_DIM), 7.0, dtype=np.float32), np.full(len(seq), 0.9, dtype=np.float32))
            for chain_id, seq in chain_sequences.items()
        }

    embeddings, confidences = entry_embeddings._structure_embeddings_by_chain(
        ordered_chain_ids=["H", "AG"],
        antibody_sequences_by_chain_id={"H": "AAAA"},
        fallback_sequence_for_chain=lambda cid: "AAAAAA",
        antibody_embed_fn=fake_antibody_embed,
    )

    assert embeddings.shape == (4 + 6, STRUCTURE_EMBEDDING_COMMON_DIM)
    assert (embeddings[:4] == 7.0).all()
    assert (confidences[:4] == 0.9).all()
    assert (embeddings[4:] == 0.0).all()
    assert (confidences[4:] == 0.0).all()


def test_pad_structure_embedding_to_common_dim():
    import numpy as np

    from data.embedding_pipeline.structure_backend import (
        STRUCTURE_EMBEDDING_COMMON_DIM,
        pad_structure_embedding_to_common_dim,
    )

    small = np.ones((5, 64), dtype=np.float32)
    padded = pad_structure_embedding_to_common_dim(small)
    assert padded.shape == (5, STRUCTURE_EMBEDDING_COMMON_DIM)
    assert (padded[:, :64] == 1.0).all()
    assert (padded[:, 64:] == 0.0).all()

    already_full = np.ones((5, STRUCTURE_EMBEDDING_COMMON_DIM), dtype=np.float32)
    assert pad_structure_embedding_to_common_dim(already_full).shape == already_full.shape

    with pytest.raises(ValueError):
        pad_structure_embedding_to_common_dim(np.ones((5, STRUCTURE_EMBEDDING_COMMON_DIM + 1), dtype=np.float32))


_RAW_DATA_AVAILABLE = SKEMPI_RAW_CSV.exists() and SKEMPI_RAW_STRUCTURES_DIR.exists()


@pytest.mark.skipif(not _RAW_DATA_AVAILABLE, reason="raw SKEMPI data not downloaded in this environment")
def test_end_to_end_pipeline_on_local_raw_data():
    from data.pipeline import run_preprocessing_pipeline

    records, report = run_preprocessing_pipeline(download=False)

    assert report.num_antibody_antigen_rows > 0
    assert report.num_samples > 0
    assert len(records) == report.num_samples

    bounded = [r for r in records if r.label_type == LabelType.BOUNDED]
    assert bounded
    assert all(r.ddg_kcal_mol is not None and r.ddg_bin is not None for r in bounded)
    assert all(r.temperature_kelvin is not None for r in records)

    assert all(len(r.mutations) >= 1 for r in records)
    assert any(len(r.mutations) > 1 for r in records)

    assert any(r.split_membership.get("held_out_pdb") == "val" for r in records)
    assert any(r.split_membership.get("same_pdb_allowed") == "val" for r in records)
    assert all(m.flat_residue_index is not None for r in records for m in r.mutations)

    held_out_pdb_sides_by_complex: dict[str, set[str]] = {}
    for record in records:
        complex_key = normalize_protein_name(record.complex_name or "")
        held_out_pdb_sides_by_complex.setdefault(complex_key, set()).add(record.split_membership.get("held_out_pdb"))
    assert all(len(sides) == 1 for sides in held_out_pdb_sides_by_complex.values())

    same_pdb_allowed_train_complexes = {
        normalize_protein_name(r.complex_name or "") for r in records if r.split_membership.get("same_pdb_allowed") == "train"
    }
    same_pdb_allowed_val_complexes = {
        normalize_protein_name(r.complex_name or "") for r in records if r.split_membership.get("same_pdb_allowed") == "val"
    }
    assert same_pdb_allowed_val_complexes - same_pdb_allowed_train_complexes

    assert report.num_homology_sibling_samples_discarded > 0
    assert homology_sibling_sample_ids(records) == set()
