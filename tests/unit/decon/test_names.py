# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""U-DECON item 6: eval names as stored in the release's ``decon_matches``."""

import pytest

from atlantic_cc.decon.names import decon_matches, eval_name


@pytest.mark.parametrize(
    ("eval_id", "name"),
    [
        ("mmlu_abstract_algebra-test-5", "mmlu_abstract_algebra"),
        ("gsm8k-test-0", "gsm8k"),
        ("squadv2-validation-11872", "squadv2"),
        ("aa_lcr-1", "aa_lcr"),
        ("gdpval_aa_v2-12", "gdpval_aa_v2"),
        ("hle-66b2c7c9795022550abb176b", "hle"),
        ("paloma_c4_en-17", "paloma_c4_en"),
        ("lima-0", "lima"),
        (
            "include_base_44_north macedonian_few_shot_og_arts_humanities-test-3",
            "include_base_44_north macedonian_few_shot_og_arts_humanities",
        ),
        # Kept whole: the last segment is neither digits nor >= 16 hex digits.
        ("scicode-77.1", "scicode-77.1"),
        ("terminal_bench_2_1-crack-7z-hash", "terminal_bench_2_1-crack-7z-hash"),
        # The lm-eval split "training" is not stripped.
        ("xyz-training-3", "xyz-training"),
    ],
)
def test_eval_name(eval_id, name):
    assert eval_name(eval_id) == name


def test_decon_matches_is_the_sorted_union_over_every_record_sharing_a_hash():
    names_of = {1: ["mmlu", "mmlu_pro"], 2: ["mmlu"], 3: ["paloma_4chan"]}
    assert decon_matches([2, 3, 1], names_of) == ["mmlu", "mmlu_pro", "paloma_4chan"]
    assert decon_matches([4], names_of) == []
    assert decon_matches([], names_of) == []
