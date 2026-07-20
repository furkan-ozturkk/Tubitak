#!/usr/bin/env python3
"""
loghub_datasets.py

Section 7.1: curated DatasetSpec/LiteralSpec per LogHub dataset. Candidate
literals/regexes below were chosen by inspecting the real fetched
*_2k.log content (not guessed); exact match counts are always recomputed
at generation time by question_generators.py, never hardcoded here. Any
candidate with too few matches is pruned automatically
at runtime (Section 3.2: "a generous candidate list is safe").
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LiteralSpec:
    literal: str
    case_sensitive: bool = False


@dataclass(frozen=True)
class LookupSpec:
    literal: str
    position: str  # "first" | "last"
    case_sensitive: bool = False


@dataclass(frozen=True)
class HardGroupSpec:
    """
    Declarative spec for one hard-tier question template (Section 7.3).
    Lines are grouped by extract_key_regex's named group "key"; the
    num_groups largest groups (by line count, ties broken by key name for
    determinism) with >= min_lines_per_group lines become the >=2 evidence
    groups referenced by one hard question.
    """
    spec_id: str
    task: str
    extract_key_regex: str
    min_lines_per_group: int
    num_groups: int
    evidence_lines_per_group: int
    question_template: str  # uses {key0}, {key1}, ... placeholders


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    log_filename: str
    count_literals: tuple = ()
    presence_literals: tuple = ()
    lookup_specs: tuple = ()
    medium_anchor_literal: str = ""
    hard_groups: tuple = ()


DATASET_SPECS = {
    "Linux": DatasetSpec(
        name="Linux",
        log_filename="Linux_2k.log",
        count_literals=(
            LiteralSpec("authentication failure"),
        ),
        presence_literals=(
            LiteralSpec("Invalid user"),  # 0 hits in the 2k sample -> a "no" example
        ),
        lookup_specs=(
            LookupSpec("authentication failure", position="first"),
        ),
        medium_anchor_literal="check pass; user unknown",
        hard_groups=(
            HardGroupSpec(
                spec_id="brute_force_compare",
                task="RootCauseAnalysis",
                extract_key_regex=r"rhost=(?P<key>[0-9A-Za-z\.\-]+)",
                min_lines_per_group=5,
                num_groups=2,
                evidence_lines_per_group=4,
                question_template=(
                    "Compare the SSH authentication attempts coming from {key0} and "
                    "{key1}: contrast their volume and timing. Which source looks more "
                    "anomalous, and what is your root-cause hypothesis?"
                ),
            ),
        ),
    ),
    "Apache": DatasetSpec(
        name="Apache",
        log_filename="Apache_2k.log",
        count_literals=(
            LiteralSpec("[notice]"),
        ),
        presence_literals=(
            LiteralSpec("mod_jk"),
        ),
        lookup_specs=(
            LookupSpec("child workerEnv in error state", position="first"),
        ),
        medium_anchor_literal="child workerEnv in error state",
    ),
    "Windows": DatasetSpec(
        name="Windows",
        log_filename="Windows_2k.log",
        count_literals=(
            LiteralSpec("CBS"),
        ),
        presence_literals=(
            LiteralSpec("Error"),  # 0 hits -> a "no" example
        ),
        lookup_specs=(
            LookupSpec("CSI", position="first"),
        ),
        medium_anchor_literal="CSI",
    ),
    "Mac": DatasetSpec(
        name="Mac",
        log_filename="Mac_2k.log",
        count_literals=(
            LiteralSpec("kernel"),
        ),
        presence_literals=(
            LiteralSpec("Thermal pressure"),
        ),
        lookup_specs=(
            LookupSpec("IOThunderboltSwitch", position="first"),
        ),
        medium_anchor_literal="Thermal pressure",
    ),
    "HDFS": DatasetSpec(
        name="HDFS",
        log_filename="HDFS_2k.log",
        count_literals=(
            LiteralSpec("PacketResponder"),
        ),
        presence_literals=(
            LiteralSpec("Exception"),  # 0 hits -> a "no" example
        ),
        lookup_specs=(
            LookupSpec("addStoredBlock", position="first"),
        ),
        medium_anchor_literal="addStoredBlock",
        hard_groups=(
            HardGroupSpec(
                spec_id="block_lifecycle_compare",
                task="Correlation",
                extract_key_regex=r"(?P<key>blk_-?\d+)",
                min_lines_per_group=2,
                num_groups=2,
                evidence_lines_per_group=4,
                question_template=(
                    "Compare the lifecycle events recorded for block {key0} and block "
                    "{key1}. Correlate the sequence of events for each block and explain "
                    "whether either one shows an unusual replication/termination pattern."
                ),
            ),
        ),
    ),
    "OpenSSH": DatasetSpec(
        name="OpenSSH",
        log_filename="OpenSSH_2k.log",
        count_literals=(
            LiteralSpec("Failed password"),
        ),
        presence_literals=(
            LiteralSpec("POSSIBLE BREAK-IN ATTEMPT"),
        ),
        lookup_specs=(
            LookupSpec("POSSIBLE BREAK-IN ATTEMPT", position="first"),
        ),
        medium_anchor_literal="POSSIBLE BREAK-IN ATTEMPT",
        hard_groups=(
            HardGroupSpec(
                spec_id="attacker_compare",
                task="RootCauseAnalysis",
                extract_key_regex=r"from (?P<key>[0-9]{1,3}(?:\.[0-9]{1,3}){3})",
                min_lines_per_group=5,
                num_groups=2,
                evidence_lines_per_group=4,
                question_template=(
                    "Compare the attack patterns from source {key0} and source {key1} "
                    "(target accounts, request cadence, outcome). Which one is the more "
                    "anomalous attacker, and what is your root-cause hypothesis?"
                ),
            ),
        ),
    ),
    "BGL": DatasetSpec(
        name="BGL",
        log_filename="BGL_2k.log",
        count_literals=(
            LiteralSpec("double-hummer alignment exceptions"),
        ),
        presence_literals=(
            LiteralSpec("KERNDTLB"),
        ),
        lookup_specs=(
            LookupSpec("double-hummer alignment exceptions", position="first"),
        ),
        medium_anchor_literal="instruction cache parity error corrected",
        hard_groups=(
            HardGroupSpec(
                spec_id="hardware_error_chain",
                task="RootCauseAnalysis",
                extract_key_regex=r"(?P<key>double-hummer alignment exceptions|instruction cache parity error corrected)",
                min_lines_per_group=5,
                num_groups=2,
                evidence_lines_per_group=4,
                question_template=(
                    "Correlate the occurrences of '{key0}' and '{key1}' across the "
                    "observation window. Propose a root-cause hypothesis for why this "
                    "supercomputer node is showing both error types."
                ),
            ),
        ),
    ),
    "Hadoop": DatasetSpec(
        name="Hadoop",
        log_filename="Hadoop_2k.log",
        count_literals=(
            LiteralSpec("Container"),
        ),
        presence_literals=(
            LiteralSpec("MRAppMaster"),
        ),
        lookup_specs=(
            LookupSpec("MRAppMaster", position="first"),
        ),
        medium_anchor_literal="MRAppMaster",
        hard_groups=(
            HardGroupSpec(
                spec_id="container_task_compare",
                task="Correlation",
                extract_key_regex=r"(?P<key>container_[0-9_]+)",
                min_lines_per_group=3,
                num_groups=2,
                evidence_lines_per_group=4,
                question_template=(
                    "Correlate the task events logged for {key0} and {key1}. Do their "
                    "event sequences suggest a normal execution, or does one of them show "
                    "signs of a task/container failure? Give your root-cause hypothesis."
                ),
            ),
        ),
    ),
    "Zookeeper": DatasetSpec(
        name="Zookeeper",
        log_filename="Zookeeper_2k.log",
        count_literals=(
            LiteralSpec("WARN"),
        ),
        presence_literals=(
            LiteralSpec("Exception"),
        ),
        lookup_specs=(
            LookupSpec("FastLeaderElection", position="first"),
        ),
        medium_anchor_literal="FastLeaderElection",
        hard_groups=(
            HardGroupSpec(
                spec_id="ensemble_member_compare",
                task="Correlation",
                extract_key_regex=r"myid=(?P<key>\d+)",
                min_lines_per_group=5,
                num_groups=2,
                evidence_lines_per_group=4,
                question_template=(
                    "Compare the coordination activity logged for ensemble member myid={key0} "
                    "and myid={key1}. Correlate their event sequences and explain whether "
                    "either member's behavior looks anomalous for a Zookeeper ensemble."
                ),
            ),
        ),
    ),
    "OpenStack": DatasetSpec(
        name="OpenStack",
        log_filename="OpenStack_2k.log",
        count_literals=(
            LiteralSpec("status: 200"),
        ),
        presence_literals=(
            LiteralSpec("ERROR"),  # 0 hits -> a "no" example
        ),
        lookup_specs=(
            LookupSpec("status: 200", position="first"),
        ),
        medium_anchor_literal="nova.osapi_compute.wsgi.server",
    ),
}
