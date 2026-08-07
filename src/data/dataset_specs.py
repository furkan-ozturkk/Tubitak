"""Curated per-dataset question specs (Section 7.1).

One ``DatasetSpec`` per LogHub dataset, declaring the literals the easy tier
counts and looks up, the anchor the medium tier explains, and the grouping rules
the hard tier correlates across.

The literals and regexes were chosen by inspecting the real fetched ``*_2k.log``
content, never guessed — but nothing here asserts how often they occur. Exact
match counts are recomputed against Postgres at generation time, and any
candidate with too few matches is pruned automatically, which is what makes a
generous candidate list safe (Section 3.2): an over-optimistic literal costs a
pruned question, never a wrong answer.

Two hard-tier curation decisions worth their history. ``HDFS`` carries no
``hard_groups``: in the 2k sample no block id appears on more than two lines, so
any block-vs-block "correlation" question would rest on two lines per side —
measured, not assumed, and the reason the earlier ``block_lifecycle_compare``
spec (``min_lines_per_group=2``) was retired rather than relaxed further.
``BGL``'s spec keys groups by compute-node id, not by error-message text: an
earlier revision grouped on the message literals themselves, which made the two
"entities" two phrasings of the question and collided with the easy tier's count
literal. The node ids give real entities with 30–60 matching lines each.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LiteralSpec:
    """One literal the easy tier counts or tests for presence.

    Attributes:
        literal: Substring to search for.
        case_sensitive: Whether matching is case sensitive. Left false for almost
            every spec, because the questions are phrased in natural language and a
            reader would not expect ``ERROR`` and ``Error`` to be different
            questions.
    """

    literal: str
    case_sensitive: bool = False


@dataclass(frozen=True)
class LookupSpec:
    """One first-or-last line lookup for the easy tier.

    Attributes:
        literal: Substring the looked-up line must contain.
        position: ``"first"`` or ``"last"`` matching line.
        case_sensitive: Whether matching is case sensitive.
    """

    literal: str
    position: str
    case_sensitive: bool = False


@dataclass(frozen=True)
class HardGroupSpec:
    """Declarative spec for one hard-tier question template (Section 7.3).

    Lines are grouped by ``extract_key_regex``'s named group ``key``. The
    ``num_groups`` largest groups holding at least ``min_lines_per_group`` lines,
    ranked by line count with the key name breaking ties for determinism, become the
    evidence groups one hard question correlates across.

    Attributes:
        spec_id: Stable identifier, used in the question id and the groundedness
            report filename.
        task: Task label recorded on the record, e.g. ``RootCauseAnalysis``.
        extract_key_regex: Regex with a named ``key`` group that identifies the
            entity a line belongs to (a block id, container id, source address).
        min_lines_per_group: Groups below this are discarded rather than padded; a
            correlation question over two lines correlates nothing.
        num_groups: How many groups the question needs. The schema requires at least
            two, so a spec below that would build records that cannot validate.
        question_templates: Question phrasings, each with ``{key0}``, ``{key1}``, ...
            placeholders filled from the selected group keys. A dataset drafting
            several group-sets from the same spec cycles through these so the
            questions are not all asked in identical words — the same reasoning
            the easy tier's phrasing families rest on, applied here even though
            ``validate.py`` does not enforce it on the semantic path.
    """

    spec_id: str
    task: str
    extract_key_regex: str
    min_lines_per_group: int
    num_groups: int
    question_templates: tuple


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the three tiers need to know about one LogHub dataset.

    A tier contributes nothing for a dataset whose corresponding field is empty,
    which is deliberate: not every log format carries an entity worth correlating,
    and a fabricated anchor would produce a question about whatever happened to be
    at the top of the file.

    Attributes:
        name: Dataset name as LogHub publishes it.
        log_filename: The fetched ``*_2k.log`` file.
        count_literals: Literals the easy tier counts.
        presence_literals: Literals the easy tier tests for presence. Whether one
            occurs in the 2k sample is not asserted here and is recomputed at
            generation time; ``Linux``'s ``Invalid user`` and ``OpenStack``'s
            ``ERROR`` currently yield the "No" answers, but that is a fact about the
            corpus, not a property of the spec.
        lookup_specs: First/last line lookups for the easy tier.
        medium_anchor_literal: Anchor the medium tier builds evidence windows around.
        hard_groups: Hard-tier question templates.
    """

    name: str
    log_filename: str
    count_literals: tuple = ()
    presence_literals: tuple = ()
    lookup_specs: tuple = ()
    medium_anchor_literal: str = ""
    hard_groups: tuple = ()


DATASET_SPECS: dict[str, "DatasetSpec"] = {
    "Linux": DatasetSpec(
        name="Linux",
        log_filename="Linux_2k.log",
        count_literals=(LiteralSpec("authentication failure"),),
        presence_literals=(LiteralSpec("Invalid user"),),
        lookup_specs=(LookupSpec("authentication failure", position="first"),),
        medium_anchor_literal="check pass; user unknown",
        hard_groups=(
            HardGroupSpec(
                spec_id="brute_force_compare",
                task="RootCauseAnalysis",
                extract_key_regex=r"rhost=(?P<key>[0-9A-Za-z\.\-]+)",
                min_lines_per_group=5,
                num_groups=2,
                question_templates=(
                    "Compare the SSH authentication attempts coming from {key0} and "
                    "{key1}: contrast their timing and the accounts they target. "
                    "Which source looks more anomalous, and what is your root-cause "
                    "hypothesis?",
                    "How do the login attempts from {key0} differ from those from "
                    "{key1}? Contrast their timing and target accounts, and say "
                    "which source looks more anomalous with your root-cause "
                    "hypothesis.",
                    "Contrast the authentication activity originating from {key0} "
                    "and {key1}. Which one shows more suspicious behavior, and why "
                    "do you think so?",
                    "{key0} and {key1} both appear as sources of failed logins in "
                    "this log. Compare their attack patterns and propose a "
                    "root-cause hypothesis for the more anomalous one.",
                    "Looking at the authentication attempts from {key0} versus "
                    "{key1}, which source's behavior looks more like an attack, "
                    "and what's your reasoning?",
                ),
            ),
        ),
    ),
    "Apache": DatasetSpec(
        name="Apache",
        log_filename="Apache_2k.log",
        count_literals=(LiteralSpec("[notice]"),),
        presence_literals=(LiteralSpec("mod_jk"),),
        lookup_specs=(LookupSpec("child workerEnv in error state", position="first"),),
        medium_anchor_literal="child workerEnv in error state",
    ),
    "Windows": DatasetSpec(
        name="Windows",
        log_filename="Windows_2k.log",
        count_literals=(LiteralSpec("CBS"),),
        presence_literals=(LiteralSpec("Error"),),
        lookup_specs=(LookupSpec("CSI", position="first"),),
        medium_anchor_literal="CSI",
    ),
    "Mac": DatasetSpec(
        name="Mac",
        log_filename="Mac_2k.log",
        count_literals=(LiteralSpec("kernel"),),
        presence_literals=(LiteralSpec("Thermal pressure"),),
        lookup_specs=(LookupSpec("IOThunderboltSwitch", position="first"),),
        medium_anchor_literal="Thermal pressure",
    ),
    "HDFS": DatasetSpec(
        name="HDFS",
        log_filename="HDFS_2k.log",
        count_literals=(LiteralSpec("PacketResponder"),),
        presence_literals=(LiteralSpec("Exception"),),
        lookup_specs=(LookupSpec("addStoredBlock", position="first"),),
        medium_anchor_literal="addStoredBlock",
    ),
    "OpenSSH": DatasetSpec(
        name="OpenSSH",
        log_filename="OpenSSH_2k.log",
        count_literals=(LiteralSpec("Failed password"),),
        presence_literals=(LiteralSpec("POSSIBLE BREAK-IN ATTEMPT"),),
        lookup_specs=(LookupSpec("POSSIBLE BREAK-IN ATTEMPT", position="first"),),
        medium_anchor_literal="POSSIBLE BREAK-IN ATTEMPT",
        hard_groups=(
            HardGroupSpec(
                spec_id="attacker_compare",
                task="RootCauseAnalysis",
                extract_key_regex=r"from (?P<key>[0-9]{1,3}(?:\.[0-9]{1,3}){3})",
                min_lines_per_group=5,
                num_groups=2,
                question_templates=(
                    "Compare the attack patterns from source {key0} and source {key1} "
                    "(target accounts, request cadence, outcome). Which one is the more "
                    "anomalous attacker, and what is your root-cause hypothesis?",
                    "How do the attack patterns from {key0} and {key1} differ in "
                    "terms of target accounts, request cadence, and outcome? Which "
                    "attacker looks more anomalous?",
                    "Contrast the intrusion attempts coming from {key0} and {key1}. "
                    "Which source behaves more suspiciously, and what is your "
                    "hypothesis for why?",
                    "{key0} and {key1} both show up as attacking sources in this "
                    "log. Compare their behavior and identify which one is the "
                    "more anomalous attacker.",
                    "Examine the break-in attempts from {key0} versus {key1}: how "
                    "do their targets and timing compare, and which looks like the "
                    "more serious threat?",
                ),
            ),
        ),
    ),
    "BGL": DatasetSpec(
        name="BGL",
        log_filename="BGL_2k.log",
        count_literals=(LiteralSpec("double-hummer alignment exceptions"),),
        presence_literals=(LiteralSpec("KERNDTLB"),),
        lookup_specs=(
            LookupSpec("double-hummer alignment exceptions", position="first"),
        ),
        medium_anchor_literal="instruction cache parity error corrected",
        hard_groups=(
            HardGroupSpec(
                spec_id="node_error_compare",
                task="RootCauseAnalysis",
                extract_key_regex=(
                    r"(?P<key>R\d{2}-M\d-N[0-9A-F]+-[A-Z]:J\d{2}-U\d{2})"
                ),
                min_lines_per_group=5,
                num_groups=2,
                question_templates=(
                    "Compare the RAS events logged for compute node {key0} and node "
                    "{key1}. Correlate the error patterns each node shows and propose "
                    "a root-cause hypothesis for the node whose behavior looks more "
                    "anomalous.",
                    "How do the RAS events for compute node {key0} compare to those "
                    "for node {key1}? Correlate their error patterns and say which "
                    "node looks more anomalous.",
                    "Contrast the hardware/error activity on nodes {key0} and "
                    "{key1}. Which node's behavior is more concerning, and what is "
                    "your root-cause hypothesis?",
                    "Nodes {key0} and {key1} both show RAS events in this log. "
                    "Compare their error patterns and identify the more anomalous "
                    "node.",
                    "Looking at the events logged for {key0} versus {key1}, which "
                    "compute node shows a more troubling pattern, and why?",
                ),
            ),
        ),
    ),
    "Hadoop": DatasetSpec(
        name="Hadoop",
        log_filename="Hadoop_2k.log",
        count_literals=(LiteralSpec("Container"),),
        presence_literals=(LiteralSpec("MRAppMaster"),),
        lookup_specs=(LookupSpec("MRAppMaster", position="first"),),
        medium_anchor_literal="MRAppMaster",
        hard_groups=(
            HardGroupSpec(
                spec_id="container_task_compare",
                task="Correlation",
                extract_key_regex=r"(?P<key>container_[0-9_]+)",
                min_lines_per_group=3,
                num_groups=2,
                question_templates=(
                    "Correlate the task events logged for {key0} and {key1}. Do their "
                    "event sequences suggest a normal execution, or does one of them show "
                    "signs of a task/container failure? Give your root-cause hypothesis.",
                    "How do the task event sequences for {key0} and {key1} compare? "
                    "Does either show signs of failure, and what's your root-cause "
                    "hypothesis?",
                    "Contrast the execution logs for containers {key0} and {key1}. "
                    "Which one, if either, shows signs of a task failure?",
                    "{key0} and {key1} both appear in this Hadoop log's task "
                    "events. Compare their sequences and say whether one shows "
                    "failure signs.",
                    "Examine the events for container {key0} versus {key1}: do "
                    "both reflect normal execution, or does one suggest a "
                    "problem? Explain your reasoning.",
                ),
            ),
        ),
    ),
    "Zookeeper": DatasetSpec(
        name="Zookeeper",
        log_filename="Zookeeper_2k.log",
        count_literals=(LiteralSpec("WARN"),),
        presence_literals=(LiteralSpec("Exception"),),
        lookup_specs=(LookupSpec("FastLeaderElection", position="first"),),
        medium_anchor_literal="FastLeaderElection",
        hard_groups=(
            HardGroupSpec(
                spec_id="ensemble_member_compare",
                task="Correlation",
                extract_key_regex=r"myid=(?P<key>\d+)",
                min_lines_per_group=5,
                num_groups=2,
                question_templates=(
                    "Compare the coordination activity logged for ensemble member myid={key0} "
                    "and myid={key1}. Correlate their event sequences and explain whether "
                    "either member's behavior looks anomalous for a Zookeeper ensemble.",
                    "How does the coordination activity for ensemble member "
                    "myid={key0} compare to myid={key1}? Does either look "
                    "anomalous for a Zookeeper ensemble?",
                    "Contrast the events logged for Zookeeper members myid={key0} "
                    "and myid={key1}. Which, if either, shows unusual behavior?",
                    "Members myid={key0} and myid={key1} both appear in this "
                    "ensemble's logs. Compare their activity and flag anything "
                    "anomalous.",
                    "Looking at the log events for myid={key0} versus myid={key1}, "
                    "does either ensemble member's behavior stand out as unusual?",
                ),
            ),
        ),
    ),
    "OpenStack": DatasetSpec(
        name="OpenStack",
        log_filename="OpenStack_2k.log",
        count_literals=(LiteralSpec("status: 200"),),
        presence_literals=(LiteralSpec("ERROR"),),
        lookup_specs=(LookupSpec("status: 200", position="first"),),
        medium_anchor_literal="nova.osapi_compute.wsgi.server",
    ),
}
