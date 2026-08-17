"""Curated per-dataset question specs (Section 7.1).

One ``DatasetSpec`` per LogHub dataset, declaring the literals the easy tier
counts and looks up, the anchors the medium tier explains, and the grouping rules
the hard tier compares or correlates across.

The literals and regexes were chosen by inspecting the real fetched ``*_2k.log``
content, never guessed — but nothing here asserts how often they occur. Exact
match counts are recomputed against Postgres at generation time, and any
candidate with too few matches is pruned automatically, which is what makes a
generous candidate list safe (Section 3.2): an over-optimistic literal costs a
pruned question, never a wrong answer.

Every dataset's ``presence_literals`` now carries two entries: one that matches
(a "Yes" example) and one verified absent from the real 2k sample (a "No"
example) — a dataset holding only "Yes" presence questions cannot tell a router
that always answers "Yes" from one that actually checks. ``medium_anchors`` carries
two entries per dataset for the same reason applied to event *type* rather than
verdict: a dataset anchored on one literal only ever explains one event family.

Two hard-tier curation decisions worth their history. ``HDFS`` carries no
``hard_correlation`` spec: no block-id/datanode-address pair proven linked on
one line reaches three lines on both sides in the 2k sample — measured, not
assumed, the same reason the earlier ``block_lifecycle_compare`` spec
(``min_lines_per_group=2``) was retired rather than relaxed further. It does
carry a ``hard_comparative`` spec now, keyed on the DataNode address rather
than the block id that comparison was first tried against: a block id rarely
exceeds four lines in the 2k sample, but the address a block's own
``DataXceiver``/``FSNamesystem`` lines name (``src:``/``is added to``) reaches
15-20 lines for the busiest nodes — the same corpus, a richer key. ``BGL``'s
comparative spec keys groups by compute-node id, not by error-message text: an
earlier revision grouped on the message literals themselves, which made the two
"entities" two phrasings of the question and collided with the easy tier's count
literal. The node ids give real entities with 30-60 matching lines each.

``Apache``, ``Mac`` and ``OpenStack`` each carry a ``hard_comparative`` spec added
after inspecting their real 2k samples for a key with both enough matching lines
(``>=5``) and enough distinct qualifying values to fill at least one pair.
``Windows`` was checked the same way and excluded: its richest recurring value
(a servicing package id) tops out at four lines in the 2k sample, and every
occurrence of it reads identically (``ApplicableState: 112, CurrentState:112``),
so neither the volume nor the content supports a comparison — the same
"measured, not assumed" standard HDFS's block id failed by is why Windows has
no hard tier at all rather than one forced to fit. ``Apache``'s key is the mod_jk
error-state code (``error state N``) rather than a source address or session id:
the 2k sample's only other volume candidate, ``scoreboard slot N``, groups lines
that are close to verbatim identical worker-registration notices with nothing to
compare beyond timestamps, where the error-state code carries a real per-value
signal -- state 6 recurs roughly 3.5x as often as state 7 in the same file.

``Hadoop`` is the one dataset with a ``hard_correlation`` spec so far: a
container-launch line in the raw log explicitly names both a container id and
the task attempt it was launched for (``"...for container container_X
taskAttempt attempt_Y..."``), which is the proof a correlation spec's
``key_a_regex``/``key_b_regex`` pair requires — the link is not assumed by
picking two entities side by side, it is stated in the log itself. No other
dataset in the 2k samples was found with the same property at
``min_lines_per_group>=3`` on both sides (checked against HDFS's block/datanode
pairing and Zookeeper's session/``myid`` pairing, both too sparse); a dataset
simply not supporting correlation is the same kind of measured fact as HDFS not
supporting comparison.
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
class MediumAnchorSpec:
    """One curated anchor the medium tier builds evidence windows around.

    Attributes:
        literal: Substring to search for; every occurrence is a candidate
            window center.
        event_type: A short, human-readable label for the kind of event this
            anchor represents (e.g. ``"cache_parity_error"``). Documents why
            this anchor is a distinct event family from a dataset's other
            anchors; not used inside the drafting prompt itself, since the
            model derives its own ``EVENT_TYPE`` from the evidence rather than
            being told one.
    """

    literal: str
    event_type: str


@dataclass(frozen=True)
class HardComparativeSpec:
    """Declarative spec for one hard-comparative question template (Section 7.3).

    Lines are grouped by ``extract_key_regex``'s named group ``key``. The
    ``num_groups`` largest groups holding at least ``min_lines_per_group`` lines,
    ranked by line count with the key name breaking ties for determinism, become
    the two same-kind entities one comparative question contrasts.

    Attributes:
        spec_id: Stable identifier, used in the question id and the validation
            report filename.
        task: Task label recorded on the record, e.g. ``"RootCauseAnalysis"``.
        extract_key_regex: Regex with a named ``key`` group that identifies the
            entity a line belongs to (a block id, container id, source address).
        min_lines_per_group: Groups below this are discarded rather than padded; a
            comparison over two lines compares nothing.
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
class HardCorrelationSpec:
    """Declarative spec for one hard-correlation question template (Section 7.3).

    Unlike a comparative spec, the two entities a correlation question links are
    not two same-kind values of one regex — they can be different kinds of
    entity entirely (a container id and a task attempt id) — and the link
    between them is not assumed by picking two side by side. It is *proven*: at
    least one line in the corpus must match both ``key_a_regex`` and
    ``key_b_regex`` at once, which is what establishes that the two extracted
    values genuinely refer to related things rather than two entities that
    happen to coexist in the same file.

    Attributes:
        spec_id: Stable identifier, used in the question id and the validation
            report filename.
        task: Task label recorded on the record, conventionally
            ``"Correlation"``.
        key_a_regex: Regex with a named ``key`` group for the first entity kind.
        key_b_regex: Regex with a named ``key`` group for the second entity kind.
        min_lines_per_group: Each side of a proven pair must independently reach
            this many matching lines across the whole file (not just the proof
            line) for the pair to qualify.
        question_templates: Question phrasings, with ``{key0}`` filled from the
            proven value of ``key_a_regex`` and ``{key1}`` from ``key_b_regex``.
    """

    spec_id: str
    task: str
    key_a_regex: str
    key_b_regex: str
    min_lines_per_group: int
    question_templates: tuple


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the three tiers need to know about one LogHub dataset.

    A tier contributes nothing for a dataset whose corresponding field is empty,
    which is deliberate: not every log format carries an entity worth comparing
    or correlating, and a fabricated one would produce a question about whatever
    happened to be at the top of the file.

    Attributes:
        name: Dataset name as LogHub publishes it.
        log_filename: The fetched ``*_2k.log`` file.
        count_literals: Literals the easy tier counts.
        presence_literals: Literals the easy tier tests for presence. Convention
            (Section 7.1) is two entries per dataset: one that matches in the
            real 2k sample and one verified absent, so presence questions are
            not all "Yes". Whether one occurs is recomputed at generation time
            regardless of which case a literal was curated for.
        lookup_specs: First/last line lookups for the easy tier.
        medium_anchors: Anchors the medium tier builds evidence windows around;
            convention is two per dataset, each a distinct event family.
        hard_comparative: Hard-comparative question templates (two same-kind
            entities contrasted).
        hard_correlation: Hard-correlation question templates (two entities
            whose link is proven by a shared line, not assumed).
    """

    name: str
    log_filename: str
    count_literals: tuple = ()
    presence_literals: tuple = ()
    lookup_specs: tuple = ()
    medium_anchors: tuple = ()
    hard_comparative: tuple = ()
    hard_correlation: tuple = ()


DATASET_SPECS: dict[str, "DatasetSpec"] = {
    "Linux": DatasetSpec(
        name="Linux",
        log_filename="Linux_2k.log",
        count_literals=(LiteralSpec("authentication failure"),),
        presence_literals=(
            LiteralSpec("Invalid user"),
            LiteralSpec("session opened"),
        ),
        lookup_specs=(LookupSpec("authentication failure", position="first"),),
        medium_anchors=(
            MediumAnchorSpec("check pass; user unknown", event_type="auth_failure"),
            MediumAnchorSpec("session opened", event_type="session_lifecycle"),
        ),
        hard_comparative=(
            HardComparativeSpec(
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
        presence_literals=(
            LiteralSpec("mod_jk"),
            LiteralSpec("Segmentation fault"),
        ),
        lookup_specs=(LookupSpec("child workerEnv in error state", position="first"),),
        medium_anchors=(
            MediumAnchorSpec(
                "child workerEnv in error state", event_type="worker_error"
            ),
            MediumAnchorSpec("jk2_init", event_type="jk2_worker_init"),
        ),
        hard_comparative=(
            HardComparativeSpec(
                spec_id="error_state_compare",
                task="Correlation",
                extract_key_regex=r"error state (?P<key>\d+)",
                min_lines_per_group=5,
                num_groups=2,
                question_templates=(
                    "Compare how often mod_jk reports error state {key0} versus "
                    "error state {key1} in this log. Which code recurs more "
                    "persistently, and what does that suggest about the "
                    "underlying worker problem?",
                    "How does the recurrence of error state {key0} compare to "
                    "error state {key1}? Which looks like the more persistent "
                    "issue, and why?",
                    "Contrast the occurrences of error state {key0} and error "
                    "state {key1} in this Apache log. Which code's pattern "
                    "looks more concerning?",
                    "Error state {key0} and error state {key1} both appear "
                    "repeatedly in this log. Compare their frequency and "
                    "timing, and say which points to a more persistent "
                    "problem.",
                    "Looking at error state {key0} versus error state {key1}, "
                    "which one's recurrence pattern looks more like an "
                    "unresolved issue, and what's your reasoning?",
                ),
            ),
        ),
    ),
    "Windows": DatasetSpec(
        name="Windows",
        log_filename="Windows_2k.log",
        count_literals=(LiteralSpec("CBS"),),
        presence_literals=(
            LiteralSpec("Error"),
            LiteralSpec("Access is denied"),
        ),
        lookup_specs=(LookupSpec("CSI", position="first"),),
        medium_anchors=(
            MediumAnchorSpec("CSI", event_type="csi_servicing_trace"),
            MediumAnchorSpec(
                "Failed to internally open", event_type="internal_open_failure"
            ),
        ),
    ),
    "Mac": DatasetSpec(
        name="Mac",
        log_filename="Mac_2k.log",
        count_literals=(LiteralSpec("kernel"),),
        presence_literals=(
            LiteralSpec("Thermal pressure"),
            LiteralSpec("kernel panic"),
        ),
        lookup_specs=(LookupSpec("IOThunderboltSwitch", position="first"),),
        medium_anchors=(
            MediumAnchorSpec("Thermal pressure", event_type="thermal_pressure"),
            MediumAnchorSpec("wake reason", event_type="sleep_wake"),
        ),
        hard_comparative=(
            HardComparativeSpec(
                spec_id="device_activity_compare",
                task="RootCauseAnalysis",
                extract_key_regex=r"^\S+\s+\S+\s+\S+\s+(?P<key>[\w.-]+)\s",
                min_lines_per_group=5,
                num_groups=2,
                question_templates=(
                    "Compare the system activity logged for device {key0} and "
                    "device {key1}. Which one shows a more troubling pattern "
                    "of kernel or process events, and what is your root-cause "
                    "hypothesis?",
                    "How does the logged activity on {key0} differ from "
                    "{key1}? Which device's behavior looks more anomalous, "
                    "and why?",
                    "Contrast the kernel/process events on devices {key0} and "
                    "{key1}. Which one's activity is more concerning?",
                    "{key0} and {key1} both appear as sources of events in "
                    "this log. Compare their activity and identify which "
                    "looks more anomalous.",
                    "Looking at the events logged for {key0} versus {key1}, "
                    "which device shows a more troubling pattern, and what's "
                    "your reasoning?",
                ),
            ),
        ),
    ),
    "HDFS": DatasetSpec(
        name="HDFS",
        log_filename="HDFS_2k.log",
        count_literals=(LiteralSpec("PacketResponder"),),
        presence_literals=(
            LiteralSpec("Exception"),
            LiteralSpec("OutOfMemoryError"),
        ),
        lookup_specs=(LookupSpec("addStoredBlock", position="first"),),
        medium_anchors=(
            MediumAnchorSpec("addStoredBlock", event_type="block_registered"),
            MediumAnchorSpec("Deleting block", event_type="block_deletion"),
        ),
        hard_comparative=(
            HardComparativeSpec(
                spec_id="datanode_activity_compare",
                task="RootCauseAnalysis",
                extract_key_regex=r"(?P<key>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
                min_lines_per_group=5,
                num_groups=2,
                question_templates=(
                    "Compare the block activity logged for DataNode {key0} and "
                    "DataNode {key1}. Which one shows a more troubling pattern "
                    "of block operations, and what is your root-cause "
                    "hypothesis?",
                    "How does the logged block activity on {key0} differ from "
                    "{key1}? Which DataNode's behavior looks more anomalous, "
                    "and why?",
                    "Contrast the block-related events on DataNodes {key0} "
                    "and {key1}. Which one's activity is more concerning?",
                    "{key0} and {key1} both appear as DataNode addresses in "
                    "this log. Compare their block activity and identify "
                    "which looks more anomalous.",
                    "Looking at the block events logged for {key0} versus "
                    "{key1}, which DataNode shows a more troubling pattern, "
                    "and what's your reasoning?",
                ),
            ),
        ),
    ),
    "OpenSSH": DatasetSpec(
        name="OpenSSH",
        log_filename="OpenSSH_2k.log",
        count_literals=(LiteralSpec("Failed password"),),
        presence_literals=(
            LiteralSpec("POSSIBLE BREAK-IN ATTEMPT"),
            LiteralSpec("maximum authentication attempts"),
        ),
        lookup_specs=(LookupSpec("POSSIBLE BREAK-IN ATTEMPT", position="first"),),
        medium_anchors=(
            MediumAnchorSpec(
                "POSSIBLE BREAK-IN ATTEMPT", event_type="break_in_warning"
            ),
            MediumAnchorSpec("Received disconnect", event_type="disconnect"),
        ),
        hard_comparative=(
            HardComparativeSpec(
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
        presence_literals=(
            LiteralSpec("KERNDTLB"),
            LiteralSpec("kernel panic"),
        ),
        lookup_specs=(
            LookupSpec("double-hummer alignment exceptions", position="first"),
        ),
        medium_anchors=(
            MediumAnchorSpec(
                "instruction cache parity error corrected",
                event_type="cache_parity_error",
            ),
            MediumAnchorSpec(
                "double-hummer alignment exceptions", event_type="alignment_exception"
            ),
        ),
        hard_comparative=(
            HardComparativeSpec(
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
        presence_literals=(
            LiteralSpec("MRAppMaster"),
            LiteralSpec("OutOfMemoryError"),
        ),
        lookup_specs=(LookupSpec("MRAppMaster", position="first"),),
        medium_anchors=(
            MediumAnchorSpec("MRAppMaster", event_type="app_master"),
            MediumAnchorSpec(
                "CONTAINER_REMOTE_LAUNCH", event_type="container_launch"
            ),
        ),
        hard_comparative=(
            HardComparativeSpec(
                spec_id="container_task_compare",
                task="RootCauseAnalysis",
                extract_key_regex=r"(?P<key>container_[0-9_]+)",
                min_lines_per_group=3,
                num_groups=2,
                question_templates=(
                    "Compare the task events logged for container {key0} and container "
                    "{key1}. Do their event sequences suggest a normal execution, or does "
                    "one of them show signs of a task/container failure? Give your "
                    "root-cause hypothesis.",
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
        hard_correlation=(
            HardCorrelationSpec(
                spec_id="container_attempt_link",
                task="Correlation",
                key_a_regex=r"(?P<key>container_[0-9_]+)",
                key_b_regex=r"(?P<key>attempt_[0-9_mr]+)",
                min_lines_per_group=3,
                question_templates=(
                    "Container {key0} and task attempt {key1} are linked in this "
                    "Hadoop log -- a line explicitly assigns {key0} to {key1}. "
                    "Correlate their event sequences: does the fuller history of "
                    "{key1} explain what happened to {key0}, or does one side show "
                    "something the other does not account for?",
                    "How does the event history of task attempt {key1} relate to "
                    "that of the container it was assigned to, {key0}? Correlate "
                    "their sequences and say whether either side shows a problem "
                    "the other doesn't explain.",
                    "Container {key0} was launched for task attempt {key1}. "
                    "Contrast what each side's log lines show -- do they tell a "
                    "consistent story, or does one reveal an issue the other's "
                    "lines do not?",
                    "{key0} and {key1} both appear in this Hadoop log, explicitly "
                    "linked by a container-launch line. Correlate their sequences "
                    "and identify whether either shows signs of a problem.",
                    "Given that {key0} was assigned to {key1}, examine both "
                    "entities' event sequences: are they consistent with a normal "
                    "execution, or does the correlation reveal something the "
                    "other side's story misses?",
                ),
            ),
        ),
    ),
    "Zookeeper": DatasetSpec(
        name="Zookeeper",
        log_filename="Zookeeper_2k.log",
        count_literals=(LiteralSpec("WARN"),),
        presence_literals=(
            LiteralSpec("Exception"),
            LiteralSpec("leader election timeout"),
        ),
        lookup_specs=(LookupSpec("FastLeaderElection", position="first"),),
        medium_anchors=(
            MediumAnchorSpec("FastLeaderElection", event_type="leader_election"),
            MediumAnchorSpec(
                "Cannot open channel", event_type="channel_connection_failure"
            ),
        ),
        hard_comparative=(
            HardComparativeSpec(
                spec_id="ensemble_member_compare",
                task="Comparison",
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
        presence_literals=(
            LiteralSpec("ERROR"),
            LiteralSpec("DELETE /v2"),
        ),
        lookup_specs=(LookupSpec("status: 200", position="first"),),
        medium_anchors=(
            MediumAnchorSpec(
                "nova.osapi_compute.wsgi.server", event_type="wsgi_request"
            ),
            MediumAnchorSpec("instance:", event_type="instance_operation"),
        ),
        hard_comparative=(
            HardComparativeSpec(
                spec_id="instance_activity_compare",
                task="RootCauseAnalysis",
                extract_key_regex=r"\[instance: (?P<key>[a-f0-9-]+)\]",
                min_lines_per_group=5,
                num_groups=2,
                question_templates=(
                    "Compare the activity logged for instance {key0} and "
                    "instance {key1}. Which one shows a more troubling "
                    "pattern of events, and what is your root-cause "
                    "hypothesis?",
                    "How does the logged activity for instance {key0} differ "
                    "from instance {key1}? Which instance's behavior looks "
                    "more anomalous, and why?",
                    "Contrast the events logged for instances {key0} and "
                    "{key1}. Which one's activity is more concerning?",
                    "Instances {key0} and {key1} both appear in this "
                    "OpenStack log. Compare their activity and identify "
                    "which looks more anomalous.",
                    "Looking at the events logged for instance {key0} versus "
                    "instance {key1}, which one shows a more troubling "
                    "pattern, and what's your reasoning?",
                ),
            ),
        ),
    ),
}
