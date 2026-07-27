"""Leak-proof dev/test split assignment.

A split cannot be decided one record at a time. Section 6 requires that questions
derived from the same event share a split, and a hard question cites two or more
evidence groups at once, which links those groups: if one of them later appears in
a question that hashed to the other side, the same event is on both sides of the
boundary and the test set leaks.

Hashing each group independently does not prevent that, and neither does hashing a
record's group set — a record citing ``{A, B}`` and a record citing ``{A}`` hash to
different values, so ``A`` can still land in both splits. That is a real defect in
the first pilot output, where ``hadoop_..._000005`` hashed to test and
``hadoop_..._000003`` to dev while the hard question citing both was assigned test.

So the split is assigned per *event set*, not per group or per record. Co-citation
is an edge, connected components are the event sets, and the component's split is
hashed from its lowest-sorted group id. Two consequences that matter:

* Every record citing any group of a component gets that component's split, so a
  component is never split across dev and test.
* The assignment is a pure function of the record set, so ``validate.py`` recomputes
  it from the written file and reports any record whose stored split disagrees.

A record citing one group is a single-node component whose id is that group id, so
this reduces exactly to the previous per-group hash for the deterministic tiers —
the official 20-question output's splits are unchanged by it.
"""

import hashlib
from typing import Any, Iterable

TEST_FRACTION = 0.20


def group_ids_of(record: dict[str, Any]) -> frozenset[str]:
    """Returns the distinct evidence group ids a record cites.

    Args:
        record: A question record.

    Returns:
        The record's group ids; empty when it cites nothing.
    """
    refs = record.get("evidence", {}).get("refs", [])
    return frozenset(ref["group_id"] for ref in refs if ref.get("group_id"))


def split_for_component(component_id: str, test_fraction: float = TEST_FRACTION) -> str:
    """Assigns one event set to the dev or test split, deterministically.

    Derived purely from a hash of the component id rather than drawn at random, so
    the same event set maps to the same split across repeated runs and across
    machines. The partition is a property of the corpus, not of the run that
    happened to build it.

    Args:
        component_id: The event set's identifier, its lowest-sorted group id.
        test_fraction: Fraction of the hash space assigned to test.

    Returns:
        ``"test"`` or ``"dev"``.
    """
    digest = hashlib.sha256(component_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_fraction else "dev"


def _components(group_sets: Iterable[frozenset[str]]) -> dict[str, str]:
    """Groups co-cited evidence groups into connected components.

    Union-find over the co-citation graph, with each component named by its
    lowest-sorted member so the name does not depend on record order.

    Args:
        group_sets: One group-id set per record.

    Returns:
        Mapping from group id to its component id.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if right_root < left_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    for group_set in group_sets:
        members = sorted(group_set)
        for member in members:
            find(member)
        for member in members[1:]:
            union(members[0], member)

    roots: dict[str, str] = {}
    for group_id in parent:
        roots.setdefault(find(group_id), group_id)
        if group_id < roots[find(group_id)]:
            roots[find(group_id)] = group_id
    return {group_id: roots[find(group_id)] for group_id in parent}


def expected_splits(
    records: list[dict[str, Any]], test_fraction: float = TEST_FRACTION
) -> dict[int, str]:
    """Computes the split every record should carry.

    Args:
        records: The full record set. Splits are only well-defined over all of
            them at once, since a record's component may be linked through
            another record's citations.
        test_fraction: Fraction of the hash space assigned to test.

    Returns:
        Mapping from record index to its split. A record citing no evidence has
        no component and is absent from the mapping.
    """
    group_sets = [group_ids_of(record) for record in records]
    component_of = _components(group_set for group_set in group_sets if group_set)

    splits: dict[int, str] = {}
    for index, group_set in enumerate(group_sets):
        if not group_set:
            continue
        component_id = min(component_of[group_id] for group_id in group_set)
        splits[index] = split_for_component(component_id, test_fraction)
    return splits


def resolve_splits(
    records: list[dict[str, Any]], test_fraction: float = TEST_FRACTION
) -> list[dict[str, Any]]:
    """Stamps every record with its event set's split, in place.

    Called once by ``generate.py`` after all tiers have run. The tiers do not set
    ``split`` themselves: a tier only sees its own records and therefore cannot
    know which components its groups belong to.

    Args:
        records: The full record set.
        test_fraction: Fraction of the hash space assigned to test.

    Returns:
        The same list, with ``split`` set on every record that cites evidence.
    """
    for index, split in expected_splits(records, test_fraction).items():
        records[index]["split"] = split
    return records
