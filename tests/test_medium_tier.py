"""Tests for the medium tier's window and structured-summary parsing.

No model server: these exercise ``_evidence_window`` and
``_parse_structured_summary`` directly against synthetic input.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.generators.medium_tier import _evidence_window, _parse_structured_summary


class EvidenceWindowTest(unittest.TestCase):
    def test_symmetric_window_around_the_anchor(self):
        lines = ["line %d" % i for i in range(20)]
        window = _evidence_window(lines, anchor_idx=10, context_before=3, context_after=2)
        self.assertEqual(window, [7, 8, 9, 10, 11, 12])

    def test_clips_at_the_start_of_the_file(self):
        lines = ["line %d" % i for i in range(20)]
        window = _evidence_window(lines, anchor_idx=1, context_before=5, context_after=2)
        self.assertEqual(window, [0, 1, 2, 3])

    def test_clips_at_the_end_of_the_file(self):
        lines = ["line %d" % i for i in range(10)]
        window = _evidence_window(lines, anchor_idx=8, context_before=2, context_after=5)
        self.assertEqual(window, [6, 7, 8, 9])

    def test_single_line_file(self):
        window = _evidence_window(["only line"], anchor_idx=0, context_before=5, context_after=5)
        self.assertEqual(window, [0])


class ParseStructuredSummaryTest(unittest.TestCase):
    def test_parses_all_four_fields(self):
        text = (
            "EVENT_TYPE: cache parity error\n"
            "ENTITY: R30-M0-N9-C:J16-U01\n"
            "OBSERVED_EVENTS: a fatal interrupt is logged repeatedly\n"
            "OUTCOME: none stated\n"
        )
        parsed = _parse_structured_summary(text)
        self.assertEqual(parsed["event_type"], "cache parity error")
        self.assertEqual(parsed["entity"], "R30-M0-N9-C:J16-U01")
        self.assertEqual(
            parsed["observed_events"], "a fatal interrupt is logged repeatedly"
        )
        self.assertEqual(parsed["outcome"], "none stated")

    def test_is_case_and_order_insensitive(self):
        text = "outcome: recovered\nevent_type: retry\nentity: node-1\nobserved_events: retried twice\n"
        parsed = _parse_structured_summary(text)
        self.assertEqual(parsed["event_type"], "retry")
        self.assertEqual(parsed["outcome"], "recovered")

    def test_missing_field_falls_back_to_not_stated(self):
        text = "EVENT_TYPE: retry\nENTITY: node-1\nOBSERVED_EVENTS: retried twice\n"
        parsed = _parse_structured_summary(text)
        self.assertEqual(parsed["outcome"], "not stated")

    def test_completely_unparseable_text_falls_back_on_every_field(self):
        parsed = _parse_structured_summary("The model refused to follow the format.")
        self.assertEqual(
            parsed,
            {
                "event_type": "not stated",
                "entity": "not stated",
                "observed_events": "not stated",
                "outcome": "not stated",
            },
        )


if __name__ == "__main__":
    unittest.main()
