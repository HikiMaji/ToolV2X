"""Small contract checks for the read-only T9 admission audit."""
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from audit_t9_admission import (admit_in_order, receipt_stage_metadata,
                                receipt_stages, target_round_robin)


class T9AdmissionAuditTests(unittest.TestCase):
    def test_round_robin_keeps_distinct_context_units_and_cycles_targets(self):
        units = [
            {'id': '20-provider', 'track': 20},
            {'id': '20-subset', 'track': 20},
            {'id': '20-history', 'track': 20},
            {'id': '6-provider', 'track': 6},
            {'id': '6-subset', 'track': 6},
            {'id': '3-provider', 'track': 3},
        ]

        ordered = target_round_robin(units, target_key=lambda unit: unit['track'])

        self.assertEqual([unit['id'] for unit in ordered], [
            '20-provider', '6-provider', '3-provider',
            '20-subset', '6-subset', '20-history',
        ])

    def test_admission_skips_an_oversize_unit_and_keeps_exact_attempt_totals(self):
        units = [{'id': 'a', 'tokens': 40}, {'id': 'b', 'tokens': 80}, {'id': 'c', 'tokens': 30}]

        selected, rows = admit_in_order(
            units,
            input_tokens_for=lambda values: 100 + sum(value['tokens'] for value in values),
            max_input_tokens=170,
        )

        self.assertEqual([unit['id'] for unit in selected], ['a', 'c'])
        self.assertEqual(
            [(row['attempt_input_tokens'], row['admitted'], row['cumulative_input_tokens']) for row in rows],
            [(140, True, 140), (220, False, 140), (170, True, 170)],
        )

    def test_two_one_shot_receipts_share_the_saved_plan_stage(self):
        episode = {
            'plans': [{'stage': 0, 'ledger_snapshot': 0}, {'stage': 1, 'ledger_snapshot': 1}],
            'ledger_snapshots': [
                {'receipts': []},
                {'receipts': [{'receipt_id': 'p'}, {'receipt_id': 'f'}]},
            ],
        }

        self.assertEqual(receipt_stages(episode), {'p': 1, 'f': 1})

    def test_request_stage_and_first_visible_stage_follow_outer_events(self):
        alternating = {
            'plans': [
                {'stage': 0, 'request_id': None, 'ledger_snapshot': 0},
                {'stage': 1, 'request_id': 'q0', 'ledger_snapshot': 1},
                {'stage': 2, 'request_id': 'q1', 'ledger_snapshot': 2},
            ],
            'ledger_snapshots': [
                {'receipts': []},
                {'receipts': [{'receipt_id': 'p'}]},
                {'receipts': [{'receipt_id': 'p'}, {'receipt_id': 'f'}]},
            ],
            'events': [
                {'kind': 'request_started', 'request_id': 'q0', 'stage': 0},
                {'kind': 'request_started', 'request_id': 'q1', 'stage': 1},
            ],
            'requests': [{'request_id': 'q0'}, {'request_id': 'q1'}],
        }
        primitives = [
            {'request': {'request_id': 'q0'}, 'response': {'receipt_id': 'p'}},
            {'request': {'request_id': 'q1'}, 'response': {'receipt_id': 'f'}},
        ]
        alternating_meta = receipt_stage_metadata(
            alternating, {'primitive_records': primitives, 'bundle_records': []})
        self.assertEqual(alternating_meta, {
            'p': {'request_stage': 0, 'first_visible_plan_stage': 1, 'outer_request_id': 'q0'},
            'f': {'request_stage': 1, 'first_visible_plan_stage': 2, 'outer_request_id': 'q1'},
        })

        one_shot = {
            'plans': [
                {'stage': 0, 'request_id': None, 'ledger_snapshot': 0},
                {'stage': 1, 'request_id': 'bundle0', 'ledger_snapshot': 1},
            ],
            'ledger_snapshots': [
                {'receipts': []},
                {'receipts': [{'receipt_id': 'p'}, {'receipt_id': 'f'}]},
            ],
            'events': [{'kind': 'request_started', 'request_id': 'bundle0', 'stage': 0}],
            'requests': [{'request_id': 'bundle0'}],
        }
        one_shot_meta = receipt_stage_metadata(one_shot, {
            'primitive_records': primitives,
            'bundle_records': [{'request': {'request_id': 'bundle0'}, 'primitive_records': primitives}],
        })
        self.assertEqual(one_shot_meta, {
            'p': {'request_stage': 0, 'first_visible_plan_stage': 1, 'outer_request_id': 'bundle0'},
            'f': {'request_stage': 0, 'first_visible_plan_stage': 1, 'outer_request_id': 'bundle0'},
        })


if __name__ == '__main__':
    unittest.main()
