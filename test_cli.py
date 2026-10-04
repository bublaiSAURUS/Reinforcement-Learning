"""The command's argument handling is independent of scientific dependencies."""

import unittest

from snake_rl.cli import build_parser


class CommandTests(unittest.TestCase):
    def test_default_checkpoint_and_unlimited_training(self):
        args = build_parser().parse_args([])
        self.assertIsNone(args.episodes)
        self.assertEqual(args.checkpoint, 'model/model.pth')
        self.assertFalse(args.headless)

    def test_finite_headless_run(self):
        args = build_parser().parse_args(
            ['--episodes', '20', '--headless', '--seed', '42']
        )
        self.assertEqual(args.episodes, 20)
        self.assertEqual(args.seed, 42)
        self.assertTrue(args.headless)
