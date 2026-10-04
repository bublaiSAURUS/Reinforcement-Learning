"""Regression checks for the original movement and reward rules."""

import unittest

from snake_rl.game import BLOCK_SIZE, Direction, Point, SnakeGameAI


class GameTests(unittest.TestCase):
    def setUp(self):
        self.game = SnakeGameAI(render=False)
        self.game.food = Point(0, 0)

    def tearDown(self):
        self.game.close()

    def test_straight_move_preserves_length(self):
        head = self.game.head
        result = self.game.play_step([1, 0, 0])
        self.assertEqual(self.game.head, Point(head.x + BLOCK_SIZE, head.y))
        self.assertEqual(result, (0, False, 0))
        self.assertEqual(len(self.game.snake), 3)

    def test_right_turn(self):
        self.game.play_step([0, 1, 0])
        self.assertEqual(self.game.direction, Direction.DOWN)

    def test_food_reward_and_growth(self):
        self.game.food = Point(self.game.head.x + BLOCK_SIZE, self.game.head.y)
        self.assertEqual(self.game.play_step([1, 0, 0]), (10, False, 1))
        self.assertEqual(len(self.game.snake), 4)

    def test_wall_collision(self):
        self.game.head = Point(self.game.w - BLOCK_SIZE, self.game.head.y)
        self.game.snake[0] = self.game.head
        self.assertEqual(self.game.play_step([1, 0, 0]), (-10, True, 0))

    def test_timeout(self):
        self.game.frame_iteration = 401
        self.assertEqual(self.game.play_step([1, 0, 0]), (-10, True, 0))


if __name__ == '__main__':
    unittest.main()
