import numpy as np
import unittest

from app.remover import (
    GreenScreenConfig,
    GreenScreenRemover,
    MaskConfig,
    pack_rgb_and_alpha,
    probability_to_alpha,
)


class RemoverTests(unittest.TestCase):
    def test_pack_rgb_and_alpha_places_planes_side_by_side(self) -> None:
        rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        alpha = np.array([[0, 255]], dtype=np.uint8)

        packed = pack_rgb_and_alpha(rgb, alpha)

        self.assertEqual(packed.shape, (1, 4, 3))
        np.testing.assert_array_equal(packed[:, :2], rgb)
        np.testing.assert_array_equal(
            packed[:, 2:],
            np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8),
        )

    def test_pack_rejects_mismatched_mask(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        alpha = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            pack_rgb_and_alpha(rgb, alpha)

    def test_probability_to_alpha_keeps_soft_edges(self) -> None:
        probability = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        config = MaskConfig(
            threshold=0.5, feather=0.4, temporal_smoothing=0, blur_size=1
        )

        alpha, state = probability_to_alpha(probability, config)

        self.assertEqual(alpha[0, 0], 0)
        self.assertTrue(100 <= alpha[0, 1] <= 155)
        self.assertEqual(alpha[0, 2], 255)
        self.assertEqual(state.dtype, np.float32)

    def test_probability_to_alpha_can_shrink_background_halo(self) -> None:
        probability = np.zeros((9, 9), dtype=np.float32)
        probability[2:7, 2:7] = 1.0
        config = MaskConfig(
            threshold=0.5,
            feather=0.2,
            temporal_smoothing=0,
            blur_size=1,
            edge_shrink=3,
        )

        alpha, _ = probability_to_alpha(probability, config)

        self.assertEqual(alpha[2, 2], 0)
        self.assertEqual(alpha[4, 4], 255)

    def test_green_screen_remover_keys_green_and_keeps_subject(self) -> None:
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        rgb[:, :6] = (20, 220, 25)
        rgb[:, 6:] = (210, 40, 35)
        remover = GreenScreenRemover(
            GreenScreenConfig(temporal_smoothing=0, blur_size=1)
        )

        alpha = remover.mask(rgb)

        self.assertLess(int(alpha[:, :6].max()), 10)
        self.assertGreater(int(alpha[:, 6:].min()), 245)

    def test_green_screen_remover_suppresses_green_edge_spill(self) -> None:
        rgb = np.array([[[60, 115, 65], [180, 70, 60]]], dtype=np.uint8)
        alpha = np.array([[180, 255]], dtype=np.uint8)
        remover = GreenScreenRemover(
            GreenScreenConfig(temporal_smoothing=0, blur_size=1)
        )

        refined = remover.refine_rgb(rgb, alpha)

        self.assertLess(int(refined[0, 0, 1]), int(rgb[0, 0, 1]))
        np.testing.assert_array_equal(refined[0, 1], rgb[0, 1])

    def test_green_screen_remover_does_not_leave_motion_trails(self) -> None:
        remover = GreenScreenRemover(
            GreenScreenConfig(temporal_smoothing=0.75, blur_size=1)
        )
        subject = np.full((8, 8, 3), (210, 40, 35), dtype=np.uint8)
        green = np.full((8, 8, 3), (20, 220, 25), dtype=np.uint8)

        foreground_alpha = remover.mask(subject)
        cleared_alpha = remover.mask(green)

        self.assertGreater(int(foreground_alpha.min()), 245)
        self.assertLess(int(cleared_alpha.max()), 10)

    def test_green_screen_remover_keeps_fast_new_motion_opaque(self) -> None:
        remover = GreenScreenRemover(GreenScreenConfig(blur_size=1))
        green = np.full((8, 8, 3), (20, 220, 25), dtype=np.uint8)
        subject = np.full((8, 8, 3), (225, 175, 160), dtype=np.uint8)

        remover.mask(green)
        moving_hand_alpha = remover.mask(subject)

        self.assertGreater(int(moving_hand_alpha.min()), 245)


if __name__ == "__main__":
    unittest.main()

