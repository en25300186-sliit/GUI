import unittest

from neural_ui import NeuralWorld, Object, ObjectGroup, ObjectState


class NeuralWorldTests(unittest.TestCase):
    def test_screen_to_world_respects_center_and_aspect(self):
        x_world, y_world = NeuralWorld.screen_to_world(100, 50, 200, 100)
        self.assertAlmostEqual(x_world, 0.0)
        self.assertAlmostEqual(y_world, 0.0)

        left_x, _ = NeuralWorld.screen_to_world(0, 50, 200, 100)
        right_x, _ = NeuralWorld.screen_to_world(200, 50, 200, 100)
        self.assertAlmostEqual(left_x, -2.0)
        self.assertAlmostEqual(right_x, 2.0)

    def test_hit_mask_filters_active_objects_only(self):
        world = NeuralWorld(use_cupy=False)
        active = Object(x=0, y=0, width=2, height=2, state=ObjectState.ACTIVE, on_hover=lambda _: None)
        hidden = Object(x=0, y=0, width=2, height=2, state=ObjectState.HIDDEN, on_hover=lambda _: None)
        offscreen = Object(x=0, y=0, width=2, height=2, state=ObjectState.OUTOFSCREEN, on_hover=lambda _: None)
        world.register(active)
        world.register(hidden)
        world.register(offscreen)

        mask = world.hit_mask(0, 0, event="hover")
        self.assertEqual(list(mask), [True, False, False])

    def test_topmost_winner_uses_z(self):
        world = NeuralWorld(use_cupy=False)
        low = Object(x=0, y=0, width=2, height=2, z=1, on_click=lambda _: None)
        high = Object(x=0, y=0, width=2, height=2, z=5, on_click=lambda _: None)
        world.register(low)
        world.register(high)

        winner = world.topmost_hit(0, 0, event="click")
        self.assertEqual(winner, high.tensor_index)

    def test_dispatch_bridges_to_object_callbacks(self):
        world = NeuralWorld(use_cupy=False)
        calls = []

        clickable = Object(
            x=0,
            y=0,
            width=2,
            height=2,
            on_hover=lambda obj: calls.append(("hover", obj.tensor_index)),
            on_click=lambda obj: calls.append(("click", obj.tensor_index)),
        )
        world.register(clickable)

        hovered = world.win_proc("mouse_move", 50, 50, 100, 100)
        clicked = world.win_proc("mouse_click", 50, 50, 100, 100)

        self.assertIs(hovered, clickable)
        self.assertIs(clicked, clickable)
        self.assertEqual(calls, [("hover", clickable.tensor_index), ("click", clickable.tensor_index)])

    def test_object_group_supports_nested_objects(self):
        world = NeuralWorld(use_cupy=False)
        child = Object(x=0, y=0, width=1, height=1, on_hover=lambda _: None)
        nested = ObjectGroup(x=0, y=0, width=1, height=1, subitems=[child])

        # create using explicit add() to verify ObjectGroup constraints
        group = ObjectGroup(x=0, y=0, width=1, height=1)
        group.add(nested)

        world.register(group)
        self.assertIsNotNone(group.tensor_index)
        self.assertIsNotNone(nested.tensor_index)
        self.assertIsNotNone(child.tensor_index)

    def test_preallocated_tensor_grows_in_chunks(self):
        world = NeuralWorld(use_cupy=False, initial_capacity=2, growth_chunk=2)
        world.register(Object(x=0, y=0, width=1, height=1, on_hover=lambda _: None))
        world.register(Object(x=1, y=1, width=1, height=1, on_hover=lambda _: None))
        world.register(Object(x=2, y=2, width=1, height=1, on_hover=lambda _: None))

        self.assertEqual(world.size, 3)
        if world.backend == "python":
            self.assertEqual(world.capacity, 3)
            self.assertEqual(len(world.world_tensor), 3)
        else:
            self.assertEqual(world.capacity, 4)
            self.assertEqual(int(world.world_tensor.shape[0]), 4)

    def test_hierarchy_updates_global_transform(self):
        world = NeuralWorld(use_cupy=False)
        child = Object(x=0.5, y=-0.5, width=1, height=1, z=2, on_hover=lambda _: None)
        parent = ObjectGroup(child, x=1.0, y=2.0, width=1, height=1, z=3, on_hover=lambda _: None)
        world.register(parent)

        row = world.global_row(child.tensor_index)
        self.assertAlmostEqual(row[0], 1.5)
        self.assertAlmostEqual(row[1], 1.5)
        self.assertAlmostEqual(row[4], 5.0)


if __name__ == "__main__":
    unittest.main()
