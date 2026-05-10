import math
import threading
import time

import neural_ui as nui


def _set_object_xy(world: nui.NeuralWorld, index: int, x: float, y: float) -> None:
    if world.backend == "python":
        world.world_tensor[index][nui.NeuralWorld._ROW_X] = float(x)
        world.world_tensor[index][nui.NeuralWorld._ROW_Y] = float(y)
    else:
        world.world_tensor[index, nui.NeuralWorld._ROW_X] = float(x)
        world.world_tensor[index, nui.NeuralWorld._ROW_Y] = float(y)
    world._global_dirty = True


def main() -> None:
    world = nui.NeuralWorld(use_cupy=True, initial_capacity=5_000, growth_chunk=2_000)

    ship = nui.ObjectGroup(x=0.0, y=0.0, width=0.08, height=0.08, z=2.0)
    drone_count = 200
    for i in range(drone_count):
        angle = (i / drone_count) * (2.0 * math.pi)
        ship.add(
            nui.Object(
                x=math.sin(angle) * 0.2,
                y=math.cos(angle) * 0.2,
                width=0.01,
                height=0.01,
                z=1.0,
            )
        )

    world.register(ship)
    ship_index = ship.tensor_index
    if ship_index is None:
        raise RuntimeError("Ship registration failed")

    renderer = nui.InstancedModernGLRenderer(
        world,
        title="Neural UI - Instanced Renderer Example",
        width=1280,
        height=720,
        background_color=0x0D0D12,
        object_color=0x5FA6FF,
        hover_color=0x5CF2A0,
        random_object_colors=True,
        vsync=True,
    )

    def logic_loop() -> None:
        t = 0.0
        while True:
            if renderer.window is not None and nui.glfw is not None and nui.glfw.window_should_close(renderer.window):
                return
            _set_object_xy(world, ship_index, math.sin(t) * 0.6, math.cos(t * 0.8) * 0.3)
            t += 0.02
            time.sleep(0.01)

    threading.Thread(target=logic_loop, daemon=True).start()
    renderer.run()


if __name__ == "__main__":
    main()
