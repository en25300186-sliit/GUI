import math
import threading
import time

import neural_ui as nui

# Horizontal ship movement amplitude (world units).
SHIP_X_AMPLITUDE = 0.6
# Vertical ship movement amplitude (world units).
SHIP_Y_AMPLITUDE = 0.3
# Vertical oscillation frequency multiplier.
SHIP_Y_FREQUENCY = 0.8
# Animation time increment per logic tick.
TIME_STEP = 0.02
# Delay between logic ticks (seconds).
SLEEP_SECONDS = 0.01
# Maximum time to wait for logic thread shutdown (seconds).
THREAD_SHUTDOWN_TIMEOUT = 1.0


def _window_should_close(renderer: nui.InstancedModernGLRenderer) -> bool:
    if renderer.window is None or nui.glfw is None:
        return False
    return bool(nui.glfw.window_should_close(renderer.window))


def main() -> None:
    world = nui.NeuralWorld(use_cupy=nui.cp is not None, initial_capacity=5000, growth_chunk=2000)

    ship = nui.ObjectGroup(x=0.0, y=0.0, width=0.08, height=0.08, z=2.0)
    child_object_count = 200
    for i in range(child_object_count):
        angle = (i / child_object_count) * (2.0 * math.pi)
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

    stop_event = threading.Event()

    def logic_loop() -> None:
        t = 0.0
        while not stop_event.is_set():
            if _window_should_close(renderer):
                return
            world.set_local_position(
                ship_index, math.sin(t) * SHIP_X_AMPLITUDE, math.cos(t * SHIP_Y_FREQUENCY) * SHIP_Y_AMPLITUDE
            )
            t += TIME_STEP
            time.sleep(SLEEP_SECONDS)

    logic_thread = threading.Thread(target=logic_loop)
    logic_thread.start()
    try:
        renderer.run()
    finally:
        stop_event.set()
        logic_thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT)


if __name__ == "__main__":
    main()
