"""Player character example with directional walking animation.

Expects a ``character.png`` spritesheet in the same directory, laid out as a
grid of ``SPRITE_COLS`` columns × ``SPRITE_ROWS`` rows where each row is one
walk direction (top → bottom):

  Row 0 – walk down   (facing the camera)
  Row 1 – walk left
  Row 2 – walk right
  Row 3 – walk up

Controls
--------
  WASD or arrow keys – move the character
  ESC                – close the window
"""

import threading
import time
from pathlib import Path

import neural_ui as nui

# ---------------------------------------------------------------------------
# Spritesheet layout — adjust to match your character.png
# ---------------------------------------------------------------------------
SPRITE_COLS = 4   # animation frames per direction
SPRITE_ROWS = 4   # number of walk directions

# ---------------------------------------------------------------------------
# Animation & movement tuning
# ---------------------------------------------------------------------------
ANIM_FPS = 8.0          # frames per second while walking
MOVE_SPEED = 0.6        # world units per second
LOGIC_SLEEP = 0.016     # ~60 Hz logic tick

# Direction row indices
DIR_DOWN = 0
DIR_LEFT = 1
DIR_RIGHT = 2
DIR_UP = 3

THREAD_SHUTDOWN_TIMEOUT = 1.0

# World-coordinate bounds that the player cannot leave
WORLD_MIN_X = -1.0
WORLD_MAX_X = 1.0
WORLD_MIN_Y = -1.0
WORLD_MAX_Y = 1.0


def _window_should_close(renderer: nui.InstancedModernGLRenderer) -> bool:
    if renderer.window is None or nui.glfw is None:
        return False
    return bool(nui.glfw.window_should_close(renderer.window))


def main() -> None:
    world = nui.NeuralWorld(use_cupy=False)
    spritesheet_path = Path(__file__).resolve().with_name("character.png")
    if not spritesheet_path.is_file():
        raise FileNotFoundError(f"Missing spritesheet image: {spritesheet_path}")

    player = nui.SpriteObject(
        x=0.0,
        y=0.0,
        width=0.15,
        height=0.2,
        z=1.0,
        spritesheet=str(spritesheet_path),
        grid=(SPRITE_COLS, SPRITE_ROWS),
        fps=0.0,
    )
    world.register(player)
    player_index = player.tensor_index
    if player_index is None:
        raise RuntimeError("Player registration failed")

    renderer = nui.InstancedModernGLRenderer(
        world,
        title="Neural UI – Player Character",
        width=1280,
        height=720,
        background_color=0x1A2A1A,
        vsync=True,
    )

    # Open the window so we can set up textures and key callback before rendering.
    renderer.create_window()
    renderer._sync_sprite_costumes()

    # Start in idle-down pose (first frame of the down row, fps=0 → no auto-advance).
    current_direction = DIR_DOWN
    is_moving = False
    if not player._texture_layers:
        raise RuntimeError("Player spritesheet failed to load into texture layers")

    def row_layers(direction: int):
        """Return the layer-ID slice for the given direction row."""
        start = direction * SPRITE_COLS
        return player._texture_layers[start : start + SPRITE_COLS]

    world.configure_sprite_animation(player_index, row_layers(DIR_DOWN), 0.0)

    # --- Keyboard state (updated from GLFW callback on the main thread) ---
    keys_pressed: set[int] = set()
    keys_lock = threading.Lock()

    def key_callback(window, key, scancode, action, mods):
        with keys_lock:
            if action == nui.glfw.PRESS:
                keys_pressed.add(key)
            elif action == nui.glfw.RELEASE:
                keys_pressed.discard(key)
        if key == nui.glfw.KEY_ESCAPE and action == nui.glfw.PRESS:
            nui.glfw.set_window_should_close(window, True)

    nui.glfw.set_key_callback(renderer.window, key_callback)

    # --- Logic thread ---
    stop_event = threading.Event()

    def logic_loop() -> None:
        nonlocal current_direction, is_moving
        # Track world position locally (no get_local_position API).
        px, py = 0.0, 0.0
        last = time.perf_counter()

        while not stop_event.is_set():
            if _window_should_close(renderer):
                return

            now = time.perf_counter()
            dt = now - last
            last = now

            with keys_lock:
                keys = frozenset(keys_pressed)

            dx = dy = 0.0
            new_direction = current_direction
            new_moving = False

            # Direction priority: up > down > left > right.  Each axis is
            # accumulated independently so diagonal movement is supported.
            if nui.glfw.KEY_W in keys or nui.glfw.KEY_UP in keys:
                dy += MOVE_SPEED * dt
                new_direction = DIR_UP
                new_moving = True
            elif nui.glfw.KEY_S in keys or nui.glfw.KEY_DOWN in keys:
                dy -= MOVE_SPEED * dt
                new_direction = DIR_DOWN
                new_moving = True

            if nui.glfw.KEY_A in keys or nui.glfw.KEY_LEFT in keys:
                dx -= MOVE_SPEED * dt
                if not new_moving:
                    new_direction = DIR_LEFT
                new_moving = True
            elif nui.glfw.KEY_D in keys or nui.glfw.KEY_RIGHT in keys:
                dx += MOVE_SPEED * dt
                if not new_moving:
                    new_direction = DIR_RIGHT
                new_moving = True

            # Reconfigure animation only when direction or movement state changes.
            if new_direction != current_direction or new_moving != is_moving:
                current_direction = new_direction
                is_moving = new_moving
                fps = ANIM_FPS if is_moving else 0.0
                world.configure_sprite_animation(
                    player_index, row_layers(current_direction), fps
                )

            if is_moving:
                px = max(WORLD_MIN_X, min(WORLD_MAX_X, px + dx))
                py = max(WORLD_MIN_Y, min(WORLD_MAX_Y, py + dy))
                world.set_local_position(player_index, px, py)
                world.update(dt)

            time.sleep(LOGIC_SLEEP)

    logic_thread = threading.Thread(target=logic_loop, daemon=True)
    logic_thread.start()

    try:
        renderer.run()
    finally:
        stop_event.set()
        logic_thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT)


if __name__ == "__main__":
    main()
