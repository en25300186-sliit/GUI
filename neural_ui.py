from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised when CuPy is installed
    import cupy as cp
except ImportError:  # pragma: no cover - fallback for environments without CuPy
    cp = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - fallback for environments without NumPy
    np = None

try:  # pragma: no cover - optional dependency for shader rendering
    import glfw
except ImportError:  # pragma: no cover - fallback for environments without GLFW
    glfw = None

try:  # pragma: no cover - optional dependency for shader rendering
    import moderngl
except ImportError:  # pragma: no cover - fallback for environments without ModernGL
    moderngl = None


class ObjectState(IntEnum):
    ACTIVE = 1
    HIDDEN = 0
    OUTOFSCREEN = -1


@dataclass
class Object:
    x: float
    y: float
    width: float
    height: float
    z: float = 0.0
    state: ObjectState = ObjectState.ACTIVE
    on_hover: Optional[Callable[["Object"], None]] = None
    on_click: Optional[Callable[["Object"], None]] = None
    tensor_index: Optional[int] = None


class ObjectGroup(Object):
    def __init__(self, *items: Object, subitems: Optional[Sequence[Object]] = None, **kwargs):
        super().__init__(**kwargs)
        self.subitems: List[Object] = []
        for subitem in items:
            self.add(subitem)
        for subitem in subitems or []:
            self.add(subitem)

    def add(self, subitem: Object) -> None:
        if not isinstance(subitem, Object):
            raise TypeError("ObjectGroup accepts Object or ObjectGroup instances")
        self.subitems.append(subitem)


class NeuralWorld:
    _ROW_X = 0
    _ROW_Y = 1
    _ROW_HALF_W = 2
    _ROW_HALF_H = 3
    _ROW_Z = 4
    _ROW_STATE = 5

    def __init__(self, use_cupy: bool = True, initial_capacity: int = 10_000, growth_chunk: int = 10_000) -> None:
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        if growth_chunk <= 0:
            raise ValueError("growth_chunk must be positive")
        self._growth_chunk = int(growth_chunk)
        self._capacity = int(initial_capacity)
        self._size = 0
        self._global_dirty = False
        if use_cupy and cp is not None:
            self.xp = cp
            self.backend = "cupy"
            self.world_tensor = self.xp.zeros((self._capacity, 6), dtype=self.xp.float32)
            self._global_tensor = self.xp.zeros((self._capacity, 6), dtype=self.xp.float32)
            self._parent_index = self.xp.full((self._capacity,), -1, dtype=self.xp.int32)
        elif np is not None:
            self.xp = np
            self.backend = "numpy"
            self.world_tensor = self.xp.zeros((self._capacity, 6), dtype=self.xp.float32)
            self._global_tensor = self.xp.zeros((self._capacity, 6), dtype=self.xp.float32)
            self._parent_index = self.xp.full((self._capacity,), -1, dtype=self.xp.int32)
        else:
            self.xp = None
            self.backend = "python"
            self.world_tensor: List[List[float]] = []
            self._python_global_rows: List[List[float]] = []
            self._python_parent_index: List[int] = []
        self._objects: List[Object] = []
        self._index_to_object: Dict[int, Object] = {}
        self._last_hover_index: Optional[int] = None

    @property
    def size(self) -> int:
        return self._size if self.backend != "python" else len(self.world_tensor)

    @property
    def capacity(self) -> int:
        return self._capacity if self.backend != "python" else len(self.world_tensor)

    def register(self, obj: Object, _parent_index: int = -1) -> int:
        if isinstance(obj, ObjectGroup):
            group_index = self._register_single(obj, parent_index=_parent_index)
            for subitem in obj.subitems:
                self.register(subitem, _parent_index=group_index)
            return group_index
        return self._register_single(obj, parent_index=_parent_index)

    def _ensure_capacity(self, minimum_capacity: int) -> None:
        if self.backend == "python":
            return
        if minimum_capacity <= self._capacity:
            return
        new_capacity = self._capacity
        while new_capacity < minimum_capacity:
            new_capacity += self._growth_chunk
        expand_by = new_capacity - self._capacity
        zeros = self.xp.zeros((expand_by, 6), dtype=self.xp.float32)
        self.world_tensor = self.xp.concatenate((self.world_tensor, zeros), axis=0)
        self._global_tensor = self.xp.concatenate((self._global_tensor, zeros), axis=0)
        parent_padding = self.xp.full((expand_by,), -1, dtype=self.xp.int32)
        self._parent_index = self.xp.concatenate((self._parent_index, parent_padding), axis=0)
        self._capacity = new_capacity

    def _register_single(self, obj: Object, parent_index: int = -1) -> int:
        row = [
            obj.x,
            obj.y,
            obj.width / 2.0,
            obj.height / 2.0,
            obj.z,
            float(obj.state),
        ]
        if self.backend == "python":
            self.world_tensor.append(row)
            self._python_parent_index.append(parent_index)
            self._python_global_rows.append(row[:])
            index = len(self.world_tensor) - 1
        else:
            index = self._size
            self._ensure_capacity(index + 1)
            row_tensor = self.xp.asarray(row, dtype=self.xp.float32)
            self.world_tensor[index, :] = row_tensor
            self._parent_index[index] = int(parent_index)
            self._global_tensor[index, :] = row_tensor
            self._size += 1
        obj.tensor_index = index
        self._objects.append(obj)
        self._index_to_object[index] = obj
        self._global_dirty = True
        return index

    def _sync_global_transforms(self) -> None:
        if not self._global_dirty:
            return

        if self.backend == "python":
            object_count = len(self.world_tensor)
            if object_count == 0:
                self._global_dirty = False
                return

            for idx, row in enumerate(self.world_tensor):
                self._python_global_rows[idx] = row[:]
                parent_idx = self._python_parent_index[idx]
                if parent_idx < -1 or parent_idx >= object_count:
                    raise ValueError(f"Parent index {parent_idx} for object {idx} is out of bounds")

            resolved = [parent_idx == -1 for parent_idx in self._python_parent_index]
            remaining = object_count - sum(resolved)
            while remaining > 0:
                progressed = False
                for idx in range(object_count):
                    if resolved[idx]:
                        continue
                    parent_idx = self._python_parent_index[idx]
                    if parent_idx >= 0 and resolved[parent_idx]:
                        local_row = self.world_tensor[idx]
                        parent_global = self._python_global_rows[parent_idx]
                        global_row = local_row[:]
                        global_row[self._ROW_X] = local_row[self._ROW_X] + parent_global[self._ROW_X]
                        global_row[self._ROW_Y] = local_row[self._ROW_Y] + parent_global[self._ROW_Y]
                        global_row[self._ROW_Z] = local_row[self._ROW_Z] + parent_global[self._ROW_Z]
                        self._python_global_rows[idx] = global_row
                        resolved[idx] = True
                        remaining -= 1
                        progressed = True
                if not progressed:
                    raise ValueError("Hierarchy cycle detected")
            self._global_dirty = False
            return

        if self._size == 0:
            self._global_dirty = False
            return
        active_local = self.world_tensor[: self._size]
        active_global = self._global_tensor[: self._size]
        active_global[:, :] = active_local
        active_parent = self._parent_index[: self._size]
        invalid_parent = (active_parent < -1) | (active_parent >= self._size)
        if bool(self._to_scalar(invalid_parent.any())):
            invalid_indices = self.xp.where(invalid_parent)[0]
            bad_index = int(self._to_scalar(invalid_indices[0]))
            bad_parent = int(self._to_scalar(active_parent[bad_index]))
            raise ValueError(f"Parent index {bad_parent} for object {bad_index} is out of bounds")

        resolved = active_parent == -1
        remaining = self._size - int(self._to_scalar(resolved.sum()))
        while remaining > 0:
            unresolved_indices = self.xp.where(~resolved)[0]
            unresolved_parents = active_parent[unresolved_indices]
            ready_mask = resolved[unresolved_parents]
            ready_indices = unresolved_indices[ready_mask]
            ready_count = int(self._to_scalar(ready_indices.size))
            if ready_count == 0:
                raise ValueError("Hierarchy cycle detected")
            parent_indices = active_parent[ready_indices]
            active_global[ready_indices, self._ROW_X] = (
                active_local[ready_indices, self._ROW_X] + active_global[parent_indices, self._ROW_X]
            )
            active_global[ready_indices, self._ROW_Y] = (
                active_local[ready_indices, self._ROW_Y] + active_global[parent_indices, self._ROW_Y]
            )
            active_global[ready_indices, self._ROW_Z] = (
                active_local[ready_indices, self._ROW_Z] + active_global[parent_indices, self._ROW_Z]
            )
            resolved[ready_indices] = True
            remaining -= ready_count
        self._global_dirty = False

    def global_row(self, index: int) -> Optional[Sequence[float]]:
        if index < 0 or index >= self.size:
            return None
        self._sync_global_transforms()
        if self.backend == "python":
            return self._python_global_rows[index][:]
        return self._to_scalar(self._global_tensor[index, :]).tolist()

    def iter_render_rows(self) -> Iterable[Tuple[Object, Sequence[float]]]:
        self._sync_global_transforms()
        if self.backend == "python":
            for obj, row in zip(self._objects, self._python_global_rows):
                yield obj, row
            return

        active_rows = self._global_tensor[: self._size]
        if self.backend == "cupy":
            active_rows_cpu = cp.asnumpy(active_rows)
        else:
            active_rows_cpu = active_rows
        for idx, obj in enumerate(self._objects):
            yield obj, active_rows_cpu[idx].tolist()

    @staticmethod
    def screen_to_world(mouse_x_px: float, mouse_y_px: float, width_px: float, height_px: float) -> Sequence[float]:
        if width_px <= 0 or height_px <= 0:
            raise ValueError("Viewport dimensions must be positive")
        aspect = width_px / height_px
        x_world = ((mouse_x_px / width_px) * 2.0 - 1.0) * aspect
        y_world = 1.0 - (mouse_y_px / height_px) * 2.0
        return x_world, y_world

    @staticmethod
    def _has_handler(obj: Object, event: str) -> bool:
        if event == "hover":
            return obj.on_hover is not None
        if event == "click":
            return obj.on_click is not None
        return False

    def _to_scalar(self, value: Any) -> Any:
        if self.backend in {"python", "numpy"}:
            return value
        return self.xp.asnumpy(value)

    def hit_mask(self, x_world: float, y_world: float, *, event: str = "hover"):
        if (self.backend == "python" and not self.world_tensor) or (self.backend != "python" and self._size == 0):
            return [] if self.backend == "python" else self.xp.asarray([], dtype=bool)

        self._sync_global_transforms()

        if self.backend == "python":
            mask = []
            for row, obj in zip(self._python_global_rows, self._objects):
                callback_ok = self._has_handler(obj, event)
                is_hit = (
                    row[self._ROW_STATE] == float(ObjectState.ACTIVE)
                    and abs(row[self._ROW_X] - x_world) <= row[self._ROW_HALF_W]
                    and abs(row[self._ROW_Y] - y_world) <= row[self._ROW_HALF_H]
                    and callback_ok
                )
                mask.append(is_hit)
            return mask

        rows = self._global_tensor[: self._size]
        active_mask = rows[:, self._ROW_STATE] == float(ObjectState.ACTIVE)
        within_x = self.xp.abs(rows[:, self._ROW_X] - x_world) <= rows[:, self._ROW_HALF_W]
        within_y = self.xp.abs(rows[:, self._ROW_Y] - y_world) <= rows[:, self._ROW_HALF_H]
        callback_mask = self.xp.ones(rows.shape[0], dtype=bool)
        for index, obj in self._index_to_object.items():
            if not self._has_handler(obj, event):
                callback_mask[index] = False

        return active_mask & within_x & within_y & callback_mask

    def topmost_hit(self, x_world: float, y_world: float, *, event: str = "hover") -> Optional[int]:
        mask = self.hit_mask(x_world, y_world, event=event)
        if self.backend == "python":
            candidate_indices = [i for i, hit in enumerate(mask) if hit]
            if not candidate_indices:
                return None
            return max(candidate_indices, key=lambda idx: (self._python_global_rows[idx][self._ROW_Z], idx))

        if int(self._to_scalar(mask.sum())) == 0:
            return None

        candidate_indices = self.xp.where(mask)[0]
        candidate_z = self._global_tensor[candidate_indices, self._ROW_Z]
        best_local_idx = int(self._to_scalar(self.xp.argmax(candidate_z)))
        return int(self._to_scalar(candidate_indices[best_local_idx]))

    def dispatch(self, event: str, mouse_x_px: float, mouse_y_px: float, width_px: float, height_px: float) -> Optional[Object]:
        x_world, y_world = self.screen_to_world(mouse_x_px, mouse_y_px, width_px, height_px)
        winner = self.topmost_hit(x_world, y_world, event=event)

        if event == "hover":
            self._last_hover_index = winner

        if winner is None:
            return None

        obj = self._index_to_object[winner]
        if event == "hover" and obj.on_hover is not None:
            obj.on_hover(obj)
        elif event == "click" and obj.on_click is not None:
            obj.on_click(obj)
        return obj

    def win_proc(self, message: str, mouse_x_px: float, mouse_y_px: float, width_px: float, height_px: float) -> Optional[Object]:
        if message == "mouse_move":
            return self.dispatch("hover", mouse_x_px, mouse_y_px, width_px, height_px)
        if message == "mouse_click":
            return self.dispatch("click", mouse_x_px, mouse_y_px, width_px, height_px)
        return None


class ModernGLRenderer:
    def __init__(
        self,
        world: NeuralWorld,
        *,
        title: str = "Neural UI",
        width: int = 1280,
        height: int = 720,
        background_color: int = 0x202020,
        object_color: int = 0x4080FF,
        hover_color: int = 0x40D880,
        vsync: bool = True,
    ) -> None:
        if moderngl is None or glfw is None:
            raise RuntimeError("ModernGLRenderer requires both moderngl and glfw")
        self.world = world
        self.title = title
        self.width = width
        self.height = height
        self.background_color = self._int_to_rgb(background_color)
        self.object_color = self._int_to_rgb(object_color)
        self.hover_color = self._int_to_rgb(hover_color)
        self.vsync = vsync
        self.window = None
        self._ctx = None
        self._program = None
        self._vbo = None
        self._vao = None
        self._left_down = False

    @staticmethod
    def _int_to_rgb(value: int) -> Tuple[float, float, float]:
        return (
            ((value >> 16) & 0xFF) / 255.0,
            ((value >> 8) & 0xFF) / 255.0,
            (value & 0xFF) / 255.0,
        )

    @staticmethod
    def world_to_ndc(
        x_world: float, y_world: float, width_px: int, height_px: int, half_w_world: float, half_h_world: float
    ) -> Tuple[float, float, float, float]:
        aspect = width_px / height_px
        left_ndc = (x_world - half_w_world) / aspect
        right_ndc = (x_world + half_w_world) / aspect
        top_ndc = y_world + half_h_world
        bottom_ndc = y_world - half_h_world
        return left_ndc, top_ndc, right_ndc, bottom_ndc

    def _build_vertices(self) -> bytes:
        vertices = array("f")
        for obj, row in self.world.iter_render_rows():
            if row[NeuralWorld._ROW_STATE] != float(ObjectState.ACTIVE):
                continue
            color = self.hover_color if obj.tensor_index == self.world._last_hover_index else self.object_color
            left, top, right, bottom = self.world_to_ndc(
                row[NeuralWorld._ROW_X],
                row[NeuralWorld._ROW_Y],
                self.width,
                self.height,
                row[NeuralWorld._ROW_HALF_W],
                row[NeuralWorld._ROW_HALF_H],
            )
            r, g, b = color
            vertices.extend((left, top, r, g, b))
            vertices.extend((right, top, r, g, b))
            vertices.extend((right, bottom, r, g, b))
            vertices.extend((left, top, r, g, b))
            vertices.extend((right, bottom, r, g, b))
            vertices.extend((left, bottom, r, g, b))
        return vertices.tobytes()

    def _draw_world(self) -> None:
        self._ctx.clear(*self.background_color, 1.0)
        payload = self._build_vertices()
        if not payload:
            return
        if len(payload) > self._vbo.size:
            self._vbo.orphan(len(payload))
        self._vbo.write(payload)
        self._vao.render(moderngl.TRIANGLES, vertices=len(payload) // (5 * 4))

    def create_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if self.vsync else 0)

        self._ctx = moderngl.create_context()
        self._program = self._ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec3 in_color;
                out vec3 v_color;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(v_color, 1.0);
                }
            """,
        )
        self._vbo = self._ctx.buffer(reserve=8192)
        self._vao = self._ctx.simple_vertex_array(self._program, self._vbo, "in_pos", "in_color")
        return self.window

    def run(self) -> None:
        if self.window is None:
            self.create_window()
        try:
            while not glfw.window_should_close(self.window):
                mouse_x_px, mouse_y_px = glfw.get_cursor_pos(self.window)
                self.world.win_proc("mouse_move", mouse_x_px, mouse_y_px, self.width, self.height)

                left_pressed = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                if left_pressed and not self._left_down:
                    self.world.win_proc("mouse_click", mouse_x_px, mouse_y_px, self.width, self.height)
                self._left_down = left_pressed

                self._draw_world()
                glfw.swap_buffers(self.window)
                glfw.poll_events()
        finally:
            if self.window is not None:
                glfw.destroy_window(self.window)
                self.window = None
            glfw.terminate()


class Win32Renderer(ModernGLRenderer):
    pass
