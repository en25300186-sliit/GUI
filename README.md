# GUI

Neural UI core with:
- preallocated tensor-backed object storage for NumPy/CuPy backends
- iterative GPU/array-side parent-child global transform propagation (no recursive hierarchy walk)
- event dispatch (hover/click) and top-most hit detection
- optional `ModernGLRenderer` (glfw + moderngl shader pipeline) for GPU rendering
- optional `InstancedModernGLRenderer` for batched instanced quad rendering with optional per-object colors
- optional per-object image costumes/flipbook animation via texture-array layers (`Object`/`SpriteObject`)

Run the example:
- `python example.py`
