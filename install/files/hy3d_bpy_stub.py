"""Local stub (NOT Blender). No bpy wheel exists for CPython 3.12 on aarch64.

hy3dpaint/DifferentiableRenderer/mesh_utils.py does `import bpy` at module top, but only
uses it in convert_obj_to_glb() (OBJ->GLB). Run the paint pipeline with save_glb=False and
convert via hy3dpaint.convert_utils.create_glb_with_pbr_materials (pygltflib) as gradio_app.py does.
Any attribute access raises, so accidental use fails loudly.
"""
__version__ = "0.0.0-stub"


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    raise RuntimeError(f"bpy stub: real Blender is unavailable on this machine (accessed bpy.{name}); use save_glb=False")
