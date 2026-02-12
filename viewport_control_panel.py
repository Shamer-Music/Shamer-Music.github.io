bl_info = {
    "name": "Viewport Control Panel",
    "author": "OpenAI",
    "version": (1, 0, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > View",
    "description": "Convenient viewport navigation controls with per-file speed and clipping overrides",
    "category": "3D View",
}

import math
from mathutils import Euler, Quaternion, Vector
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_view3d_space_and_region3d(context):
    """Return active SpaceView3D and RegionView3D if available."""
    area = context.area
    space = context.space_data
    if area and area.type == 'VIEW_3D' and space and space.type == 'VIEW_3D':
        return space, space.region_3d
    return None, None


def get_pref_with_fallback(context, attr_names, owner="inputs"):
    """Return (owner_object, attr_name) for first attribute that exists, otherwise (None, None)."""
    prefs = context.preferences
    pref_owner = getattr(prefs, owner, None)
    if pref_owner is None:
        return None, None

    for attr_name in attr_names:
        if hasattr(pref_owner, attr_name):
            return pref_owner, attr_name
    return None, None


def draw_pref_prop(layout, context, label, owner, attrs):
    pref_owner, attr_name = get_pref_with_fallback(context, attrs, owner=owner)
    if pref_owner and attr_name:
        layout.prop(pref_owner, attr_name, text=label)
    else:
        row = layout.row()
        row.enabled = False
        row.label(text=f"{label}: Not available in this build")


def apply_slow_zoom_mode(scene, context):
    """Temporarily disable zoom-to-mouse and auto-depth when enabled."""
    inputs = getattr(context.preferences, "inputs", None)
    if not inputs:
        return

    zoom_owner, zoom_attr = get_pref_with_fallback(context, ["zoom_to_mouse_position", "use_zoom_to_mouse"], owner="inputs")
    depth_owner, depth_attr = get_pref_with_fallback(context, ["use_auto_depth", "use_mouse_depth_navigate"], owner="inputs")

    if scene.vcp_slow_zoom_mode:
        # Cache current values once when entering slow mode.
        if zoom_owner and zoom_attr and not scene.vcp_slow_zoom_cached:
            scene.vcp_prev_zoom_to_mouse = bool(getattr(zoom_owner, zoom_attr))
        if depth_owner and depth_attr and not scene.vcp_slow_zoom_cached:
            scene.vcp_prev_auto_depth = bool(getattr(depth_owner, depth_attr))

        if zoom_owner and zoom_attr:
            setattr(zoom_owner, zoom_attr, False)
        if depth_owner and depth_attr:
            setattr(depth_owner, depth_attr, False)
        scene.vcp_slow_zoom_cached = True
    else:
        # Restore values when leaving slow mode.
        if scene.vcp_slow_zoom_cached:
            if zoom_owner and zoom_attr:
                setattr(zoom_owner, zoom_attr, scene.vcp_prev_zoom_to_mouse)
            if depth_owner and depth_attr:
                setattr(depth_owner, depth_attr, scene.vcp_prev_auto_depth)
            scene.vcp_slow_zoom_cached = False


def restore_slow_zoom_mode_preferences():
    """Restore global input preferences if any scene left slow zoom mode active."""
    context = bpy.context
    if context is None:
        return

    zoom_owner, zoom_attr = get_pref_with_fallback(context, ["zoom_to_mouse_position", "use_zoom_to_mouse"], owner="inputs")
    depth_owner, depth_attr = get_pref_with_fallback(context, ["use_auto_depth", "use_mouse_depth_navigate"], owner="inputs")

    for scene in bpy.data.scenes:
        if not getattr(scene, "vcp_slow_zoom_cached", False):
            continue

        if zoom_owner and zoom_attr:
            setattr(zoom_owner, zoom_attr, bool(scene.vcp_prev_zoom_to_mouse))
        if depth_owner and depth_attr:
            setattr(depth_owner, depth_attr, bool(scene.vcp_prev_auto_depth))

        scene.vcp_slow_zoom_cached = False
        scene.vcp_slow_zoom_mode = False
        break


# -----------------------------------------------------------------------------
# View operators driven by Scene multipliers
# -----------------------------------------------------------------------------

class VCP_OT_zoom_small(bpy.types.Operator):
    bl_idname = "view3d.vcp_zoom_small"
    bl_label = "Zoom Small"
    bl_description = "Zoom in/out in small controlled increments"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(
        items=[
            ("IN", "In", "Zoom in"),
            ("OUT", "Out", "Zoom out"),
        ],
        default="IN",
    )

    @classmethod
    def poll(cls, context):
        _, rv3d = get_view3d_space_and_region3d(context)
        return rv3d is not None

    def execute(self, context):
        scene = context.scene
        _, rv3d = get_view3d_space_and_region3d(context)
        if rv3d is None:
            return {'CANCELLED'}

        factor = max(0.01, scene.vcp_zoom_step_multiplier)
        # Lower factor = subtler zoom; use multiplicative distance updates.
        step = 0.92 ** factor

        if self.direction == "IN":
            rv3d.view_distance = max(0.001, rv3d.view_distance * step)
        else:
            rv3d.view_distance = rv3d.view_distance / step
        return {'FINISHED'}


class VCP_OT_pan_small(bpy.types.Operator):
    bl_idname = "view3d.vcp_pan_small"
    bl_label = "Pan Small"
    bl_description = "Pan view in small controlled increments"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(
        items=[
            ("LEFT", "Left", "Pan left"),
            ("RIGHT", "Right", "Pan right"),
            ("UP", "Up", "Pan up"),
            ("DOWN", "Down", "Pan down"),
        ],
        default="LEFT",
    )

    @classmethod
    def poll(cls, context):
        _, rv3d = get_view3d_space_and_region3d(context)
        return rv3d is not None

    def execute(self, context):
        scene = context.scene
        _, rv3d = get_view3d_space_and_region3d(context)
        if rv3d is None:
            return {'CANCELLED'}

        step = 0.1 * max(0.01, scene.vcp_pan_step_multiplier) * max(0.1, rv3d.view_distance)
        view_rot = rv3d.view_rotation
        right = view_rot @ Vector((1.0, 0.0, 0.0))
        up = view_rot @ Vector((0.0, 1.0, 0.0))

        delta = Vector((0.0, 0.0, 0.0))
        if self.direction == "LEFT":
            delta = -right * step
        elif self.direction == "RIGHT":
            delta = right * step
        elif self.direction == "UP":
            delta = up * step
        elif self.direction == "DOWN":
            delta = -up * step

        rv3d.view_location += delta
        return {'FINISHED'}


class VCP_OT_orbit_small(bpy.types.Operator):
    bl_idname = "view3d.vcp_orbit_small"
    bl_label = "Orbit Small"
    bl_description = "Orbit view in small controlled increments"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(
        items=[
            ("LEFT", "Left", "Orbit left"),
            ("RIGHT", "Right", "Orbit right"),
            ("UP", "Up", "Orbit up"),
            ("DOWN", "Down", "Orbit down"),
        ],
        default="LEFT",
    )

    @classmethod
    def poll(cls, context):
        _, rv3d = get_view3d_space_and_region3d(context)
        return rv3d is not None

    def execute(self, context):
        scene = context.scene
        _, rv3d = get_view3d_space_and_region3d(context)
        if rv3d is None:
            return {'CANCELLED'}

        sensitivity = max(0.01, scene.vcp_orbit_sensitivity)
        angle = math.radians(5.0 * sensitivity)

        current_rot = rv3d.view_rotation
        right_axis = current_rot @ Vector((1.0, 0.0, 0.0))
        world_up = Vector((0.0, 0.0, 1.0))

        if self.direction == "LEFT":
            q = Quaternion(world_up, angle)
        elif self.direction == "RIGHT":
            q = Quaternion(world_up, -angle)
        elif self.direction == "UP":
            q = Quaternion(right_axis, angle)
        else:  # DOWN
            q = Quaternion(right_axis, -angle)

        rv3d.view_rotation = q @ current_rot
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Fix My View operators
# -----------------------------------------------------------------------------

class VCP_OT_frame_selected(bpy.types.Operator):
    bl_idname = "view3d.vcp_frame_selected"
    bl_label = "Frame Selected"
    bl_description = "Frame selected objects"

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D'

    def execute(self, context):
        bpy.ops.view3d.view_selected('INVOKE_DEFAULT')
        return {'FINISHED'}


class VCP_OT_frame_all(bpy.types.Operator):
    bl_idname = "view3d.vcp_frame_all"
    bl_label = "Frame All"
    bl_description = "Frame all visible objects"

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D'

    def execute(self, context):
        bpy.ops.view3d.view_all('INVOKE_DEFAULT', center=False)
        return {'FINISHED'}


class VCP_OT_reset_view(bpy.types.Operator):
    bl_idname = "view3d.vcp_reset_view"
    bl_label = "Reset View"
    bl_description = "Reset viewport to a stable front perspective"

    @classmethod
    def poll(cls, context):
        _, rv3d = get_view3d_space_and_region3d(context)
        return rv3d is not None

    def execute(self, context):
        _, rv3d = get_view3d_space_and_region3d(context)
        if rv3d is None:
            return {'CANCELLED'}

        rv3d.view_perspective = 'PERSP'
        # Front view quaternion aligned with Blender's default front orientation.
        rv3d.view_rotation = Euler((math.radians(90.0), 0.0, 0.0), 'XYZ').to_quaternion()
        rv3d.view_location = Vector((0.0, 0.0, 0.0))
        rv3d.view_distance = 10.0
        return {'FINISHED'}


class VCP_OT_normalize_clipping(bpy.types.Operator):
    bl_idname = "view3d.vcp_normalize_clipping"
    bl_label = "Normalize Clipping"
    bl_description = "Set clipping values to stable defaults"

    @classmethod
    def poll(cls, context):
        space, _ = get_view3d_space_and_region3d(context)
        return space is not None

    def execute(self, context):
        scene = context.scene
        space, _ = get_view3d_space_and_region3d(context)
        if not space:
            return {'CANCELLED'}

        space.clip_start = scene.vcp_clip_start_default
        space.clip_end = scene.vcp_clip_end_default
        return {'FINISHED'}


class VCP_OT_set_pivot_to_selection(bpy.types.Operator):
    bl_idname = "view3d.vcp_set_pivot_to_selection"
    bl_label = "Set Pivot To Selection"
    bl_description = "Frame selection to force a sane orbit pivot"

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D'

    def execute(self, context):
        bpy.ops.view3d.view_selected('INVOKE_DEFAULT')
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Panel
# -----------------------------------------------------------------------------

class VIEW3D_PT_viewport_control_panel(bpy.types.Panel):
    bl_label = "Viewport Control Panel"
    bl_idname = "VIEW3D_PT_viewport_control_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'View'

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        space, rv3d = get_view3d_space_and_region3d(context)

        pref_box = layout.box()
        pref_box.label(text="Navigation Preferences", icon='PREFERENCES')
        draw_pref_prop(pref_box, context, "Zoom To Mouse", "inputs", ["zoom_to_mouse_position", "use_zoom_to_mouse"])
        draw_pref_prop(pref_box, context, "Auto Depth", "inputs", ["use_auto_depth", "use_mouse_depth_navigate"])
        draw_pref_prop(pref_box, context, "Orbit Around Active", "inputs", ["use_rotate_around_active"])
        draw_pref_prop(pref_box, context, "Rotate Method", "inputs", ["view_rotate_method"])
        draw_pref_prop(pref_box, context, "Zoom Method", "inputs", ["view_zoom_method"])
        draw_pref_prop(pref_box, context, "Wheel Zoom Direction", "inputs", ["wheel_scroll_lines", "invert_mouse_zoom"])

        if space:
            clip_box = layout.box()
            clip_box.label(text="Active View Clipping", icon='VIEW_PERSPECTIVE')
            clip_box.prop(space, "clip_start", text="Clip Start")
            clip_box.prop(space, "clip_end", text="Clip End")
        elif rv3d:
            clip_box = layout.box()
            clip_box.label(text="Active View Distance", icon='VIEW_PERSPECTIVE')
            clip_box.prop(rv3d, "view_distance", text="View Distance")

        speed_box = layout.box()
        speed_box.label(text="Per-File Speed Controls", icon='DRIVER')
        speed_box.prop(scene, "vcp_zoom_step_multiplier", slider=True)
        speed_box.prop(scene, "vcp_pan_step_multiplier", slider=True)
        speed_box.prop(scene, "vcp_orbit_sensitivity", slider=True)

        row = speed_box.row(align=True)
        row.operator("view3d.vcp_zoom_small", text="Zoom In Small").direction = "IN"
        row.operator("view3d.vcp_zoom_small", text="Zoom Out Small").direction = "OUT"

        row = speed_box.row(align=True)
        row.operator("view3d.vcp_pan_small", text="Pan Left").direction = "LEFT"
        row.operator("view3d.vcp_pan_small", text="Pan Right").direction = "RIGHT"
        row.operator("view3d.vcp_pan_small", text="Pan Up").direction = "UP"
        row.operator("view3d.vcp_pan_small", text="Pan Down").direction = "DOWN"

        row = speed_box.row(align=True)
        row.operator("view3d.vcp_orbit_small", text="Orbit Left").direction = "LEFT"
        row.operator("view3d.vcp_orbit_small", text="Orbit Right").direction = "RIGHT"
        row.operator("view3d.vcp_orbit_small", text="Orbit Up").direction = "UP"
        row.operator("view3d.vcp_orbit_small", text="Orbit Down").direction = "DOWN"

        fix_box = layout.box()
        fix_box.label(text="Fix My View", icon='TOOL_SETTINGS')
        row = fix_box.row(align=True)
        row.operator("view3d.vcp_frame_selected")
        row.operator("view3d.vcp_frame_all")
        fix_box.operator("view3d.vcp_reset_view")
        fix_box.operator("view3d.vcp_normalize_clipping")
        fix_box.operator("view3d.vcp_set_pivot_to_selection")

        slow_box = layout.box()
        slow_box.label(text="Stability", icon='MOD_SMOOTH')
        slow_box.prop(scene, "vcp_slow_zoom_mode", text="Slow Zoom Mode")

        status_lines = []
        zoom_owner, zoom_attr = get_pref_with_fallback(context, ["zoom_to_mouse_position", "use_zoom_to_mouse"], owner="inputs")
        depth_owner, depth_attr = get_pref_with_fallback(context, ["use_auto_depth", "use_mouse_depth_navigate"], owner="inputs")
        if zoom_owner and zoom_attr:
            status_lines.append(f"ZoomToMouse={'ON' if getattr(zoom_owner, zoom_attr) else 'OFF'}")
        if depth_owner and depth_attr:
            status_lines.append(f"AutoDepth={'ON' if getattr(depth_owner, depth_attr) else 'OFF'}")
        status_lines.append(f"SlowZoom={'ON' if scene.vcp_slow_zoom_mode else 'OFF'}")

        slow_box.label(text=" | ".join(status_lines), icon='INFO')


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

classes = (
    VCP_OT_zoom_small,
    VCP_OT_pan_small,
    VCP_OT_orbit_small,
    VCP_OT_frame_selected,
    VCP_OT_frame_all,
    VCP_OT_reset_view,
    VCP_OT_normalize_clipping,
    VCP_OT_set_pivot_to_selection,
    VIEW3D_PT_viewport_control_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.vcp_zoom_step_multiplier = FloatProperty(
        name="Zoom Step Multiplier",
        description="Per-file zoom step multiplier for addon controls",
        default=1.0,
        min=0.1,
        max=10.0,
    )
    bpy.types.Scene.vcp_pan_step_multiplier = FloatProperty(
        name="Pan Step Multiplier",
        description="Per-file pan step multiplier for addon controls",
        default=1.0,
        min=0.1,
        max=10.0,
    )
    bpy.types.Scene.vcp_orbit_sensitivity = FloatProperty(
        name="Orbit Sensitivity",
        description="Per-file orbit sensitivity for addon controls",
        default=1.0,
        min=0.1,
        max=10.0,
    )
    bpy.types.Scene.vcp_clip_start_default = FloatProperty(
        name="Default Clip Start",
        description="Per-file clip start used by Normalize Clipping",
        default=0.01,
        min=0.0001,
        max=1000.0,
        precision=4,
    )
    bpy.types.Scene.vcp_clip_end_default = FloatProperty(
        name="Default Clip End",
        description="Per-file clip end used by Normalize Clipping",
        default=1000.0,
        min=1.0,
        max=1000000.0,
    )

    bpy.types.Scene.vcp_prev_zoom_to_mouse = BoolProperty(default=False)
    bpy.types.Scene.vcp_prev_auto_depth = BoolProperty(default=False)
    bpy.types.Scene.vcp_slow_zoom_cached = BoolProperty(default=False)
    bpy.types.Scene.vcp_slow_zoom_mode = BoolProperty(
        name="Slow Zoom Mode",
        description="Temporarily disable zoom-to-mouse and auto-depth for stable viewport zooming",
        default=False,
        update=apply_slow_zoom_mode,
    )


def unregister():
    restore_slow_zoom_mode_preferences()

    del bpy.types.Scene.vcp_slow_zoom_mode
    del bpy.types.Scene.vcp_slow_zoom_cached
    del bpy.types.Scene.vcp_prev_auto_depth
    del bpy.types.Scene.vcp_prev_zoom_to_mouse
    del bpy.types.Scene.vcp_clip_end_default
    del bpy.types.Scene.vcp_clip_start_default
    del bpy.types.Scene.vcp_orbit_sensitivity
    del bpy.types.Scene.vcp_pan_step_multiplier
    del bpy.types.Scene.vcp_zoom_step_multiplier

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
