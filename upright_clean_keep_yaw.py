bl_info = {
    "name": "Upright Clean (Keep Yaw)",
    "author": "OpenAI",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Upright Clean",
    "description": "Remove pitch/roll from selected objects while preserving their current yaw heading",
    "category": "Object",
}

import bpy
from bpy.props import BoolProperty, EnumProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Matrix, Vector

EPSILON = 1e-8
LOCAL_FORWARD_AXES = {
    'X': Vector((1.0, 0.0, 0.0)),
    'Y': Vector((0.0, 1.0, 0.0)),
    'Z': Vector((0.0, 0.0, 1.0)),
}
WORLD_UP = Vector((0.0, 0.0, 1.0))


class UprightCleanProperties(PropertyGroup):
    forward_axis: EnumProperty(
        name="Forward Axis",
        description="Which local axis should be treated as the forward direction",
        items=(
            ('X', "X", "Local +X is forward"),
            ('Y', "Y", "Local +Y is forward"),
            ('Z', "Z", "Local +Z is forward"),
        ),
        default='Y',
    )

    up_axis: EnumProperty(
        name="Up Axis",
        description="World up axis to align against",
        items=(
            ('Z', "Z", "World +Z is up"),
        ),
        default='Z',
    )

    preserve_location: BoolProperty(
        name="Preserve Location",
        description="Keep world-space location unchanged",
        default=True,
    )

    preserve_scale: BoolProperty(
        name="Preserve Scale",
        description="Keep world-space scale unchanged",
        default=True,
    )

    include_children: BoolProperty(
        name="Include Children",
        description="Also process descendants of selected objects",
        default=False,
    )


def gather_targets(selected_objects, include_children):
    ordered = []
    seen = set()

    def visit(obj):
        if obj in seen:
            return
        seen.add(obj)
        ordered.append(obj)
        if include_children:
            for child in obj.children:
                visit(child)

    for obj in selected_objects:
        visit(obj)

    return ordered


def flatten_on_plane(vector, plane_normal):
    return vector - (vector.dot(plane_normal) * plane_normal)


def compose_world_matrix(location, rotation_matrix_3x3, scale):
    rotation_matrix = rotation_matrix_3x3.to_4x4()
    scale_matrix = Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0))
    return Matrix.Translation(location) @ rotation_matrix @ scale_matrix


def build_clean_world_rotation(forward_clean, forward_axis):
    right = WORLD_UP.cross(forward_clean)
    if right.length_squared < EPSILON:
        return None
    right.normalize()

    forward_clean = right.cross(WORLD_UP)
    if forward_clean.length_squared < EPSILON:
        return None
    forward_clean.normalize()

    if forward_axis == 'Y':
        x_axis = right
        y_axis = forward_clean
        z_axis = WORLD_UP
    elif forward_axis == 'X':
        x_axis = forward_clean
        y_axis = WORLD_UP.cross(x_axis)
        if y_axis.length_squared < EPSILON:
            return None
        y_axis.normalize()
        z_axis = x_axis.cross(y_axis)
        if z_axis.length_squared < EPSILON:
            return None
        z_axis.normalize()
    else:  # forward_axis == 'Z'
        z_axis = forward_clean
        y_axis = WORLD_UP.copy()
        x_axis = y_axis.cross(z_axis)
        if x_axis.length_squared < EPSILON:
            return None
        x_axis.normalize()
        y_axis = z_axis.cross(x_axis)
        if y_axis.length_squared < EPSILON:
            return None
        y_axis.normalize()

    # Matrix columns define local X, Y, Z axes in world space.
    return Matrix((x_axis, y_axis, z_axis)).transposed()


class OBJECT_OT_upright_clean_keep_yaw(Operator):
    bl_idname = "object.upright_clean_keep_yaw"
    bl_label = "Upright Selected"
    bl_description = "Remove tilt from selected objects while preserving heading"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        props = context.scene.upright_clean_props
        selected = list(context.selected_objects)

        if not selected:
            self.report({'ERROR'}, "Select at least one object in Object Mode")
            return {'CANCELLED'}

        targets = gather_targets(selected, props.include_children)
        processed_count = 0

        for obj in targets:
            original_matrix = obj.matrix_world.copy()
            original_location, _, original_scale = original_matrix.decompose()

            local_forward = LOCAL_FORWARD_AXES[props.forward_axis]
            forward_world = (obj.matrix_world.to_3x3() @ local_forward).normalized()

            fallback_used = False
            forward_flat = flatten_on_plane(forward_world, WORLD_UP)

            if forward_flat.length_squared < EPSILON:
                fallback_used = True
                fallback_world_y = (obj.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
                forward_flat = flatten_on_plane(fallback_world_y, WORLD_UP)

            if forward_flat.length_squared < EPSILON:
                self.report({'WARNING'}, f"Skipping {obj.name}: unable to resolve heading on XY plane")
                print(
                    f"[Upright Clean] {obj.name} | "
                    f"forward_world={tuple(round(v, 6) for v in forward_world)} | "
                    f"forward_flat=(0.0, 0.0, 0.0) | fallback_used={fallback_used} | skipped=True"
                )
                continue

            forward_flat.normalize()
            clean_rotation = build_clean_world_rotation(forward_flat, props.forward_axis)

            if clean_rotation is None:
                self.report({'WARNING'}, f"Skipping {obj.name}: failed to construct stable orthonormal basis")
                print(
                    f"[Upright Clean] {obj.name} | "
                    f"forward_world={tuple(round(v, 6) for v in forward_world)} | "
                    f"forward_flat={tuple(round(v, 6) for v in forward_flat)} | "
                    f"fallback_used={fallback_used} | skipped=True"
                )
                continue

            final_location = original_location.copy() if props.preserve_location else obj.matrix_world.to_translation()
            final_scale = original_scale.copy() if props.preserve_scale else obj.matrix_world.to_scale()

            obj.matrix_world = compose_world_matrix(final_location, clean_rotation, final_scale)
            processed_count += 1

            print(
                f"[Upright Clean] {obj.name} | "
                f"forward_world={tuple(round(v, 6) for v in forward_world)} | "
                f"forward_flat={tuple(round(v, 6) for v in forward_flat)} | "
                f"fallback_used={fallback_used}"
            )

        context.view_layer.update()
        self.report({'INFO'}, f"Upright-cleaned {processed_count} object(s)")
        return {'FINISHED'}


class VIEW3D_PT_upright_clean_keep_yaw(Panel):
    bl_label = "Upright Clean (Keep Yaw)"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Upright Clean'

    def draw(self, context):
        layout = self.layout
        props = context.scene.upright_clean_props

        layout.prop(props, "forward_axis")
        layout.prop(props, "up_axis")
        layout.prop(props, "preserve_location")
        layout.prop(props, "preserve_scale")
        layout.prop(props, "include_children")
        layout.separator()
        layout.operator(OBJECT_OT_upright_clean_keep_yaw.bl_idname, icon='ORIENTATION_GLOBAL')


classes = (
    UprightCleanProperties,
    OBJECT_OT_upright_clean_keep_yaw,
    VIEW3D_PT_upright_clean_keep_yaw,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.upright_clean_props = PointerProperty(type=UprightCleanProperties)


def unregister():
    del bpy.types.Scene.upright_clean_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
