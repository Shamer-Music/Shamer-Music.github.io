bl_info = {
    "name": "Dominant Axis Clean",
    "author": "OpenAI",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Dominant Axis",
    "description": "Snap selected object forward axes to nearest world axis while preserving key transforms",
    "category": "Object",
}

import bpy
from bpy.props import BoolProperty, EnumProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup
from mathutils import Matrix, Vector

WORLD_AXES = (
    Vector((1.0, 0.0, 0.0)),
    Vector((-1.0, 0.0, 0.0)),
    Vector((0.0, 1.0, 0.0)),
    Vector((0.0, -1.0, 0.0)),
    Vector((0.0, 0.0, 1.0)),
    Vector((0.0, 0.0, -1.0)),
)

LOCAL_FORWARD_AXES = {
    'X': Vector((1.0, 0.0, 0.0)),
    'Y': Vector((0.0, 1.0, 0.0)),
    'Z': Vector((0.0, 0.0, 1.0)),
}


class DominantAxisCleanProperties(PropertyGroup):
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


def get_forward_world_vector(obj, forward_axis_key):
    local_forward = LOCAL_FORWARD_AXES[forward_axis_key]
    world_basis = obj.matrix_world.to_3x3().normalized()
    return (world_basis @ local_forward).normalized()


def nearest_world_axis(vector):
    return max(WORLD_AXES, key=lambda axis: vector.dot(axis))


def get_local_up_axis_key(forward_axis_key):
    return 'Y' if forward_axis_key == 'Z' else 'Z'


def orthogonalize_to_axis(vector, axis):
    projected = vector - axis * vector.dot(axis)
    if projected.length < 1e-8:
        return None
    return projected.normalized()


def build_clean_rotation_matrix(obj, forward_axis_key, target_forward):
    world_basis = obj.matrix_world.to_3x3().normalized()
    local_up_key = get_local_up_axis_key(forward_axis_key)
    local_up = LOCAL_FORWARD_AXES[local_up_key]
    current_up_world = (world_basis @ local_up).normalized()

    up_world = orthogonalize_to_axis(current_up_world, target_forward)
    if up_world is None:
        up_world = orthogonalize_to_axis(Vector((0.0, 0.0, 1.0)), target_forward)
    if up_world is None:
        up_world = orthogonalize_to_axis(Vector((0.0, 1.0, 0.0)), target_forward)
    if up_world is None:
        up_world = orthogonalize_to_axis(Vector((1.0, 0.0, 0.0)), target_forward)

    axis_index = {'X': 0, 'Y': 1, 'Z': 2}
    forward_i = axis_index[forward_axis_key]
    up_i = axis_index[local_up_key]
    other_i = ({0, 1, 2} - {forward_i, up_i}).pop()

    basis = [None, None, None]
    basis[forward_i] = target_forward
    basis[up_i] = up_world

    if other_i == 2:  # z = x × y
        basis[2] = basis[0].cross(basis[1]).normalized()
    elif other_i == 1:  # y = z × x
        basis[1] = basis[2].cross(basis[0]).normalized()
    else:  # x = y × z
        basis[0] = basis[1].cross(basis[2]).normalized()

    if up_i == 2:
        basis[2] = basis[0].cross(basis[1]).normalized()
    elif up_i == 1:
        basis[1] = basis[2].cross(basis[0]).normalized()
    else:
        basis[0] = basis[1].cross(basis[2]).normalized()

    return Matrix((
        (basis[0].x, basis[1].x, basis[2].x),
        (basis[0].y, basis[1].y, basis[2].y),
        (basis[0].z, basis[1].z, basis[2].z),
    ))


def compose_world_matrix(location, rotation_quaternion, scale):
    rotation_matrix = rotation_quaternion.to_matrix().to_4x4()
    scale_matrix = Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0))
    return Matrix.Translation(location) @ rotation_matrix @ scale_matrix


class OBJECT_OT_dominant_axis_clean(Operator):
    bl_idname = "object.dominant_axis_clean"
    bl_label = "Clean Selected Objects"
    bl_description = "Snap selected object forward axis to the nearest world axis direction"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        props = context.scene.dominant_axis_clean_props
        selected = list(context.selected_objects)

        if not selected:
            self.report({'ERROR'}, "Select at least one object in Object Mode")
            return {'CANCELLED'}

        for obj in selected:
            original_matrix = obj.matrix_world.copy()
            original_location, _original_rotation, original_scale = original_matrix.decompose()

            # Keep the clean-basis path (not rotation_difference) to remove residual tilt/twist noise.
            original_forward = get_forward_world_vector(obj, props.forward_axis)
            target_axis = nearest_world_axis(original_forward)
            clean_rotation_matrix = build_clean_rotation_matrix(obj, props.forward_axis, target_axis)
            new_world_rotation = clean_rotation_matrix.to_quaternion()

            final_location = original_location.copy() if props.preserve_location else original_matrix.to_translation()
            final_scale = original_scale.copy() if props.preserve_scale else obj.matrix_world.to_scale()

            obj.matrix_world = compose_world_matrix(final_location, new_world_rotation, final_scale)

            print(
                f"[Dominant Axis Clean] {obj.name} | "
                f"Original Forward: ({original_forward.x:.6f}, {original_forward.y:.6f}, {original_forward.z:.6f}) | "
                f"Target Axis: ({target_axis.x:.1f}, {target_axis.y:.1f}, {target_axis.z:.1f})"
            )

        context.view_layer.update()
        self.report({'INFO'}, f"Cleaned {len(selected)} object(s)")
        return {'FINISHED'}


class VIEW3D_PT_dominant_axis_clean(Panel):
    bl_label = "Dominant Axis Clean"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Dominant Axis'

    def draw(self, context):
        layout = self.layout
        props = context.scene.dominant_axis_clean_props

        layout.prop(props, "forward_axis")
        layout.prop(props, "preserve_location")
        layout.prop(props, "preserve_scale")
        layout.separator()
        layout.operator(OBJECT_OT_dominant_axis_clean.bl_idname, icon='ORIENTATION_GLOBAL')


classes = (
    DominantAxisCleanProperties,
    OBJECT_OT_dominant_axis_clean,
    VIEW3D_PT_dominant_axis_clean,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dominant_axis_clean_props = PointerProperty(type=DominantAxisCleanProperties)


def unregister():
    del bpy.types.Scene.dominant_axis_clean_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
