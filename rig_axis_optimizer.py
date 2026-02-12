bl_info = {
    "name": "Rig Axis Optimizer",
    "author": "OpenAI",
    "version": (1, 1, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > Rig Tools",
    "description": "Analyze rig axis issues, sync edit/pose workflows, and safely generate rig constraints",
    "category": "Rigging",
}

import math
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from bpy.props import BoolProperty, StringProperty


HINGE_KEYWORDS = ("elbow", "knee", "finger", "toe")
SPHERICAL_KEYWORDS = ("shoulder", "hip", "neck", "wrist")
_DRAW_HANDLER = None


def lower_name(name):
    return name.lower() if name else ""


def is_armature_active(context):
    obj = context.active_object
    return obj is not None and obj.type == 'ARMATURE'


def classify_bone(name):
    n = lower_name(name)
    if any(k in n for k in HINGE_KEYWORDS):
        return 'HINGE'
    if any(k in n for k in SPHERICAL_KEYWORDS):
        return 'SPHERICAL'
    return 'NONE'


def dominant_axis_from_vector(vec):
    vals = [abs(vec.x), abs(vec.y), abs(vec.z)]
    idx = vals.index(max(vals))
    return ('X', 'Y', 'Z')[idx]


def axis_to_flags(axis):
    return {
        'X': (True, False, False),
        'Y': (False, True, False),
        'Z': (False, False, True),
    }[axis]


def get_pose_bones(obj, selected_only=False):
    if not obj or obj.type != 'ARMATURE':
        return []
    if selected_only:
        return [pb for pb in obj.pose.bones if pb.bone.select]
    return list(obj.pose.bones)


def get_bone_primary_axis(pbone):
    vec = pbone.bone.tail_local - pbone.bone.head_local
    if vec.length < 1e-8:
        return 'Y'
    return dominant_axis_from_vector(vec.normalized())


def get_or_create_limit_constraint(pbone, replace_existing=False):
    existing = [c for c in pbone.constraints if c.type == 'LIMIT_ROTATION']
    if existing and not replace_existing:
        return None
    if existing:
        return existing[0]
    con = pbone.constraints.new(type='LIMIT_ROTATION')
    con.name = "RAO_LimitRotation"
    return con


def set_hinge_limits(con, axis, bone_name):
    con.owner_space = 'LOCAL'
    con.use_transform_limit = True

    use_x, use_y, use_z = axis_to_flags(axis)
    con.use_limit_x = use_x
    con.use_limit_y = use_y
    con.use_limit_z = use_z

    n = lower_name(bone_name)
    lo, hi = (0.0, math.radians(120.0))
    if "elbow" in n:
        lo, hi = (0.0, math.radians(135.0))
    elif "knee" in n:
        lo, hi = (0.0, math.radians(140.0))
    elif "finger" in n:
        lo, hi = (0.0, math.radians(90.0))
    elif "toe" in n:
        lo, hi = (0.0, math.radians(80.0))

    con.min_x = lo if use_x else -math.pi
    con.max_x = hi if use_x else math.pi
    con.min_y = lo if use_y else -math.pi
    con.max_y = hi if use_y else math.pi
    con.min_z = lo if use_z else -math.pi
    con.max_z = hi if use_z else math.pi


def set_spherical_limits(con, twist_axis):
    con.owner_space = 'LOCAL'
    con.use_transform_limit = True
    con.use_limit_x = True
    con.use_limit_y = True
    con.use_limit_z = True

    twist = math.radians(45.0)
    swing = math.radians(95.0)

    con.min_x = -twist if twist_axis == 'X' else -swing
    con.max_x = twist if twist_axis == 'X' else swing
    con.min_y = -twist if twist_axis == 'Y' else -swing
    con.max_y = twist if twist_axis == 'Y' else swing
    con.min_z = -twist if twist_axis == 'Z' else -swing
    con.max_z = twist if twist_axis == 'Z' else swing


def capture_edit_roll_data(arm_obj):
    data = {}
    for eb in arm_obj.data.edit_bones:
        parent_roll = eb.parent.roll if eb.parent else None
        data[eb.name] = {
            "roll": eb.roll,
            "parent": eb.parent.name if eb.parent else None,
            "parent_roll": parent_roll,
        }
    return data


def matrix_to_list(matrix):
    return [matrix[r][c] for r in range(4) for c in range(4)]


def list_to_matrix(data):
    if not isinstance(data, (list, tuple)) or len(data) != 16:
        return None
    try:
        rows = [data[0:4], data[4:8], data[8:12], data[12:16]]
        return Matrix(rows)
    except Exception:
        return None


def collect_analysis(context):
    obj = context.active_object
    report = []
    summary = {
        "roll_warnings": 0,
        "hinge_candidates": 0,
        "spherical_candidates": 0,
        "axis_conflicts": 0,
        "unconstrained_hinges": 0,
        "edit_mode_warning": 0,
    }

    if obj is None or obj.type != 'ARMATURE':
        return report, summary

    # Edit mode safety warning.
    has_constraints = any(pbone.constraints for pbone in obj.pose.bones)
    if has_constraints:
        summary["edit_mode_warning"] = 1
        report.append(
            "[WARN] Editing rest pose may break Pose Mode. "
            "Consider duplicating rig or using Snapshot tools."
        )

    original_mode = obj.mode
    roll_data = {}

    try:
        if original_mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        roll_data = capture_edit_roll_data(obj)
    except Exception as exc:
        report.append(f"[WARN] Could not evaluate edit rolls: {exc}")
    finally:
        try:
            if obj.mode != original_mode:
                bpy.ops.object.mode_set(mode=original_mode)
        except Exception:
            pass

    roll_threshold = math.radians(25.0)
    for bone_name, item in roll_data.items():
        if not item["parent"] or item["parent_roll"] is None:
            continue
        diff = abs(item["roll"] - item["parent_roll"])
        if diff > roll_threshold:
            summary["roll_warnings"] += 1
            report.append(
                f"[ROLL] {bone_name}: roll differs from parent by {math.degrees(diff):.1f}°"
            )

    for pbone in obj.pose.bones:
        cls = classify_bone(pbone.name)
        limit_cons = [c for c in pbone.constraints if c.type == 'LIMIT_ROTATION']

        if cls == 'HINGE':
            summary["hinge_candidates"] += 1
            if not limit_cons:
                summary["unconstrained_hinges"] += 1
                report.append(f"[LIMIT] {pbone.name}: hinge candidate has no Limit Rotation")
        elif cls == 'SPHERICAL':
            summary["spherical_candidates"] += 1

        for con in limit_cons:
            axis_count = int(con.use_limit_x) + int(con.use_limit_y) + int(con.use_limit_z)
            primary_axis = get_bone_primary_axis(pbone)

            if cls == 'HINGE' and axis_count != 1:
                summary["axis_conflicts"] += 1
                report.append(
                    f"[AXIS] {pbone.name}: hinge candidate has {axis_count} constrained axes (expected 1)"
                )

            if cls == 'HINGE':
                expected = axis_to_flags(primary_axis)
                actual = (con.use_limit_x, con.use_limit_y, con.use_limit_z)
                if actual != expected:
                    summary["axis_conflicts"] += 1
                    report.append(
                        f"[AXIS] {pbone.name}: limit axis does not match primary {primary_axis} axis"
                    )

            if cls == 'SPHERICAL' and axis_count < 2:
                summary["axis_conflicts"] += 1
                report.append(
                    f"[AXIS] {pbone.name}: spherical candidate constrained on too few axes"
                )

    return report, summary


def format_summary(summary):
    lines = [
        "Rig Axis Optimizer Summary",
        f"Roll warnings: {summary['roll_warnings']}",
        f"Hinge candidates: {summary['hinge_candidates']}",
        f"Spherical candidates: {summary['spherical_candidates']}",
        f"Axis conflicts: {summary['axis_conflicts']}",
        f"Unconstrained hinges: {summary['unconstrained_hinges']}",
    ]
    if summary.get("edit_mode_warning"):
        lines.append("Edit warning: Rest-pose edits may break existing constraints")
    return "\n".join(lines)


def tag_redraw_view3d():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def get_active_bone_axes(context):
    obj = context.active_object
    if not obj or obj.type != 'ARMATURE':
        return None

    if obj.mode == 'POSE' and context.active_pose_bone:
        pbone = context.active_pose_bone
        mat = obj.matrix_world @ pbone.matrix
        return mat.translation, mat.to_3x3(), max(0.02, pbone.length * obj.scale.length / 3.0)

    if obj.mode == 'EDIT' and obj.data.edit_bones.active:
        eb = obj.data.edit_bones.active
        mat = obj.matrix_world @ eb.matrix
        return mat.translation, mat.to_3x3(), max(0.02, eb.length * obj.scale.length / 3.0)

    return None


def draw_axis_overlay():
    scene = bpy.context.scene
    if not getattr(scene, "rao_show_overlay", False):
        return

    data = get_active_bone_axes(bpy.context)
    if not data:
        return

    origin, rot, length = data
    axes = [
        (rot @ Vector((1, 0, 0)), (1.0, 0.1, 0.1, 1.0)),
        (rot @ Vector((0, 1, 0)), (0.1, 1.0, 0.1, 1.0)),
        (rot @ Vector((0, 0, 1)), (0.1, 0.4, 1.0, 1.0)),
    ]

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.5)

    for direction, color in axes:
        points = [origin, origin + direction.normalized() * length]
        batch = batch_for_shader(shader, 'LINES', {"pos": points})
        shader.bind()
        shader.uniform_float("color", color)
        batch.draw(shader)

    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def ensure_draw_handler(enable):
    global _DRAW_HANDLER
    if enable and _DRAW_HANDLER is None:
        _DRAW_HANDLER = bpy.types.SpaceView3D.draw_handler_add(draw_axis_overlay, (), 'WINDOW', 'POST_VIEW')
    elif not enable and _DRAW_HANDLER is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLER, 'WINDOW')
        _DRAW_HANDLER = None
    tag_redraw_view3d()


def overlay_update(self, context):
    ensure_draw_handler(bool(self.rao_show_overlay))


class RAO_OT_analyze_armature(bpy.types.Operator):
    bl_idname = "rao.analyze_armature"
    bl_label = "Analyze Active Armature"
    bl_description = "Analyze selected armature for axis, roll, limit, and pose-space issues"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context)

    def execute(self, context):
        report, summary = collect_analysis(context)
        scene = context.scene
        scene.rao_report_text = "\n".join(report) if report else "No critical issues detected."
        scene.rao_summary_text = format_summary(summary)

        print("\n=== Rig Axis Optimizer Report ===")
        print(scene.rao_summary_text)
        for line in report:
            print(line)

        self.report({'INFO'}, "Rig analysis complete. See console and panel summary.")
        return {'FINISHED'}


class RAO_OT_duplicate_armature_safe(bpy.types.Operator):
    bl_idname = "rao.duplicate_armature_safe"
    bl_label = "Duplicate Armature For Safe Edit"
    bl_description = "Duplicate active armature object/data and append _EDITSAFE"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context)

    def execute(self, context):
        src = context.active_object
        try:
            dup = src.copy()
            dup.data = src.data.copy()
            dup.animation_data_clear()
            dup.name = f"{src.name}_EDITSAFE"
            dup.data.name = f"{src.data.name}_EDITSAFE"
            context.collection.objects.link(dup)
            dup.matrix_world = src.matrix_world.copy()
            self.report({'INFO'}, f"Created duplicate: {dup.name}")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Duplication failed: {exc}")
            return {'CANCELLED'}


class RAO_OT_align_roll_to_parent(bpy.types.Operator):
    bl_idname = "rao.align_roll_to_parent"
    bl_label = "Align Bone Roll to Parent"
    bl_description = "In Edit Mode, align selected bone rolls to parent orientation"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE' and obj.mode == 'EDIT'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj = context.active_object
        adjusted = 0
        for eb in obj.data.edit_bones:
            if eb.select and eb.parent:
                eb.roll = eb.parent.roll
                adjusted += 1
        self.report({'INFO'}, f"Aligned roll for {adjusted} selected bones.")
        return {'FINISHED'}


class RAO_OT_generate_limits(bpy.types.Operator):
    bl_idname = "rao.generate_limits"
    bl_label = "Generate Mechanical Limits"
    bl_description = "Add/update LOCAL-space Limit Rotation constraints for hinge/spherical candidates"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context)

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        original_mode = obj.mode

        try:
            if obj.mode != 'POSE':
                bpy.ops.object.mode_set(mode='POSE')

            changed, skipped = 0, 0
            for pbone in obj.pose.bones:
                cls = classify_bone(pbone.name)
                if cls == 'NONE':
                    continue

                con = get_or_create_limit_constraint(pbone, scene.rao_replace_existing_limits)
                if con is None:
                    skipped += 1
                    continue

                axis = get_bone_primary_axis(pbone)
                if cls == 'HINGE':
                    set_hinge_limits(con, axis, pbone.name)
                else:
                    set_spherical_limits(con, axis)
                changed += 1

            self.report({'INFO'}, f"Limits generated/updated: {changed}, skipped: {skipped}")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Limit generation failed: {exc}")
            return {'CANCELLED'}
        finally:
            try:
                if obj.mode != original_mode:
                    bpy.ops.object.mode_set(mode=original_mode)
            except Exception:
                pass


class RAO_OT_snapshot_pose(bpy.types.Operator):
    bl_idname = "rao.snapshot_pose"
    bl_label = "Snapshot Current Pose"
    bl_description = "Store pose matrix_basis snapshot in Scene custom properties"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE' and obj.mode == 'POSE'

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        use_selected = scene.rao_snapshot_selected_only
        bones = get_pose_bones(obj, selected_only=use_selected)

        snapshot = {pb.name: matrix_to_list(pb.matrix_basis.copy()) for pb in bones}
        scene["rao_pose_snapshot"] = snapshot
        scene.rao_snapshot_info = f"Stored {len(snapshot)} pose transforms"
        self.report({'INFO'}, scene.rao_snapshot_info)
        return {'FINISHED'}


class RAO_OT_restore_pose(bpy.types.Operator):
    bl_idname = "rao.restore_pose"
    bl_label = "Restore Snapshot Pose"
    bl_description = "Restore saved matrix_basis snapshot to pose bones"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE' and obj.mode == 'POSE'

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        data = scene.get("rao_pose_snapshot", {})
        if not isinstance(data, dict) or not data:
            self.report({'WARNING'}, "No pose snapshot available")
            return {'CANCELLED'}

        selected_names = {pb.name for pb in get_pose_bones(obj, selected_only=True)}
        restored = 0
        for pb in obj.pose.bones:
            if scene.rao_restore_selected_only and pb.name not in selected_names:
                continue
            mat = list_to_matrix(data.get(pb.name))
            if mat is None:
                continue
            pb.matrix_basis = mat
            restored += 1

        context.view_layer.update()
        self.report({'INFO'}, f"Restored snapshot to {restored} pose bones")
        return {'FINISHED'}


class RAO_OT_recalc_childof_inverses(bpy.types.Operator):
    bl_idname = "rao.recalc_childof_inverses"
    bl_label = "Recalculate Child Of Inverses"
    bl_description = "Run Child Of Set Inverse for selected bones while preserving visual pose"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE' and obj.mode == 'POSE'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj = context.active_object
        total = 0
        failed = 0

        for pbone in get_pose_bones(obj, selected_only=True):
            for con in pbone.constraints:
                if con.type != 'CHILD_OF' or con.mute:
                    continue
                try:
                    pbone.bone.select = True
                    obj.data.bones.active = pbone.bone
                    override = context.copy()
                    override["object"] = obj
                    override["active_object"] = obj
                    override["pose_bone"] = pbone
                    override["active_pose_bone"] = pbone
                    with context.temp_override(**override):
                        bpy.ops.constraint.childof_set_inverse(constraint=con.name, owner='BONE')
                    total += 1
                except Exception:
                    failed += 1

        self.report({'INFO'}, f"Child Of inverses recalculated: {total}, failed: {failed}")
        return {'FINISHED'}


class RAO_OT_recalc_constraint_space(bpy.types.Operator):
    bl_idname = "rao.recalc_constraint_space"
    bl_label = "Recalculate IK / Constraint Space"
    bl_description = "Normalize constraint spaces after edit changes and report WORLD-space constraints"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE' and obj.mode == 'POSE'

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        converted = 0
        world_space_lines = []

        for pbone in get_pose_bones(obj, selected_only=True):
            for con in pbone.constraints:
                if con.type in {'COPY_ROTATION', 'COPY_TRANSFORMS'}:
                    if getattr(con, "owner_space", None) == 'WORLD':
                        world_space_lines.append(f"{pbone.name}: {con.name} owner_space=WORLD")
                    if getattr(con, "target_space", None) == 'WORLD':
                        world_space_lines.append(f"{pbone.name}: {con.name} target_space=WORLD")

                    if scene.rao_force_local_copy_space:
                        if hasattr(con, "owner_space") and con.owner_space != 'LOCAL':
                            con.owner_space = 'LOCAL'
                            converted += 1
                        if hasattr(con, "target_space") and con.target_space != 'LOCAL':
                            con.target_space = 'LOCAL'
                            converted += 1

                elif con.type == 'LIMIT_ROTATION':
                    if getattr(con, "owner_space", None) == 'WORLD':
                        world_space_lines.append(f"{pbone.name}: {con.name} owner_space=WORLD")
                    if con.owner_space != 'LOCAL':
                        con.owner_space = 'LOCAL'
                        converted += 1

        scene.rao_constraint_report = "\n".join(world_space_lines) if world_space_lines else "No WORLD-space constraints found on selected bones."
        self.report({'INFO'}, f"Constraint spaces updated: {converted}")
        return {'FINISHED'}


class RAO_PT_main_panel(bpy.types.Panel):
    bl_label = "Rig Axis Optimizer"
    bl_idname = "RAO_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Rig Tools"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.active_object

        analyze_box = layout.box()
        analyze_box.label(text="Analyze Armature")
        row = analyze_box.row()
        row.enabled = is_armature_active(context)
        row.operator("rao.analyze_armature", icon='VIEWZOOM')
        row = analyze_box.row()
        row.enabled = is_armature_active(context)
        row.operator("rao.duplicate_armature_safe", icon='DUPLICATE')

        fixes_box = layout.box()
        fixes_box.label(text="Suggested Fixes")
        if scene.rao_summary_text.strip():
            for line in scene.rao_summary_text.split("\n"):
                fixes_box.label(text=line)
        else:
            fixes_box.label(text="Run analysis to see summary.")

        detail = scene.rao_report_text.strip()
        if detail:
            col = fixes_box.column(align=True)
            for line in detail.split("\n")[:7]:
                col.label(text=line)
            if len(detail.split("\n")) > 7:
                col.label(text="...")

        auto_box = layout.box()
        auto_box.label(text="Auto-Fix Options")
        auto_box.prop(scene, "rao_replace_existing_limits", text="Replace Existing Limits")
        auto_box.prop(scene, "rao_show_overlay", text="Show Axis Overlay")
        row = auto_box.row()
        row.enabled = obj is not None and obj.type == 'ARMATURE' and obj.mode == 'EDIT'
        row.operator("rao.align_roll_to_parent", icon='DRIVER_ROTATIONAL_DIFFERENCE')

        limit_box = layout.box()
        limit_box.label(text="Limit Generator")
        row = limit_box.row()
        row.enabled = is_armature_active(context)
        row.operator("rao.generate_limits", icon='CON_ROTLIKE')

        sync_box = layout.box()
        sync_box.label(text="Edit/Pose Sync Tools")
        sync_box.prop(scene, "rao_snapshot_selected_only", text="Snapshot Selected Bones Only")
        sync_box.prop(scene, "rao_restore_selected_only", text="Restore Selected Bones Only")
        row = sync_box.row(align=True)
        row.enabled = obj is not None and obj.type == 'ARMATURE' and obj.mode == 'POSE'
        row.operator("rao.snapshot_pose", icon='IMPORT')
        row.operator("rao.restore_pose", icon='LOOP_BACK')
        sync_box.label(text=scene.rao_snapshot_info)
        row = sync_box.row()
        row.enabled = obj is not None and obj.type == 'ARMATURE' and obj.mode == 'POSE'
        row.operator("rao.recalc_childof_inverses", icon='CON_CHILDOF')
        sync_box.prop(scene, "rao_force_local_copy_space", text="Force LOCAL for Copy Constraints")
        row = sync_box.row()
        row.enabled = obj is not None and obj.type == 'ARMATURE' and obj.mode == 'POSE'
        row.operator("rao.recalc_constraint_space", icon='CONSTRAINT')
        if scene.rao_constraint_report.strip():
            col = sync_box.column(align=True)
            for line in scene.rao_constraint_report.split("\n")[:6]:
                col.label(text=line)
            if len(scene.rao_constraint_report.split("\n")) > 6:
                col.label(text="...")


classes = (
    RAO_OT_analyze_armature,
    RAO_OT_duplicate_armature_safe,
    RAO_OT_align_roll_to_parent,
    RAO_OT_generate_limits,
    RAO_OT_snapshot_pose,
    RAO_OT_restore_pose,
    RAO_OT_recalc_childof_inverses,
    RAO_OT_recalc_constraint_space,
    RAO_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.rao_replace_existing_limits = BoolProperty(
        name="Replace Existing Limits",
        description="Update existing Limit Rotation constraints",
        default=False,
    )
    bpy.types.Scene.rao_show_overlay = BoolProperty(
        name="Show Axis Overlay",
        description="Display local axis lines for active bone",
        default=False,
        update=overlay_update,
    )
    bpy.types.Scene.rao_snapshot_selected_only = BoolProperty(
        name="Snapshot Selected Bones Only",
        default=False,
    )
    bpy.types.Scene.rao_restore_selected_only = BoolProperty(
        name="Restore Selected Bones Only",
        default=False,
    )
    bpy.types.Scene.rao_force_local_copy_space = BoolProperty(
        name="Force LOCAL for Copy Constraints",
        default=True,
    )
    bpy.types.Scene.rao_summary_text = StringProperty(name="Summary", default="")
    bpy.types.Scene.rao_report_text = StringProperty(name="Report", default="")
    bpy.types.Scene.rao_snapshot_info = StringProperty(name="Snapshot Info", default="No snapshot stored")
    bpy.types.Scene.rao_constraint_report = StringProperty(name="Constraint Report", default="")


def unregister():
    ensure_draw_handler(False)

    del bpy.types.Scene.rao_constraint_report
    del bpy.types.Scene.rao_snapshot_info
    del bpy.types.Scene.rao_report_text
    del bpy.types.Scene.rao_summary_text
    del bpy.types.Scene.rao_force_local_copy_space
    del bpy.types.Scene.rao_restore_selected_only
    del bpy.types.Scene.rao_snapshot_selected_only
    del bpy.types.Scene.rao_show_overlay
    del bpy.types.Scene.rao_replace_existing_limits

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
