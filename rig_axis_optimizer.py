bl_info = {
    "name": "Rig Doctor",
    "author": "OpenAI",
    "version": (2, 0, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > Rig Tools",
    "description": "Diagnose and safely fix common rigging issues in Edit and Pose workflows",
    "category": "Rigging",
}

import math
import bpy
import gpu
from mathutils import Matrix, Vector
from gpu_extras.batch import batch_for_shader
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty, CollectionProperty
from bpy.types import PropertyGroup


HINGE_KEYWORDS = ("elbow", "knee", "finger", "toe")
SPHERICAL_KEYWORDS = ("shoulder", "hip", "neck", "wrist")
DEF_TAGS = ("DEF", "DEF_")
CTRL_TAGS = ("CTRL", "CTRL_")
MCH_TAGS = ("MCH", "MCH_")
ROLL_THRESHOLD = math.radians(25.0)
_DRAW_HANDLER = None


def safe_lower(name):
    return name.lower() if name else ""


def is_armature_active(context):
    obj = context.active_object
    return obj is not None and obj.type == 'ARMATURE'


def classify_joint(name):
    n = safe_lower(name)
    if any(k in n for k in HINGE_KEYWORDS):
        return "HINGE"
    if any(k in n for k in SPHERICAL_KEYWORDS):
        return "SPHERICAL"
    return "OTHER"


def add_report(scene, level, message):
    item = scene.rd_reports.add()
    item.level = level
    item.message = message


def clear_reports(scene):
    scene.rd_reports.clear()
    scene.rd_summary_text = ""


def matrix_to_list(mat):
    return [mat[r][c] for r in range(4) for c in range(4)]


def list_to_matrix(data):
    if not isinstance(data, (list, tuple)) or len(data) != 16:
        return None
    try:
        return Matrix((data[0:4], data[4:8], data[8:12], data[12:16]))
    except Exception:
        return None


def get_pose_bones(obj, selected_only=False):
    if selected_only:
        return [pb for pb in obj.pose.bones if pb.bone.select]
    return list(obj.pose.bones)


def dominant_axis(vec):
    vals = [abs(vec.x), abs(vec.y), abs(vec.z)]
    return ("X", "Y", "Z")[vals.index(max(vals))]


def get_primary_axis(pbone):
    bone_vec = pbone.bone.tail_local - pbone.bone.head_local
    return dominant_axis(bone_vec) if bone_vec.length > 1e-8 else "Y"


def apply_hinge_limits(con, axis, bone_name):
    ranges = {
        "elbow": (0.0, math.radians(135.0)),
        "knee": (0.0, math.radians(140.0)),
        "finger": (0.0, math.radians(90.0)),
        "toe": (0.0, math.radians(80.0)),
        "wrist": (-math.radians(45.0), math.radians(45.0)),
    }
    lo, hi = (0.0, math.radians(120.0))
    ln = safe_lower(bone_name)
    for key, val in ranges.items():
        if key in ln:
            lo, hi = val
            break

    con.owner_space = 'LOCAL'
    con.use_transform_limit = True
    con.use_limit_x = axis == 'X'
    con.use_limit_y = axis == 'Y'
    con.use_limit_z = axis == 'Z'

    con.min_x, con.max_x = ((lo, hi) if con.use_limit_x else (-math.pi, math.pi))
    con.min_y, con.max_y = ((lo, hi) if con.use_limit_y else (-math.pi, math.pi))
    con.min_z, con.max_z = ((lo, hi) if con.use_limit_z else (-math.pi, math.pi))


def apply_spherical_limits(con, twist_axis, bone_name):
    ln = safe_lower(bone_name)
    twist = math.radians(45.0)
    swing = math.radians(95.0)
    if "shoulder" in ln or "hip" in ln:
        swing = math.radians(110.0)
    if "neck" in ln:
        swing = math.radians(65.0)

    con.owner_space = 'LOCAL'
    con.use_transform_limit = True
    con.use_limit_x = True
    con.use_limit_y = True
    con.use_limit_z = True
    con.min_x, con.max_x = (-twist, twist) if twist_axis == 'X' else (-swing, swing)
    con.min_y, con.max_y = (-twist, twist) if twist_axis == 'Y' else (-swing, swing)
    con.min_z, con.max_z = (-twist, twist) if twist_axis == 'Z' else (-swing, swing)


def get_or_create_limit(pbone, replace_existing):
    existing = [c for c in pbone.constraints if c.type == 'LIMIT_ROTATION']
    if existing:
        if replace_existing:
            return existing[0]
        return None
    con = pbone.constraints.new(type='LIMIT_ROTATION')
    con.name = "RD_LimitRotation"
    return con


def map_counterpart_name(name, src_prefix, dst_prefix):
    out = name
    if name.startswith(src_prefix):
        out = dst_prefix + name[len(src_prefix):]
    out = out.replace(f"_{src_prefix}", f"_{dst_prefix}")
    out = out.replace(src_prefix + "_", dst_prefix + "_")
    out = out.replace(src_prefix, dst_prefix)
    return out


def get_active_bone_axes(context):
    obj = context.active_object
    if not obj or obj.type != 'ARMATURE':
        return None
    if obj.mode == 'POSE' and context.active_pose_bone:
        pb = context.active_pose_bone
        mat = obj.matrix_world @ pb.matrix
        return mat.translation, mat.to_3x3(), max(0.02, pb.length * 0.6)
    if obj.mode == 'EDIT' and obj.data.edit_bones.active:
        eb = obj.data.edit_bones.active
        mat = obj.matrix_world @ eb.matrix
        return mat.translation, mat.to_3x3(), max(0.02, eb.length * 0.6)
    return None


def draw_axis_overlay():
    if not bpy.context.scene.rd_show_axis_overlay:
        return
    data = get_active_bone_axes(bpy.context)
    if not data:
        return
    origin, rot, length = data
    dirs = [
        (rot @ Vector((1, 0, 0)), (1.0, 0.1, 0.1, 1.0)),
        (rot @ Vector((0, 1, 0)), (0.1, 1.0, 0.1, 1.0)),
        (rot @ Vector((0, 0, 1)), (0.1, 0.4, 1.0, 1.0)),
    ]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.line_width_set(2.5)
    for direction, color in dirs:
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
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class RD_ReportItem(PropertyGroup):
    level: EnumProperty(items=[('INFO', 'INFO', ''), ('WARN', 'WARN', ''), ('ERROR', 'ERROR', '')], default='INFO')
    message: StringProperty(default="")


class RD_OT_run_diagnostics(bpy.types.Operator):
    bl_idname = "rd.run_diagnostics"
    bl_label = "Rig Doctor: Run Full Diagnostics"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context)

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        clear_reports(scene)
        counts = {'INFO': 0, 'WARN': 0, 'ERROR': 0}

        def push(level, msg):
            add_report(scene, level, msg)
            counts[level] += 1
            print(f"[{level}] {msg}")

        print("\n=== Rig Doctor Diagnostics ===")

        if obj.scale != Vector((1.0, 1.0, 1.0)):
            push('WARN', f"Armature object scale is {tuple(round(v, 4) for v in obj.scale)} (expected 1,1,1)")
        if obj.rotation_euler.length > 1e-6:
            push('WARN', "Armature object rotation is not zero; unapplied transforms can destabilize constraints")

        has_constraints = any(pb.constraints for pb in obj.pose.bones)
        if has_constraints:
            push('WARN', "Editing rest pose may break Pose Mode. Consider duplicating rig or using Snapshot tools")

        original_mode = obj.mode
        try:
            if obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            for eb in obj.data.edit_bones:
                if eb.parent:
                    diff = abs(eb.roll - eb.parent.roll)
                    if diff > ROLL_THRESHOLD:
                        push('WARN', f"Roll discontinuity: {eb.name} differs from parent by {math.degrees(diff):.1f}°")
        except Exception as exc:
            push('ERROR', f"Bone roll analysis failed: {exc}")
        finally:
            try:
                bpy.ops.object.mode_set(mode=original_mode)
            except Exception:
                pass

        for pb in obj.pose.bones:
            jtype = classify_joint(pb.name)
            has_limit = any(c.type == 'LIMIT_ROTATION' for c in pb.constraints)
            if jtype == 'HINGE' and not has_limit:
                push('WARN', f"{pb.name}: hinge-like joint is missing Limit Rotation")

            for con in pb.constraints:
                owner_space = getattr(con, "owner_space", "N/A")
                target_space = getattr(con, "target_space", "N/A")
                if owner_space == 'WORLD' or target_space == 'WORLD':
                    push('WARN', f"{pb.name}/{con.name}: WORLD space found (likely jump risk)")
                if hasattr(con, "target") and con.target and con.target.type == 'ARMATURE' and getattr(con, "subtarget", ""):
                    if con.subtarget not in con.target.data.bones:
                        push('ERROR', f"{pb.name}/{con.name}: subtarget bone '{con.subtarget}' does not exist")
                if hasattr(con, "target") and con.target is None and con.type in {'IK', 'COPY_ROTATION', 'COPY_TRANSFORMS', 'CHILD_OF'}:
                    push('ERROR', f"{pb.name}/{con.name}: missing target object")

                if con.type == 'IK':
                    chain_len = con.chain_count
                    if chain_len == 0:
                        chain_len = 2
                    push('INFO', f"IK {pb.name}/{con.name}: chain={chain_len}, pole_angle={math.degrees(con.pole_angle):.1f}°")
                    if con.pole_target is None:
                        push('WARN', f"IK {pb.name}/{con.name}: missing pole target")
                    else:
                        head = obj.matrix_world @ pb.head
                        pole_pos = con.pole_target.matrix_world.translation
                        if (pole_pos - head).length < max(pb.length * 0.2, 0.02):
                            push('WARN', f"IK {pb.name}/{con.name}: pole target too close to chain (likely flip)")

                if con.type == 'LIMIT_ROTATION' and jtype == 'HINGE':
                    axis_count = int(con.use_limit_x) + int(con.use_limit_y) + int(con.use_limit_z)
                    if con.owner_space != 'LOCAL':
                        push('WARN', f"{pb.name}/{con.name}: hinge limit in non-LOCAL space")
                    if axis_count != 1:
                        push('WARN', f"{pb.name}/{con.name}: hinge joint has {axis_count} constrained axes")

            name_u = pb.name.upper()
            if any(tag in name_u for tag in DEF_TAGS) and pb.constraints:
                push('INFO', f"{pb.name}: deform-tagged bone has constraints (verify intended control/deform split)")
            if any(tag in name_u for tag in CTRL_TAGS) and pb.bone.use_deform:
                push('WARN', f"{pb.name}: control-tagged bone has use_deform=True")
            if any(tag in name_u for tag in MCH_TAGS) and pb.bone.use_deform:
                push('WARN', f"{pb.name}: mechanism-tagged bone has use_deform=True")

            for con in pb.constraints:
                if con.type == 'COPY_ROTATION' and getattr(con, 'owner_space', 'LOCAL') != 'LOCAL':
                    push('WARN', f"{pb.name}/{con.name}: Copy Rotation may fight parenting in {con.owner_space} space")

        if obj.animation_data and obj.animation_data.action:
            push('WARN', "Armature object has keyframes; some fixes can change evaluation results")
        elif any(pb.id_data.animation_data and pb.id_data.animation_data.action for pb in obj.pose.bones):
            push('WARN', "Pose animation detected; evaluate fixes on a sandbox copy first")

        scene.rd_summary_text = f"Diagnostics complete: {counts['ERROR']} errors, {counts['WARN']} warnings, {counts['INFO']} info"
        self.report({'INFO'}, scene.rd_summary_text)
        return {'FINISHED'}


class RD_OT_apply_armature_transforms_safe(bpy.types.Operator):
    bl_idname = "rd.apply_armature_transforms_safe"
    bl_label = "Apply Armature Object Transforms (Safe)"

    force: BoolProperty(name="Force", default=False)

    @classmethod
    def poll(cls, context):
        return is_armature_active(context)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, "force")

    def execute(self, context):
        obj = context.active_object
        if obj.animation_data and obj.animation_data.action and not self.force:
            self.report({'WARNING'}, "Armature object has animation; enable Force to proceed")
            return {'CANCELLED'}
        try:
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            self.report({'INFO'}, "Applied armature object rotation/scale")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Transform apply failed: {exc}")
            return {'CANCELLED'}


class RD_OT_normalize_constraint_spaces(bpy.types.Operator):
    bl_idname = "rd.normalize_constraint_spaces"
    bl_label = "Normalize Constraint Spaces"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        changed = 0
        for pb in get_pose_bones(obj, selected_only=True):
            for con in pb.constraints:
                if scene.rd_skip_keepworld and "KEEPWORLD" in con.name.upper():
                    continue
                if con.type in {'COPY_ROTATION', 'COPY_TRANSFORMS'}:
                    if hasattr(con, "owner_space") and con.owner_space != 'LOCAL':
                        con.owner_space = 'LOCAL'
                        changed += 1
                    if hasattr(con, "target_space") and con.target_space != 'LOCAL':
                        con.target_space = 'LOCAL'
                        changed += 1
                elif con.type == 'LIMIT_ROTATION' and con.owner_space != 'LOCAL':
                    con.owner_space = 'LOCAL'
                    changed += 1
        self.report({'INFO'}, f"Constraint spaces normalized; fields changed: {changed}")
        return {'FINISHED'}


class RD_OT_fix_childof_inverses(bpy.types.Operator):
    bl_idname = "rd.fix_childof_inverses"
    bl_label = "Fix Child Of Inverses"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context) and context.active_object.mode == 'POSE'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj = context.active_object
        ok, fail = 0, 0
        for pb in get_pose_bones(obj, selected_only=True):
            world_before = obj.matrix_world @ pb.matrix
            for con in pb.constraints:
                if con.type != 'CHILD_OF' or con.mute:
                    continue
                try:
                    obj.data.bones.active = pb.bone
                    with context.temp_override(object=obj, active_object=obj, pose_bone=pb, active_pose_bone=pb):
                        bpy.ops.constraint.childof_set_inverse(constraint=con.name, owner='BONE')
                    pb.matrix = obj.matrix_world.inverted() @ world_before
                    ok += 1
                except Exception:
                    fail += 1
        self.report({'INFO'}, f"Child Of inverses fixed: {ok}, failed: {fail}")
        return {'FINISHED'}


class RD_OT_rebuild_ik_poles(bpy.types.Operator):
    bl_idname = "rd.rebuild_ik_poles"
    bl_label = "Rebuild IK Pole Targets"

    distance_factor: FloatProperty(name="Distance Factor", default=1.2, min=0.1, max=5.0)
    forward_axis: EnumProperty(name="Forward Axis", items=[('X', 'X', ''), ('Y', 'Y', ''), ('Z', 'Z', '')], default='Y')

    @classmethod
    def poll(cls, context):
        return is_armature_active(context) and context.active_object.mode == 'POSE'

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, "distance_factor")
        self.layout.prop(self, "forward_axis")

    def execute(self, context):
        obj = context.active_object
        created = 0
        for pb in get_pose_bones(obj, selected_only=True):
            for con in pb.constraints:
                if con.type != 'IK' or con.pole_target is not None:
                    continue
                try:
                    pole = bpy.data.objects.new(f"{pb.name}_pole", None)
                    pole.empty_display_type = 'ARROWS'
                    pole.empty_display_size = max(pb.length * 0.25, 0.05)
                    context.collection.objects.link(pole)
                    axis_vec = {'X': Vector((1, 0, 0)), 'Y': Vector((0, 1, 0)), 'Z': Vector((0, 0, 1))}[self.forward_axis]
                    pos = obj.matrix_world @ pb.head + (obj.matrix_world.to_3x3() @ axis_vec) * (pb.length * self.distance_factor)
                    pole.matrix_world.translation = pos
                    con.pole_target = pole
                    con.pole_angle = 0.0
                    created += 1
                except Exception:
                    continue
        self.report({'INFO'}, f"IK pole targets created: {created}")
        return {'FINISHED'}


class RD_OT_snap_ctrl_to_def(bpy.types.Operator):
    bl_idname = "rd.snap_ctrl_to_def"
    bl_label = "Snap CTRL -> DEF"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        obj = context.active_object
        snapped = 0
        for pb in get_pose_bones(obj, selected_only=True):
            target_name = map_counterpart_name(pb.name, "CTRL", "DEF")
            target = obj.pose.bones.get(target_name)
            if target:
                pb.matrix = target.matrix.copy()
                snapped += 1
        self.report({'INFO'}, f"Snapped CTRL bones to DEF bones: {snapped}")
        return {'FINISHED'}


class RD_OT_snap_def_to_ctrl(bpy.types.Operator):
    bl_idname = "rd.snap_def_to_ctrl"
    bl_label = "Snap DEF -> CTRL"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        obj = context.active_object
        snapped = 0
        for pb in get_pose_bones(obj, selected_only=True):
            target_name = map_counterpart_name(pb.name, "DEF", "CTRL")
            target = obj.pose.bones.get(target_name)
            if target:
                pb.matrix = target.matrix.copy()
                snapped += 1
        self.report({'INFO'}, f"Snapped DEF bones to CTRL bones: {snapped}")
        return {'FINISHED'}


class RD_OT_toggle_deform_by_name(bpy.types.Operator):
    bl_idname = "rd.toggle_deform_by_name"
    bl_label = "Toggle Deform Flags by Naming"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context)

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        changed = 0
        bones = obj.data.bones if scene.rd_affect_all_bones else [b for b in obj.data.bones if b.select]
        for bone in bones:
            uname = bone.name.upper()
            if "CTRL" in uname or uname.startswith("CTRL"):
                bone.use_deform = False
                changed += 1
            elif "DEF" in uname or uname.startswith("DEF"):
                bone.use_deform = True
                changed += 1
        self.report({'INFO'}, f"Updated deform flags on {changed} bones")
        return {'FINISHED'}


class RD_OT_generate_repair_limits(bpy.types.Operator):
    bl_idname = "rd.generate_repair_limits"
    bl_label = "Generate/Repair Limit Rotations"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context)

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        original_mode = obj.mode
        changed = 0
        try:
            if obj.mode != 'POSE':
                bpy.ops.object.mode_set(mode='POSE')
            for pb in obj.pose.bones:
                jt = classify_joint(pb.name)
                if jt == 'OTHER':
                    continue
                con = get_or_create_limit(pb, scene.rd_replace_existing_limits)
                if con is None:
                    continue
                axis = get_primary_axis(pb)
                if jt == 'HINGE':
                    apply_hinge_limits(con, axis, pb.name)
                else:
                    apply_spherical_limits(con, axis, pb.name)
                changed += 1
            self.report({'INFO'}, f"Limit constraints generated/repaired: {changed}")
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


class RD_OT_bake_pose_to_rest(bpy.types.Operator):
    bl_idname = "rd.bake_pose_to_rest"
    bl_label = "Bake Pose to Rest Pose (Dangerous Tool)"

    selected_only: BoolProperty(name="Selected Bones Only", default=True)

    @classmethod
    def poll(cls, context):
        return is_armature_active(context) and context.active_object.mode == 'POSE'

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.label(text="Warning: This can break animation")
        self.layout.prop(self, "selected_only")

    def execute(self, context):
        obj = context.active_object
        try:
            if self.selected_only:
                for pb in obj.pose.bones:
                    pb.bone.select = pb.bone.select
                bpy.ops.pose.armature_apply(selected=True)
            else:
                bpy.ops.pose.armature_apply(selected=False)
            self.report({'WARNING'}, "Applied pose as rest pose. Review animation and constraints.")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Bake to rest failed: {exc}")
            return {'CANCELLED'}


class RD_OT_duplicate_sandbox(bpy.types.Operator):
    bl_idname = "rd.duplicate_sandbox"
    bl_label = "Duplicate Armature (Safe Sandbox)"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context)

    def execute(self, context):
        src = context.active_object
        try:
            dup = src.copy()
            dup.data = src.data.copy()
            dup.animation_data_clear()
            dup.name = f"{src.name}_SANDBOX"
            dup.data.name = f"{src.data.name}_SANDBOX"
            context.collection.objects.link(dup)
            dup.matrix_world = src.matrix_world.copy()

            mesh_dups = 0
            for obj in list(context.scene.objects):
                if obj.type != 'MESH':
                    continue
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object == src:
                        mdup = obj.copy()
                        mdup.data = obj.data.copy()
                        mdup.name = f"{obj.name}_SANDBOX"
                        context.collection.objects.link(mdup)
                        mdup.matrix_world = obj.matrix_world.copy()
                        for md in mdup.modifiers:
                            if md.type == 'ARMATURE' and md.object == src:
                                md.object = dup
                        mesh_dups += 1
                        break
            self.report({'INFO'}, f"Sandbox created: {dup.name} (+{mesh_dups} mesh copies)")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Sandbox duplication failed: {exc}")
            return {'CANCELLED'}


class RD_OT_create_debug_view(bpy.types.Operator):
    bl_idname = "rd.create_debug_view"
    bl_label = "Create Rig Debug View"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context)

    def execute(self, context):
        obj = context.active_object
        obj.show_in_front = True
        obj.data.show_axes = True
        obj.data.show_names = True
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.overlay.show_relationship_lines = True
                space.overlay.show_bones = True
        self.report({'INFO'}, "Rig debug viewport settings enabled")
        return {'FINISHED'}


class RD_OT_align_roll_parent(bpy.types.Operator):
    bl_idname = "rd.align_roll_parent"
    bl_label = "Align Bone Roll to Parent"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context) and context.active_object.mode == 'EDIT'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj = context.active_object
        count = 0
        for eb in obj.data.edit_bones:
            if eb.select and eb.parent:
                eb.roll = eb.parent.roll
                count += 1
        self.report({'INFO'}, f"Aligned roll for {count} selected edit bones")
        return {'FINISHED'}


class RD_OT_snapshot_pose(bpy.types.Operator):
    bl_idname = "rd.snapshot_pose"
    bl_label = "Snapshot Current Pose"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        data = {pb.name: matrix_to_list(pb.matrix_basis.copy()) for pb in get_pose_bones(obj, scene.rd_snapshot_selected_only)}
        scene["rigdoctor_pose_snapshot"] = data
        scene.rd_snapshot_info = f"Stored snapshot for {len(data)} bones"
        self.report({'INFO'}, scene.rd_snapshot_info)
        return {'FINISHED'}


class RD_OT_restore_snapshot_pose(bpy.types.Operator):
    bl_idname = "rd.restore_snapshot_pose"
    bl_label = "Restore Snapshot Pose"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        snapshot = scene.get("rigdoctor_pose_snapshot", {})
        if not isinstance(snapshot, dict) or not snapshot:
            self.report({'WARNING'}, "No snapshot found")
            return {'CANCELLED'}
        selected = {pb.name for pb in get_pose_bones(obj, True)}
        restored = 0
        for pb in obj.pose.bones:
            if scene.rd_restore_selected_only and pb.name not in selected:
                continue
            mat = list_to_matrix(snapshot.get(pb.name))
            if mat:
                pb.matrix_basis = mat
                restored += 1
        context.view_layer.update()
        self.report({'INFO'}, f"Restored pose snapshot on {restored} bones")
        return {'FINISHED'}


class RD_OT_recalc_constraint_space(bpy.types.Operator):
    bl_idname = "rd.recalc_constraint_space"
    bl_label = "Recalculate IK / Constraint Space"

    @classmethod
    def poll(cls, context):
        return is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        world_lines = []
        changed = 0
        for pb in get_pose_bones(obj, selected_only=True):
            for con in pb.constraints:
                if con.type in {'COPY_ROTATION', 'COPY_TRANSFORMS'}:
                    if getattr(con, "owner_space", "") == 'WORLD' or getattr(con, "target_space", "") == 'WORLD':
                        world_lines.append(f"{pb.name}/{con.name} uses WORLD space")
                    if scene.rd_force_local_copy:
                        if hasattr(con, 'owner_space') and con.owner_space != 'LOCAL':
                            con.owner_space = 'LOCAL'
                            changed += 1
                        if hasattr(con, 'target_space') and con.target_space != 'LOCAL':
                            con.target_space = 'LOCAL'
                            changed += 1
                elif con.type == 'LIMIT_ROTATION' and con.owner_space != 'LOCAL':
                    con.owner_space = 'LOCAL'
                    changed += 1
        scene.rd_constraint_report = "\n".join(world_lines) if world_lines else "No WORLD-space constraints on selected bones"
        self.report({'INFO'}, f"Constraint space recalculation complete; changed={changed}")
        return {'FINISHED'}


class RD_PT_main_panel(bpy.types.Panel):
    bl_label = "Rig Doctor"
    bl_idname = "RD_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Rig Tools"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.active_object
        pose_ok = obj is not None and obj.type == 'ARMATURE' and obj.mode == 'POSE'

        diag = layout.box()
        diag.label(text="Diagnose")
        row = diag.row()
        row.enabled = is_armature_active(context)
        row.operator("rd.run_diagnostics", icon='VIEWZOOM')
        diag.label(text=scene.rd_summary_text or "Run diagnostics to populate report")
        for item in scene.rd_reports[:10]:
            icon = 'INFO' if item.level == 'INFO' else ('ERROR' if item.level == 'ERROR' else 'ERROR')
            diag.label(text=f"[{item.level}] {item.message}", icon=icon)

        fix = layout.box()
        fix.label(text="Fix Tools")
        fix.operator("rd.apply_armature_transforms_safe", icon='OBJECT_ORIGIN')
        fix.prop(scene, "rd_skip_keepworld")
        row = fix.row()
        row.enabled = pose_ok
        row.operator("rd.normalize_constraint_spaces", icon='CONSTRAINT')
        row = fix.row()
        row.enabled = pose_ok
        row.operator("rd.fix_childof_inverses", icon='CON_CHILDOF')
        row = fix.row()
        row.enabled = pose_ok
        row.operator("rd.rebuild_ik_poles", icon='CON_KINEMATIC')

        snap = layout.box()
        snap.label(text="Control/Deform Tools")
        row = snap.row(align=True)
        row.enabled = pose_ok
        row.operator("rd.snap_ctrl_to_def")
        row.operator("rd.snap_def_to_ctrl")
        snap.prop(scene, "rd_affect_all_bones")
        snap.operator("rd.toggle_deform_by_name", icon='BONE_DATA')

        limits = layout.box()
        limits.label(text="Limit Generator")
        limits.prop(scene, "rd_replace_existing_limits")
        row = limits.row()
        row.enabled = is_armature_active(context)
        row.operator("rd.generate_repair_limits", icon='CON_ROTLIKE')

        sync = layout.box()
        sync.label(text="Edit/Pose Sync Tools")
        sync.prop(scene, "rd_snapshot_selected_only")
        sync.prop(scene, "rd_restore_selected_only")
        row = sync.row(align=True)
        row.enabled = pose_ok
        row.operator("rd.snapshot_pose", icon='IMPORT')
        row.operator("rd.restore_snapshot_pose", icon='LOOP_BACK')
        sync.label(text=scene.rd_snapshot_info)
        row = sync.row()
        row.enabled = pose_ok
        row.operator("rd.recalc_constraint_space")
        sync.prop(scene, "rd_force_local_copy")
        if scene.rd_constraint_report:
            for line in scene.rd_constraint_report.split("\n")[:4]:
                sync.label(text=line)

        safety = layout.box()
        safety.label(text="Safe Sandbox & View")
        row = safety.row()
        row.enabled = is_armature_active(context)
        row.operator("rd.duplicate_sandbox", icon='DUPLICATE')
        row = safety.row()
        row.enabled = is_armature_active(context)
        row.operator("rd.create_debug_view", icon='HIDE_OFF')
        safety.prop(scene, "rd_show_axis_overlay")
        row = safety.row()
        row.enabled = obj is not None and obj.type == 'ARMATURE' and obj.mode == 'EDIT'
        row.operator("rd.align_roll_parent")
        row = safety.row()
        row.enabled = pose_ok
        row.operator("rd.bake_pose_to_rest", icon='ERROR')


classes = (
    RD_ReportItem,
    RD_OT_run_diagnostics,
    RD_OT_apply_armature_transforms_safe,
    RD_OT_normalize_constraint_spaces,
    RD_OT_fix_childof_inverses,
    RD_OT_rebuild_ik_poles,
    RD_OT_snap_ctrl_to_def,
    RD_OT_snap_def_to_ctrl,
    RD_OT_toggle_deform_by_name,
    RD_OT_generate_repair_limits,
    RD_OT_bake_pose_to_rest,
    RD_OT_duplicate_sandbox,
    RD_OT_create_debug_view,
    RD_OT_align_roll_parent,
    RD_OT_snapshot_pose,
    RD_OT_restore_snapshot_pose,
    RD_OT_recalc_constraint_space,
    RD_PT_main_panel,
)


def overlay_update(self, context):
    ensure_draw_handler(bool(self.rd_show_axis_overlay))


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.rd_reports = CollectionProperty(type=RD_ReportItem)
    bpy.types.Scene.rd_summary_text = StringProperty(default="")
    bpy.types.Scene.rd_replace_existing_limits = BoolProperty(default=False)
    bpy.types.Scene.rd_skip_keepworld = BoolProperty(name="Skip constraints containing KEEPWORLD", default=True)
    bpy.types.Scene.rd_affect_all_bones = BoolProperty(name="Affect All Bones", default=False)
    bpy.types.Scene.rd_snapshot_selected_only = BoolProperty(default=False)
    bpy.types.Scene.rd_restore_selected_only = BoolProperty(default=False)
    bpy.types.Scene.rd_snapshot_info = StringProperty(default="No snapshot stored")
    bpy.types.Scene.rd_force_local_copy = BoolProperty(name="Force LOCAL for Copy Constraints", default=True)
    bpy.types.Scene.rd_constraint_report = StringProperty(default="")
    bpy.types.Scene.rd_show_axis_overlay = BoolProperty(default=False, update=overlay_update)


def unregister():
    ensure_draw_handler(False)

    del bpy.types.Scene.rd_show_axis_overlay
    del bpy.types.Scene.rd_constraint_report
    del bpy.types.Scene.rd_force_local_copy
    del bpy.types.Scene.rd_snapshot_info
    del bpy.types.Scene.rd_restore_selected_only
    del bpy.types.Scene.rd_snapshot_selected_only
    del bpy.types.Scene.rd_affect_all_bones
    del bpy.types.Scene.rd_skip_keepworld
    del bpy.types.Scene.rd_replace_existing_limits
    del bpy.types.Scene.rd_summary_text
    del bpy.types.Scene.rd_reports

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
