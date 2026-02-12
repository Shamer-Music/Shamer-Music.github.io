bl_info = {
    "name": "Rig Doctor",
    "author": "OpenAI",
    "version": (1, 0, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > Rig Tools",
    "description": "Diagnose and safely repair common rig issues (constraints, IK, roll, transform sanity)",
    "category": "Rigging",
}

import json
import math
import bpy
from mathutils import Matrix, Vector
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty

HINGE_KEYS = ("elbow", "knee", "finger", "toe")
SPHERICAL_KEYS = ("wrist", "shoulder", "hip", "neck")


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def report_clear(scene):
    scene["rd_items"] = "[]"
    scene.rd_summary = "No diagnostics run yet."


def report_add(scene, level, code, text):
    try:
        items = json.loads(scene.get("rd_items", "[]"))
        if not isinstance(items, list):
            items = []
    except Exception:
        items = []
    items.append({"level": level, "code": code, "text": text})
    scene["rd_items"] = json.dumps(items)


def report_items(scene):
    try:
        items = json.loads(scene.get("rd_items", "[]"))
        return items if isinstance(items, list) else []
    except Exception:
        return []


def print_report(scene):
    print("\n=== Rig Doctor Report ===")
    print(scene.rd_summary)
    for i in report_items(scene):
        print(f"[{i.get('level', 'INFO')}] {i.get('code', 'GEN')}: {i.get('text', '')}")


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def active_armature(context):
    obj = context.active_object
    return obj if obj and obj.type == 'ARMATURE' else None


def pose_bones(obj, selected_only=False):
    return [pb for pb in obj.pose.bones if (pb.bone.select or not selected_only)]


def classify_joint(name):
    low = name.lower()
    if any(k in low for k in HINGE_KEYS):
        return "HINGE"
    if any(k in low for k in SPHERICAL_KEYS):
        return "SPHERICAL"
    return "NONE"


def primary_axis(pbone):
    vec = pbone.bone.tail_local - pbone.bone.head_local
    if vec.length < 1e-8:
        return "Y"
    vec = vec.normalized()
    vals = [abs(vec.x), abs(vec.y), abs(vec.z)]
    return ("X", "Y", "Z")[vals.index(max(vals))]


def axis_flags(axis):
    return {
        "X": (True, False, False),
        "Y": (False, True, False),
        "Z": (False, False, True),
    }[axis]


def object_has_transform_animation(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return False
    for f in ad.action.fcurves:
        if f.data_path.startswith(("location", "rotation", "scale")):
            return True
    return False


def rig_has_pose_animation(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return False
    return any(fc.data_path.startswith("pose.bones[") for fc in ad.action.fcurves)


def skip_keepworld(scene, con):
    return scene.rd_skip_keepworld and "KEEPWORLD" in con.name.upper()


def mapped_name(src_name, from_tag, to_tag):
    name = src_name
    name = name.replace(f"{from_tag}_", f"{to_tag}_")
    name = name.replace(f".{from_tag}", f".{to_tag}")
    if name.startswith(from_tag):
        name = name.replace(from_tag, to_tag, 1)
    return name


def has_world_space(con):
    return getattr(con, "owner_space", None) == 'WORLD' or getattr(con, "target_space", None) == 'WORLD'


def set_limit_for_joint(con, joint_name, pbone):
    con.owner_space = 'LOCAL'
    con.use_transform_limit = True

    low = joint_name.lower()
    if any(k in low for k in ("elbow", "knee", "finger", "toe")):
        axis = primary_axis(pbone)
        fx, fy, fz = axis_flags(axis)
        con.use_limit_x, con.use_limit_y, con.use_limit_z = fx, fy, fz

        lo, hi = 0.0, math.radians(120.0)
        if "elbow" in low:
            hi = math.radians(135.0)
        elif "knee" in low:
            hi = math.radians(140.0)
        elif "finger" in low:
            hi = math.radians(90.0)

        con.min_x, con.max_x = (lo, hi) if fx else (-math.pi, math.pi)
        con.min_y, con.max_y = (lo, hi) if fy else (-math.pi, math.pi)
        con.min_z, con.max_z = (lo, hi) if fz else (-math.pi, math.pi)
    else:
        twist_axis = primary_axis(pbone)
        con.use_limit_x = con.use_limit_y = con.use_limit_z = True
        twist = math.radians(45.0)
        swing = math.radians(95.0)

        con.min_x, con.max_x = (-twist, twist) if twist_axis == 'X' else (-swing, swing)
        con.min_y, con.max_y = (-twist, twist) if twist_axis == 'Y' else (-swing, swing)
        con.min_z, con.max_z = (-twist, twist) if twist_axis == 'Z' else (-swing, swing)


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

class RD_OT_run_diagnostics(bpy.types.Operator):
    bl_idname = "rig_doctor.run_diagnostics"
    bl_label = "Rig Doctor: Run Full Diagnostics"

    @classmethod
    def poll(cls, context):
        return active_armature(context) is not None

    def execute(self, context):
        scene = context.scene
        arm = active_armature(context)
        report_clear(scene)

        warns, errs = 0, 0

        # 1) mode/transform sanity
        if any(abs(v - 1.0) > 1e-4 for v in arm.scale):
            warns += 1
            report_add(scene, "WARN", "OBJ_SCALE", f"Armature scale is {tuple(round(v, 4) for v in arm.scale)}.")
        if any(abs(v) > 1e-4 for v in arm.rotation_euler):
            warns += 1
            report_add(scene, "WARN", "OBJ_ROT", "Armature rotation is non-zero; consider applying transforms.")

        # 2) bone roll consistency
        original_mode = arm.mode
        try:
            if original_mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            for eb in arm.data.edit_bones:
                if not eb.parent:
                    continue
                diff = abs(eb.roll - eb.parent.roll)
                if diff > math.radians(25.0):
                    warns += 1
                    report_add(scene, "WARN", "ROLL", f"{eb.name}: roll delta to parent = {math.degrees(diff):.1f}°")
        except Exception as exc:
            errs += 1
            report_add(scene, "ERROR", "ROLL_SCAN", f"Roll scan failed: {exc}")
        finally:
            try:
                if arm.mode != original_mode:
                    bpy.ops.object.mode_set(mode=original_mode)
            except Exception:
                pass

        # 3..8 audits
        for pb in arm.pose.bones:
            joint = classify_joint(pb.name)
            has_limit = False

            nup = pb.name.upper()
            if "CTRL" in nup and pb.bone.use_deform:
                warns += 1
                report_add(scene, "WARN", "CTRL_DEFORM", f"{pb.name} looks like CTRL but use_deform=True")
            if "DEF" in nup and pb.constraints:
                report_add(scene, "INFO", "DEF_HAS_CON", f"{pb.name} is DEF-like and has constraints")

            for con in pb.constraints:
                owner = getattr(con, "owner_space", "N/A")
                target = getattr(con, "target_space", "N/A")
                report_add(scene, "INFO", "CON_SPACE", f"{pb.name}/{con.name}: owner={owner}, target={target}")

                if has_world_space(con):
                    warns += 1
                    report_add(scene, "WARN", "WORLD_SPACE", f"{pb.name}/{con.name} uses WORLD space")

                if hasattr(con, "target") and con.target is None and con.type not in {'LIMIT_ROTATION'}:
                    errs += 1
                    report_add(scene, "ERROR", "MISSING_TARGET", f"{pb.name}/{con.name} has no target")

                if hasattr(con, "subtarget") and con.subtarget:
                    tgt = getattr(con, "target", None)
                    if tgt and tgt.type == 'ARMATURE' and con.subtarget not in tgt.data.bones:
                        errs += 1
                        report_add(scene, "ERROR", "BAD_SUBTARGET", f"{pb.name}/{con.name} subtarget '{con.subtarget}' missing")

                if con.type == 'IK':
                    chain = getattr(con, "chain_count", 0)
                    pole = getattr(con, "pole_target", None)
                    angle = math.degrees(getattr(con, "pole_angle", 0.0))
                    report_add(scene, "INFO", "IK", f"{pb.name}/{con.name}: chain={chain}, pole_angle={angle:.1f}°")

                    if pole is None:
                        warns += 1
                        report_add(scene, "WARN", "IK_NO_POLE", f"{pb.name}/{con.name}: missing pole target")
                    else:
                        head = arm.matrix_world @ pb.head
                        tail = arm.matrix_world @ pb.tail
                        chain_vec = tail - head
                        pole_loc = pole.matrix_world.translation
                        if chain_vec.length > 1e-6:
                            d = ((pole_loc - head).cross(chain_vec)).length / chain_vec.length
                            if d < pb.length * 0.15:
                                warns += 1
                                report_add(scene, "WARN", "IK_FLIP", f"{pb.name}/{con.name}: pole near limb plane (flip risk)")
                            if (pole_loc - head).dot(chain_vec) < 0.0:
                                warns += 1
                                report_add(scene, "WARN", "IK_BEHIND", f"{pb.name}/{con.name}: pole behind chain")

                if con.type == 'LIMIT_ROTATION':
                    has_limit = True
                    if con.owner_space == 'WORLD':
                        warns += 1
                        report_add(scene, "WARN", "LIMIT_WORLD", f"{pb.name}/{con.name}: owner space is WORLD")
                    axes = int(con.use_limit_x) + int(con.use_limit_y) + int(con.use_limit_z)
                    if joint == "HINGE" and axes != 1:
                        warns += 1
                        report_add(scene, "WARN", "HINGE_AXES", f"{pb.name}: hinge-like with {axes} limited axes")

                if pb.parent and con.type == 'COPY_ROTATION':
                    if getattr(con, "owner_space", None) != getattr(con, "target_space", None):
                        warns += 1
                        report_add(scene, "WARN", "PARENT_CONFLICT", f"{pb.name}/{con.name}: mixed spaces can fight parenting")

            if joint == "HINGE" and not has_limit:
                warns += 1
                report_add(scene, "WARN", "HINGE_NOLIMIT", f"{pb.name}: hinge-like but no Limit Rotation")

        # 9) keyframe warning
        if rig_has_pose_animation(arm):
            warns += 1
            report_add(scene, "WARN", "POSE_KEYS", "Pose keyframes exist; fixes can alter evaluated animation behavior")

        total = len(report_items(scene))
        scene.rd_summary = f"Diagnostics complete: {total} entries ({warns} warnings, {errs} errors)."
        print_report(scene)
        self.report({'INFO'}, scene.rd_summary)
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Fix operators
# -----------------------------------------------------------------------------

class RD_OT_apply_transforms_safe(bpy.types.Operator):
    bl_idname = "rig_doctor.apply_transforms_safe"
    bl_label = "Apply Armature Object Transforms (Safe)"

    @classmethod
    def poll(cls, context):
        return active_armature(context) is not None and context.mode == 'OBJECT'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        scene = context.scene
        arm = active_armature(context)
        if object_has_transform_animation(arm) and not scene.rd_force_apply:
            self.report({'WARNING'}, "Armature object has transform animation. Enable Force to proceed.")
            return {'CANCELLED'}
        try:
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            self.report({'INFO'}, "Applied armature rotation/scale.")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Apply transforms failed: {exc}")
            return {'CANCELLED'}


class RD_OT_normalize_spaces(bpy.types.Operator):
    bl_idname = "rig_doctor.normalize_spaces"
    bl_label = "Normalize Constraint Spaces"

    @classmethod
    def poll(cls, context):
        arm = active_armature(context)
        return arm is not None and arm.mode == 'POSE'

    def execute(self, context):
        scene = context.scene
        arm = active_armature(context)
        changed = 0
        for pb in pose_bones(arm, selected_only=True):
            for con in pb.constraints:
                if skip_keepworld(scene, con):
                    continue
                if con.type in {'COPY_ROTATION', 'COPY_TRANSFORMS', 'LIMIT_ROTATION'}:
                    if hasattr(con, "owner_space") and con.owner_space != 'LOCAL':
                        con.owner_space = 'LOCAL'
                        changed += 1
                    if con.type in {'COPY_ROTATION', 'COPY_TRANSFORMS'} and hasattr(con, "target_space") and con.target_space != 'LOCAL':
                        con.target_space = 'LOCAL'
                        changed += 1
        self.report({'INFO'}, f"Constraint spaces normalized: {changed} changes")
        return {'FINISHED'}


class RD_OT_fix_childof(bpy.types.Operator):
    bl_idname = "rig_doctor.fix_childof"
    bl_label = "Fix Child Of Inverses"

    @classmethod
    def poll(cls, context):
        arm = active_armature(context)
        return arm is not None and arm.mode == 'POSE'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        arm = active_armature(context)
        ok = 0
        fail = 0
        for pb in pose_bones(arm, selected_only=True):
            for con in pb.constraints:
                if con.type != 'CHILD_OF' or con.mute:
                    continue
                try:
                    arm.data.bones.active = pb.bone
                    with context.temp_override(object=arm, active_object=arm, pose_bone=pb, active_pose_bone=pb):
                        bpy.ops.constraint.childof_set_inverse(constraint=con.name, owner='BONE')
                    ok += 1
                except Exception:
                    fail += 1
        self.report({'INFO'}, f"Child Of inverse recalculated: {ok}, failed: {fail}")
        return {'FINISHED'}


class RD_OT_rebuild_ik_poles(bpy.types.Operator):
    bl_idname = "rig_doctor.rebuild_ik_poles"
    bl_label = "Rebuild IK Pole Targets"

    @classmethod
    def poll(cls, context):
        arm = active_armature(context)
        return arm is not None and arm.mode == 'POSE'

    def execute(self, context):
        scene = context.scene
        arm = active_armature(context)
        axis_map = {
            'X': Vector((1, 0, 0)),
            'Y': Vector((0, 1, 0)),
            'Z': Vector((0, 0, 1)),
            '-X': Vector((-1, 0, 0)),
            '-Y': Vector((0, -1, 0)),
            '-Z': Vector((0, 0, -1)),
        }
        created = 0

        for pb in pose_bones(arm, selected_only=True):
            for con in pb.constraints:
                if con.type != 'IK' or con.pole_target is not None:
                    continue

                side = ".R" if pb.name.endswith(".R") else ".L" if pb.name.endswith(".L") else ""
                name = f"{pb.name}_pole{side}"
                empty = bpy.data.objects.new(name, None)
                empty.empty_display_type = 'ARROWS'
                context.collection.objects.link(empty)

                head = arm.matrix_world @ pb.head
                tail = arm.matrix_world @ pb.tail
                chain = (tail - head)
                chain = chain.normalized() if chain.length > 1e-6 else Vector((0, 1, 0))
                forward = (arm.matrix_world.to_3x3() @ axis_map[scene.rd_pole_forward_axis]).normalized()
                pos = head + chain * (pb.length * 0.5) + forward * (pb.length * scene.rd_pole_distance)
                empty.matrix_world.translation = pos

                con.pole_target = empty
                con.pole_angle = 0.0
                created += 1

        self.report({'INFO'}, f"Created/assigned IK poles: {created}")
        return {'FINISHED'}


class RD_OT_snap_ctrl_to_def(bpy.types.Operator):
    bl_idname = "rig_doctor.snap_ctrl_to_def"
    bl_label = "Snap CTRL Bones to DEF Bones"

    @classmethod
    def poll(cls, context):
        arm = active_armature(context)
        return arm is not None and arm.mode == 'POSE'

    def execute(self, context):
        arm = active_armature(context)
        count = 0
        for pb in pose_bones(arm, selected_only=True):
            target = arm.pose.bones.get(mapped_name(pb.name, "CTRL", "DEF"))
            if target:
                pb.matrix = target.matrix.copy()
                count += 1
        context.view_layer.update()
        self.report({'INFO'}, f"Snapped CTRL->DEF: {count}")
        return {'FINISHED'}


class RD_OT_snap_def_to_ctrl(bpy.types.Operator):
    bl_idname = "rig_doctor.snap_def_to_ctrl"
    bl_label = "Snap DEF Bones to CTRL Bones"

    @classmethod
    def poll(cls, context):
        arm = active_armature(context)
        return arm is not None and arm.mode == 'POSE'

    def execute(self, context):
        arm = active_armature(context)
        count = 0
        for pb in pose_bones(arm, selected_only=True):
            target = arm.pose.bones.get(mapped_name(pb.name, "DEF", "CTRL"))
            if target:
                pb.matrix = target.matrix.copy()
                count += 1
        context.view_layer.update()
        self.report({'INFO'}, f"Snapped DEF->CTRL: {count}")
        return {'FINISHED'}


class RD_OT_toggle_deform_flags(bpy.types.Operator):
    bl_idname = "rig_doctor.toggle_deform_flags"
    bl_label = "Toggle Deform Flags by Naming"

    @classmethod
    def poll(cls, context):
        return active_armature(context) is not None

    def execute(self, context):
        scene = context.scene
        arm = active_armature(context)
        bones = arm.data.bones if scene.rd_affect_all else [b for b in arm.data.bones if b.select]
        changed = 0

        for b in bones:
            n = b.name.upper()
            if "CTRL" in n and b.use_deform:
                b.use_deform = False
                changed += 1
            elif "DEF" in n and not b.use_deform:
                b.use_deform = True
                changed += 1

        self.report({'INFO'}, f"Updated deform flags: {changed}")
        return {'FINISHED'}


class RD_OT_generate_limits(bpy.types.Operator):
    bl_idname = "rig_doctor.generate_limits"
    bl_label = "Generate/Repair Limit Rotations"

    @classmethod
    def poll(cls, context):
        arm = active_armature(context)
        return arm is not None and arm.mode == 'POSE'

    def execute(self, context):
        scene = context.scene
        arm = active_armature(context)
        changed, skipped = 0, 0

        for pb in arm.pose.bones:
            cls = classify_joint(pb.name)
            if cls == "NONE":
                continue

            existing = [c for c in pb.constraints if c.type == 'LIMIT_ROTATION']
            if existing and not scene.rd_replace_limits:
                skipped += 1
                continue
            con = existing[0] if existing else pb.constraints.new(type='LIMIT_ROTATION')
            set_limit_for_joint(con, pb.name, pb)
            changed += 1

        self.report({'INFO'}, f"Limit constraints updated: {changed}, skipped: {skipped}")
        return {'FINISHED'}


class RD_OT_bake_pose_to_rest(bpy.types.Operator):
    bl_idname = "rig_doctor.bake_pose_to_rest"
    bl_label = "Bake Pose to Rest Pose (Dangerous Tool)"

    @classmethod
    def poll(cls, context):
        arm = active_armature(context)
        return arm is not None and arm.mode == 'POSE'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        scene = context.scene
        try:
            bpy.ops.pose.armature_apply(selected=scene.rd_bake_selected_only)
            self.report({'WARNING'}, "Applied pose to rest pose. This can break animation; verify rig immediately.")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Bake pose to rest failed: {exc}")
            return {'CANCELLED'}


# -----------------------------------------------------------------------------
# Quality of life
# -----------------------------------------------------------------------------

class RD_OT_duplicate_sandbox(bpy.types.Operator):
    bl_idname = "rig_doctor.duplicate_sandbox"
    bl_label = "Duplicate Armature (Safe Sandbox)"

    @classmethod
    def poll(cls, context):
        return active_armature(context) is not None

    def execute(self, context):
        src = active_armature(context)
        try:
            dup = src.copy()
            dup.data = src.data.copy()
            dup.name = f"{src.name}_SANDBOX"
            dup.data.name = f"{src.data.name}_SANDBOX"
            context.collection.objects.link(dup)
            dup.matrix_world = src.matrix_world.copy()

            for obj in context.scene.objects:
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object == src:
                        new_mod = obj.modifiers.new(name=f"{mod.name}_SANDBOX", type='ARMATURE')
                        new_mod.object = dup

            self.report({'INFO'}, f"Created sandbox rig: {dup.name}")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Sandbox duplicate failed: {exc}")
            return {'CANCELLED'}


class RD_OT_create_debug_view(bpy.types.Operator):
    bl_idname = "rig_doctor.create_debug_view"
    bl_label = "Create Rig Debug View"

    @classmethod
    def poll(cls, context):
        return active_armature(context) is not None

    def execute(self, context):
        arm = active_armature(context)
        try:
            arm.show_in_front = True
            arm.data.show_axes = True
            arm.data.show_names = True
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type != 'VIEW_3D':
                        continue
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.overlay.show_relationship_lines = True
                            space.overlay.show_bones = True
            self.report({'INFO'}, "Rig debug view enabled")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Debug view setup failed: {exc}")
            return {'CANCELLED'}


# -----------------------------------------------------------------------------
# Panel
# -----------------------------------------------------------------------------

class RD_PT_panel(bpy.types.Panel):
    bl_label = "Rig Doctor"
    bl_idname = "RD_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Rig Tools"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        arm = active_armature(context)

        box = layout.box()
        box.label(text="Diagnose")
        row = box.row()
        row.enabled = arm is not None
        row.operator("rig_doctor.run_diagnostics", icon='VIEWZOOM')
        box.label(text=scene.rd_summary)

        items = report_items(scene)
        for i in items[:10]:
            level = i.get("level", "INFO")
            icon = 'INFO' if level == 'INFO' else 'ERROR' if level == 'WARN' else 'CANCEL'
            box.label(text=f"{i.get('code', '')}: {i.get('text', '')}", icon=icon)
        if len(items) > 10:
            box.label(text="...")

        fix = layout.box()
        fix.label(text="Fix")

        row = fix.row()
        row.enabled = arm is not None and context.mode == 'OBJECT'
        row.operator("rig_doctor.apply_transforms_safe", icon='FILE_TICK')
        fix.prop(scene, "rd_force_apply")

        row = fix.row()
        row.enabled = arm is not None and arm.mode == 'POSE'
        row.operator("rig_doctor.normalize_spaces", icon='CONSTRAINT')
        fix.prop(scene, "rd_skip_keepworld")

        row = fix.row()
        row.enabled = arm is not None and arm.mode == 'POSE'
        row.operator("rig_doctor.fix_childof", icon='CON_CHILDOF')

        row = fix.row()
        row.enabled = arm is not None and arm.mode == 'POSE'
        row.operator("rig_doctor.rebuild_ik_poles", icon='CON_KINEMATIC')
        fix.prop(scene, "rd_pole_distance")
        fix.prop(scene, "rd_pole_forward_axis")

        row = fix.row(align=True)
        row.enabled = arm is not None and arm.mode == 'POSE'
        row.operator("rig_doctor.snap_ctrl_to_def", icon='SNAP_ON')
        row.operator("rig_doctor.snap_def_to_ctrl", icon='SNAP_ON')

        row = fix.row()
        row.enabled = arm is not None
        row.operator("rig_doctor.toggle_deform_flags", icon='BONE_DATA')
        fix.prop(scene, "rd_affect_all")

        row = fix.row()
        row.enabled = arm is not None and arm.mode == 'POSE'
        row.operator("rig_doctor.generate_limits", icon='CON_ROTLIKE')
        fix.prop(scene, "rd_replace_limits")

        danger = layout.box()
        danger.label(text="Danger Zone")
        row = danger.row()
        row.enabled = arm is not None and arm.mode == 'POSE'
        row.operator("rig_doctor.bake_pose_to_rest", icon='ERROR')
        danger.prop(scene, "rd_bake_selected_only")

        qol = layout.box()
        qol.label(text="Quality of Life")
        row = qol.row()
        row.enabled = arm is not None
        row.operator("rig_doctor.duplicate_sandbox", icon='DUPLICATE')
        row = qol.row()
        row.enabled = arm is not None
        row.operator("rig_doctor.create_debug_view", icon='HIDE_OFF')


CLASSES = (
    RD_OT_run_diagnostics,
    RD_OT_apply_transforms_safe,
    RD_OT_normalize_spaces,
    RD_OT_fix_childof,
    RD_OT_rebuild_ik_poles,
    RD_OT_snap_ctrl_to_def,
    RD_OT_snap_def_to_ctrl,
    RD_OT_toggle_deform_flags,
    RD_OT_generate_limits,
    RD_OT_bake_pose_to_rest,
    RD_OT_duplicate_sandbox,
    RD_OT_create_debug_view,
    RD_PT_panel,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)

    bpy.types.Scene.rd_summary = StringProperty(default="No diagnostics run yet.")
    bpy.types.Scene.rd_force_apply = BoolProperty(name="Force apply on animated armature object", default=False)
    bpy.types.Scene.rd_skip_keepworld = BoolProperty(name="Skip KEEPWORLD constraints", default=True)
    bpy.types.Scene.rd_pole_distance = FloatProperty(name="Pole Distance Factor", default=1.5, min=0.2, max=5.0)
    bpy.types.Scene.rd_pole_forward_axis = EnumProperty(
        name="Pole Forward Axis",
        items=[('X', 'X', ''), ('Y', 'Y', ''), ('Z', 'Z', ''), ('-X', '-X', ''), ('-Y', '-Y', ''), ('-Z', '-Z', '')],
        default='Z',
    )
    bpy.types.Scene.rd_affect_all = BoolProperty(name="Affect All Bones", default=False)
    bpy.types.Scene.rd_replace_limits = BoolProperty(name="Replace Existing Limits", default=False)
    bpy.types.Scene.rd_bake_selected_only = BoolProperty(name="Bake Selected Bones Only", default=True)

    if "rd_items" not in bpy.context.scene:
        bpy.context.scene["rd_items"] = "[]"


def unregister():
    del bpy.types.Scene.rd_bake_selected_only
    del bpy.types.Scene.rd_replace_limits
    del bpy.types.Scene.rd_affect_all
    del bpy.types.Scene.rd_pole_forward_axis
    del bpy.types.Scene.rd_pole_distance
    del bpy.types.Scene.rd_skip_keepworld
    del bpy.types.Scene.rd_force_apply
    del bpy.types.Scene.rd_summary

    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
