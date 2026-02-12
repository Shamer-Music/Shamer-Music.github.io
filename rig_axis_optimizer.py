bl_info = {
    "name": "Rig Doctor",
    "author": "OpenAI",
    "version": (2, 0, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > Rig Tools",
    "description": "Diagnose and repair common rig issues causing pose mismatch, IK flips, and constraint instability",
    "category": "Rigging",
}

import json
import math
import bpy
from mathutils import Vector
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty


HINGE_NAMES = ("elbow", "knee", "finger", "toe")
SPHERICAL_NAMES = ("wrist", "shoulder", "hip", "neck")


# -----------------------------------------------------------------------------
# Report helpers
# -----------------------------------------------------------------------------

def _add_report(scene, severity, code, message):
    payload = scene.get("rd_report_items", "[]")
    try:
        items = json.loads(payload)
    except Exception:
        items = []
    items.append({"severity": severity, "code": code, "message": message})
    scene["rd_report_items"] = json.dumps(items)


def _clear_report(scene):
    scene["rd_report_items"] = "[]"
    scene.rd_report_summary = "No diagnostics run yet."


def _iter_report(scene):
    payload = scene.get("rd_report_items", "[]")
    try:
        items = json.loads(payload)
        if isinstance(items, list):
            return items
    except Exception:
        pass
    return []


def _print_console_report(scene):
    print("\n=== Rig Doctor Diagnostics ===")
    print(scene.rd_report_summary)
    for item in _iter_report(scene):
        print(f"[{item.get('severity', 'INFO')}] {item.get('code', 'GEN')}: {item.get('message', '')}")


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def _is_armature_active(context):
    obj = context.active_object
    return obj is not None and obj.type == 'ARMATURE'


def _classify_joint(name):
    n = name.lower()
    if any(k in n for k in HINGE_NAMES):
        return "HINGE"
    if any(k in n for k in SPHERICAL_NAMES):
        return "SPHERICAL"
    return "NONE"


def _pbones(obj, selected_only=False):
    if selected_only:
        return [pb for pb in obj.pose.bones if pb.bone.select]
    return list(obj.pose.bones)


def _bone_primary_axis(pbone):
    d = pbone.bone.tail_local - pbone.bone.head_local
    if d.length < 1e-8:
        return "Y"
    v = d.normalized()
    abs_vals = [abs(v.x), abs(v.y), abs(v.z)]
    return ("X", "Y", "Z")[abs_vals.index(max(abs_vals))]


def _axis_flags(axis):
    return {
        "X": (True, False, False),
        "Y": (False, True, False),
        "Z": (False, False, True),
    }[axis]


def _matrix_to_flat(m):
    return [m[r][c] for r in range(4) for c in range(4)]


def _flat_to_matrix(vals):
    if not isinstance(vals, (list, tuple)) or len(vals) != 16:
        return None
    try:
        return bpy.types.Matrix(((
            vals[0], vals[1], vals[2], vals[3]),
            (vals[4], vals[5], vals[6], vals[7]),
            (vals[8], vals[9], vals[10], vals[11]),
            (vals[12], vals[13], vals[14], vals[15]),
        ))
    except Exception:
        from mathutils import Matrix
        try:
            return Matrix((
                vals[0:4], vals[4:8], vals[8:12], vals[12:16]
            ))
        except Exception:
            return None


def _has_object_anim(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return False
    for fc in ad.action.fcurves:
        if fc.data_path.startswith(("location", "rotation", "scale")):
            return True
    return False


def _has_pose_keys(arm_obj):
    ad = arm_obj.animation_data
    if not ad or not ad.action:
        return False
    for fc in ad.action.fcurves:
        if fc.data_path.startswith("pose.bones["):
            return True
    return False


def _find_matching_name(name, from_tag, to_tag):
    out = name
    for token in (f"{from_tag}_", f".{from_tag}", from_tag):
        if token in out:
            out = out.replace(token, token.replace(from_tag, to_tag))
    if out == name and name.startswith(from_tag):
        out = name.replace(from_tag, to_tag, 1)
    return out


def _is_world_space_constraint(con):
    owner = getattr(con, "owner_space", None)
    target = getattr(con, "target_space", None)
    return owner == 'WORLD' or target == 'WORLD'


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

class RD_OT_run_diagnostics(bpy.types.Operator):
    bl_idname = "rig_doctor.run_diagnostics"
    bl_label = "Rig Doctor: Run Full Diagnostics"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context)

    def execute(self, context):
        scene = context.scene
        arm = context.active_object
        _clear_report(scene)

        warnings = 0
        errors = 0

        # 1) Mode/Transform Sanity
        s = arm.scale
        r = arm.rotation_euler
        if any(abs(v - 1.0) > 1e-4 for v in s):
            warnings += 1
            _add_report(scene, "WARN", "XFORM_SCALE", f"Armature scale is {tuple(round(v, 4) for v in s)} (expected 1,1,1).")
        if any(abs(v) > 1e-4 for v in r):
            warnings += 1
            _add_report(scene, "WARN", "XFORM_ROT", "Armature object rotation is non-zero; unapplied transforms can cause rig instability.")

        # 2) Bone Roll Consistency
        original_mode = arm.mode
        try:
            if arm.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            for eb in arm.data.edit_bones:
                if not eb.parent:
                    continue
                delta = abs(eb.roll - eb.parent.roll)
                if delta > math.radians(25.0):
                    warnings += 1
                    _add_report(scene, "WARN", "ROLL", f"{eb.name}: roll delta to parent is {math.degrees(delta):.1f}°.")
        except Exception as exc:
            errors += 1
            _add_report(scene, "ERROR", "ROLL_SCAN", f"Could not scan roll consistency: {exc}")
        finally:
            try:
                if arm.mode != original_mode:
                    bpy.ops.object.mode_set(mode=original_mode)
            except Exception:
                pass

        # 3/4/5/6/7/8) constraint, IK, naming, limit, parent/child
        for pb in arm.pose.bones:
            name_lower = pb.name.lower()
            joint_class = _classify_joint(pb.name)
            has_limit = False

            # Deform vs control audit
            if "ctrl" in name_lower and pb.bone.use_deform:
                warnings += 1
                _add_report(scene, "WARN", "DEFORM_CTRL", f"{pb.name} looks like a control bone but use_deform=True.")
            if "def" in name_lower and pb.constraints:
                _add_report(scene, "INFO", "DEFORM_HAS_CON", f"{pb.name} is DEF-like and has constraints; confirm this is intentional.")

            for con in pb.constraints:
                # 3) constraint space audit
                owner = getattr(con, "owner_space", "N/A")
                target = getattr(con, "target_space", "N/A")
                _add_report(scene, "INFO", "CON_SPACE", f"{pb.name}/{con.name}: owner={owner}, target={target}")
                if _is_world_space_constraint(con):
                    warnings += 1
                    _add_report(scene, "WARN", "WORLD_SPACE", f"{pb.name}/{con.name} uses WORLD space; likely to cause jumps.")

                # 4) missing/invalid targets
                if hasattr(con, "target") and con.target is None and con.type not in {'LIMIT_ROTATION'}:
                    errors += 1
                    _add_report(scene, "ERROR", "TARGET_MISSING", f"{pb.name}/{con.name} has no target object.")

                if hasattr(con, "subtarget") and con.subtarget:
                    tgt_obj = getattr(con, "target", None)
                    if tgt_obj and tgt_obj.type == 'ARMATURE' and con.subtarget not in tgt_obj.data.bones:
                        errors += 1
                        _add_report(scene, "ERROR", "SUBTARGET_INVALID", f"{pb.name}/{con.name} subtarget '{con.subtarget}' does not exist.")

                # 5) IK audit
                if con.type == 'IK':
                    chain_len = getattr(con, "chain_count", 0)
                    pole = getattr(con, "pole_target", None)
                    pole_angle = getattr(con, "pole_angle", 0.0)
                    _add_report(scene, "INFO", "IK", f"{pb.name}/{con.name}: chain={chain_len}, pole_angle={math.degrees(pole_angle):.1f}°")
                    if pole is None:
                        warnings += 1
                        _add_report(scene, "WARN", "IK_POLE", f"{pb.name}/{con.name} has no pole target.")
                    else:
                        head = arm.matrix_world @ pb.head
                        tail = arm.matrix_world @ pb.tail
                        line = tail - head
                        pole_loc = pole.matrix_world.translation
                        if line.length > 1e-6:
                            dist = ((pole_loc - head).cross(line)).length / line.length
                            if dist < (pb.length * 0.15):
                                warnings += 1
                                _add_report(scene, "WARN", "IK_FLIP", f"{pb.name}/{con.name}: pole appears too close to chain plane (likely flip).")
                            if (pole_loc - head).dot(line) < 0.0:
                                warnings += 1
                                _add_report(scene, "WARN", "IK_BEHIND", f"{pb.name}/{con.name}: pole appears behind chain.")

                # 7) Limit rotation audit
                if con.type == 'LIMIT_ROTATION':
                    has_limit = True
                    if con.owner_space == 'WORLD':
                        warnings += 1
                        _add_report(scene, "WARN", "LIMIT_WORLD", f"{pb.name}/{con.name} uses WORLD owner space.")
                    axes = int(con.use_limit_x) + int(con.use_limit_y) + int(con.use_limit_z)
                    if joint_class == 'HINGE' and axes != 1:
                        warnings += 1
                        _add_report(scene, "WARN", "HINGE_AXES", f"{pb.name} hinge-like but has {axes} limited axes.")

                # 8) Parent/child inconsistency heuristic
                if pb.parent and con.type == 'COPY_ROTATION' and getattr(con, "owner_space", None) != getattr(con, "target_space", None):
                    warnings += 1
                    _add_report(scene, "WARN", "PARENT_FIGHT", f"{pb.name}/{con.name} may fight parent due to mixed spaces.")

            if joint_class == 'HINGE' and not has_limit:
                warnings += 1
                _add_report(scene, "WARN", "HINGE_NO_LIMIT", f"{pb.name} looks hinge-like and has no Limit Rotation.")

        # 9) Drivers/keyframes warning
        if _has_pose_keys(arm):
            warnings += 1
            _add_report(scene, "WARN", "KEYS", "Pose keyframes detected. Running fixes may change evaluated results.")

        count = len(_iter_report(scene))
        scene.rd_report_summary = f"Diagnostics complete: {count} entries ({warnings} warnings, {errors} errors)."
        _print_console_report(scene)
        self.report({'INFO'}, scene.rd_report_summary)
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Fix operators
# -----------------------------------------------------------------------------

class RD_OT_apply_armature_transforms_safe(bpy.types.Operator):
    bl_idname = "rig_doctor.apply_armature_transforms_safe"
    bl_label = "Apply Armature Object Transforms (Safe)"
    bl_description = "Apply armature rotation/scale only when safe"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context) and context.mode == 'OBJECT'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        arm = context.active_object
        scene = context.scene
        if _has_object_anim(arm) and not scene.rd_force_apply_transforms:
            self.report({'WARNING'}, "Armature object has animation. Enable Force to proceed.")
            return {'CANCELLED'}
        try:
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            self.report({'INFO'}, "Applied armature rotation/scale.")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Could not apply transforms: {exc}")
            return {'CANCELLED'}


class RD_OT_normalize_constraint_spaces(bpy.types.Operator):
    bl_idname = "rig_doctor.normalize_constraint_spaces"
    bl_label = "Normalize Constraint Spaces"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        scene = context.scene
        arm = context.active_object
        changed = 0
        for pb in _pbones(arm, selected_only=True):
            for con in pb.constraints:
                if scene.rd_skip_keepworld and "KEEPWORLD" in con.name.upper():
                    continue
                if con.type in {'COPY_ROTATION', 'COPY_TRANSFORMS', 'LIMIT_ROTATION'}:
                    if hasattr(con, "owner_space") and con.owner_space != 'LOCAL':
                        con.owner_space = 'LOCAL'
                        changed += 1
                    if con.type in {'COPY_ROTATION', 'COPY_TRANSFORMS'} and hasattr(con, "target_space") and con.target_space != 'LOCAL':
                        con.target_space = 'LOCAL'
                        changed += 1
        self.report({'INFO'}, f"Normalized constraint spaces: {changed} changes.")
        return {'FINISHED'}


class RD_OT_fix_childof_inverses(bpy.types.Operator):
    bl_idname = "rig_doctor.fix_childof_inverses"
    bl_label = "Fix Child Of Inverses"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context) and context.active_object.mode == 'POSE'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        arm = context.active_object
        done = 0
        fail = 0
        for pb in _pbones(arm, selected_only=True):
            for con in pb.constraints:
                if con.type != 'CHILD_OF' or con.mute:
                    continue
                try:
                    arm.data.bones.active = pb.bone
                    with context.temp_override(object=arm, active_object=arm, pose_bone=pb, active_pose_bone=pb):
                        bpy.ops.constraint.childof_set_inverse(constraint=con.name, owner='BONE')
                    done += 1
                except Exception:
                    fail += 1
        self.report({'INFO'}, f"Child Of inverses updated: {done}, failed: {fail}")
        return {'FINISHED'}


class RD_OT_rebuild_ik_poles(bpy.types.Operator):
    bl_idname = "rig_doctor.rebuild_ik_poles"
    bl_label = "Rebuild IK Pole Targets"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        scene = context.scene
        arm = context.active_object
        created = 0
        assigned = 0

        axis_map = {
            'X': Vector((1, 0, 0)),
            'Y': Vector((0, 1, 0)),
            'Z': Vector((0, 0, 1)),
            '-X': Vector((-1, 0, 0)),
            '-Y': Vector((0, -1, 0)),
            '-Z': Vector((0, 0, -1)),
        }

        for pb in _pbones(arm, selected_only=True):
            for con in pb.constraints:
                if con.type != 'IK':
                    continue
                if con.pole_target is not None:
                    continue

                axis = axis_map[scene.rd_pole_forward_axis]
                side = ".R" if pb.name.endswith(".R") else ".L" if pb.name.endswith(".L") else ""
                pole_name = f"{pb.name}_pole{side}"

                empty = bpy.data.objects.new(pole_name, None)
                empty.empty_display_type = 'ARROWS'
                context.collection.objects.link(empty)

                head_w = arm.matrix_world @ pb.head
                tail_w = arm.matrix_world @ pb.tail
                chain_dir = (tail_w - head_w).normalized() if (tail_w - head_w).length > 1e-6 else Vector((0, 1, 0))
                side_dir = (arm.matrix_world.to_3x3() @ axis).normalized()
                pole_pos = head_w + chain_dir * (pb.length * 0.5) + side_dir * (pb.length * scene.rd_pole_distance_factor)
                empty.matrix_world.translation = pole_pos

                con.pole_target = empty
                con.pole_angle = 0.0
                created += 1
                assigned += 1

        self.report({'INFO'}, f"IK poles created: {created}, assigned: {assigned}")
        return {'FINISHED'}


class RD_OT_snap_ctrl_to_def(bpy.types.Operator):
    bl_idname = "rig_doctor.snap_ctrl_to_def"
    bl_label = "Snap CTRL -> DEF"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        arm = context.active_object
        moved = 0
        for pb in _pbones(arm, selected_only=True):
            target_name = _find_matching_name(pb.name, "CTRL", "DEF")
            target = arm.pose.bones.get(target_name)
            if target:
                pb.matrix = target.matrix.copy()
                moved += 1
        context.view_layer.update()
        self.report({'INFO'}, f"Snapped CTRL bones: {moved}")
        return {'FINISHED'}


class RD_OT_snap_def_to_ctrl(bpy.types.Operator):
    bl_idname = "rig_doctor.snap_def_to_ctrl"
    bl_label = "Snap DEF -> CTRL"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        arm = context.active_object
        moved = 0
        for pb in _pbones(arm, selected_only=True):
            target_name = _find_matching_name(pb.name, "DEF", "CTRL")
            target = arm.pose.bones.get(target_name)
            if target:
                pb.matrix = target.matrix.copy()
                moved += 1
        context.view_layer.update()
        self.report({'INFO'}, f"Snapped DEF bones: {moved}")
        return {'FINISHED'}


class RD_OT_toggle_deform_by_naming(bpy.types.Operator):
    bl_idname = "rig_doctor.toggle_deform_by_naming"
    bl_label = "Toggle Deform Flags by Naming"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context)

    def execute(self, context):
        scene = context.scene
        arm = context.active_object
        changed = 0
        bones = arm.data.bones if scene.rd_affect_all_bones else [b for b in arm.data.bones if b.select]
        for b in bones:
            n = b.name.upper()
            if "CTRL" in n and b.use_deform:
                b.use_deform = False
                changed += 1
            elif "DEF" in n and not b.use_deform:
                b.use_deform = True
                changed += 1
        self.report({'INFO'}, f"Updated deform flags on {changed} bones.")
        return {'FINISHED'}


class RD_OT_generate_repair_limits(bpy.types.Operator):
    bl_idname = "rig_doctor.generate_repair_limits"
    bl_label = "Generate/Repair Limit Rotations"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context) and context.active_object.mode == 'POSE'

    def execute(self, context):
        scene = context.scene
        arm = context.active_object
        changed = 0
        skipped = 0

        presets = {
            "elbow": (0.0, 135.0, "HINGE"),
            "knee": (0.0, 140.0, "HINGE"),
            "finger": (0.0, 90.0, "HINGE"),
            "wrist": (-50.0, 50.0, "SPHERICAL"),
            "shoulder": (-95.0, 95.0, "SPHERICAL"),
            "hip": (-95.0, 95.0, "SPHERICAL"),
            "neck": (-60.0, 60.0, "SPHERICAL"),
        }

        for pb in arm.pose.bones:
            n = pb.name.lower()
            preset_key = next((k for k in presets if k in n), None)
            if not preset_key:
                continue

            existing = [c for c in pb.constraints if c.type == 'LIMIT_ROTATION']
            if existing and not scene.rd_replace_limits:
                skipped += 1
                continue
            con = existing[0] if existing else pb.constraints.new(type='LIMIT_ROTATION')

            lo_deg, hi_deg, kind = presets[preset_key]
            con.owner_space = 'LOCAL'
            con.use_transform_limit = True

            if kind == "HINGE":
                axis = _bone_primary_axis(pb)
                fx, fy, fz = _axis_flags(axis)
                con.use_limit_x, con.use_limit_y, con.use_limit_z = fx, fy, fz
                lo, hi = math.radians(lo_deg), math.radians(hi_deg)
                con.min_x, con.max_x = (lo, hi) if fx else (-math.pi, math.pi)
                con.min_y, con.max_y = (lo, hi) if fy else (-math.pi, math.pi)
                con.min_z, con.max_z = (lo, hi) if fz else (-math.pi, math.pi)
            else:
                twist_axis = _bone_primary_axis(pb)
                con.use_limit_x = con.use_limit_y = con.use_limit_z = True
                twist = math.radians(45.0)
                swing = math.radians(abs(hi_deg))
                con.min_x, con.max_x = (-twist, twist) if twist_axis == 'X' else (-swing, swing)
                con.min_y, con.max_y = (-twist, twist) if twist_axis == 'Y' else (-swing, swing)
                con.min_z, con.max_z = (-twist, twist) if twist_axis == 'Z' else (-swing, swing)

            changed += 1

        self.report({'INFO'}, f"Limit rotations generated/repaired: {changed}, skipped: {skipped}")
        return {'FINISHED'}


class RD_OT_bake_pose_to_rest(bpy.types.Operator):
    bl_idname = "rig_doctor.bake_pose_to_rest"
    bl_label = "Bake Pose to Rest Pose (Dangerous Tool)"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context) and context.active_object.mode == 'POSE'

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        scene = context.scene
        try:
            bpy.ops.pose.armature_apply(selected=scene.rd_bake_selected_only)
            self.report({'WARNING'}, "Applied pose to rest pose. Verify rig and animation immediately.")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Bake pose to rest failed: {exc}")
            return {'CANCELLED'}


class RD_OT_duplicate_sandbox(bpy.types.Operator):
    bl_idname = "rig_doctor.duplicate_sandbox"
    bl_label = "Duplicate Armature (Safe Sandbox)"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context)

    def execute(self, context):
        src = context.active_object
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
            self.report({'ERROR'}, f"Sandbox duplication failed: {exc}")
            return {'CANCELLED'}


class RD_OT_create_debug_view(bpy.types.Operator):
    bl_idname = "rig_doctor.create_debug_view"
    bl_label = "Create Rig Debug View"

    @classmethod
    def poll(cls, context):
        return _is_armature_active(context)

    def execute(self, context):
        arm = context.active_object
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
            self.report({'INFO'}, "Rig debug view enabled (axes, names, in-front, relationship lines).")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Debug view setup failed: {exc}")
            return {'CANCELLED'}


# -----------------------------------------------------------------------------
# Panel
# -----------------------------------------------------------------------------

class RD_PT_main(bpy.types.Panel):
    bl_label = "Rig Doctor"
    bl_idname = "RD_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Rig Tools"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.active_object
        is_arm = _is_armature_active(context)

        diag = layout.box()
        diag.label(text="Diagnose")
        row = diag.row()
        row.enabled = is_arm
        row.operator("rig_doctor.run_diagnostics", icon='VIEWZOOM')
        diag.label(text=scene.rd_report_summary)

        report_items = _iter_report(scene)
        for item in report_items[:10]:
            sev = item.get("severity", "INFO")
            icon = 'INFO'
            if sev == 'WARN':
                icon = 'ERROR'
            elif sev == 'ERROR':
                icon = 'CANCEL'
            diag.label(text=f"{item.get('code', '')}: {item.get('message', '')}", icon=icon)
        if len(report_items) > 10:
            diag.label(text="...")

        fix = layout.box()
        fix.label(text="Fix")

        row = fix.row()
        row.enabled = is_arm and obj.mode == 'OBJECT'
        row.operator("rig_doctor.apply_armature_transforms_safe", icon='FILE_TICK')
        fix.prop(scene, "rd_force_apply_transforms", text="Force (if armature object has animation)")

        row = fix.row()
        row.enabled = is_arm and obj.mode == 'POSE'
        row.operator("rig_doctor.normalize_constraint_spaces", icon='CONSTRAINT')
        fix.prop(scene, "rd_skip_keepworld", text="Skip constraints containing KEEPWORLD")

        row = fix.row()
        row.enabled = is_arm and obj.mode == 'POSE'
        row.operator("rig_doctor.fix_childof_inverses", icon='CON_CHILDOF')

        row = fix.row()
        row.enabled = is_arm and obj.mode == 'POSE'
        row.operator("rig_doctor.rebuild_ik_poles", icon='CON_KINEMATIC')
        fix.prop(scene, "rd_pole_distance_factor")
        fix.prop(scene, "rd_pole_forward_axis")

        row = fix.row(align=True)
        row.enabled = is_arm and obj.mode == 'POSE'
        row.operator("rig_doctor.snap_ctrl_to_def", icon='SNAP_ON')
        row.operator("rig_doctor.snap_def_to_ctrl", icon='SNAP_ON')

        row = fix.row()
        row.enabled = is_arm
        row.operator("rig_doctor.toggle_deform_by_naming", icon='BONE_DATA')
        fix.prop(scene, "rd_affect_all_bones")

        row = fix.row()
        row.enabled = is_arm and obj.mode == 'POSE'
        row.operator("rig_doctor.generate_repair_limits", icon='CON_ROTLIKE')
        fix.prop(scene, "rd_replace_limits")

        danger = layout.box()
        danger.label(text="Danger Zone")
        row = danger.row()
        row.enabled = is_arm and obj.mode == 'POSE'
        row.operator("rig_doctor.bake_pose_to_rest", icon='ERROR')
        danger.prop(scene, "rd_bake_selected_only")

        qol = layout.box()
        qol.label(text="Quality of Life")
        row = qol.row()
        row.enabled = is_arm
        row.operator("rig_doctor.duplicate_sandbox", icon='DUPLICATE')
        row = qol.row()
        row.enabled = is_arm
        row.operator("rig_doctor.create_debug_view", icon='HIDE_OFF')


classes = (
    RD_OT_run_diagnostics,
    RD_OT_apply_armature_transforms_safe,
    RD_OT_normalize_constraint_spaces,
    RD_OT_fix_childof_inverses,
    RD_OT_rebuild_ik_poles,
    RD_OT_snap_ctrl_to_def,
    RD_OT_snap_def_to_ctrl,
    RD_OT_toggle_deform_by_naming,
    RD_OT_generate_repair_limits,
    RD_OT_bake_pose_to_rest,
    RD_OT_duplicate_sandbox,
    RD_OT_create_debug_view,
    RD_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.rd_report_summary = StringProperty(default="No diagnostics run yet.")
    bpy.types.Scene.rd_force_apply_transforms = BoolProperty(default=False)
    bpy.types.Scene.rd_skip_keepworld = BoolProperty(default=True)
    bpy.types.Scene.rd_pole_distance_factor = FloatProperty(default=1.5, min=0.2, max=5.0)
    bpy.types.Scene.rd_pole_forward_axis = EnumProperty(
        items=[('X', 'X', ''), ('Y', 'Y', ''), ('Z', 'Z', ''), ('-X', '-X', ''), ('-Y', '-Y', ''), ('-Z', '-Z', '')],
        default='Z',
    )
    bpy.types.Scene.rd_affect_all_bones = BoolProperty(default=False)
    bpy.types.Scene.rd_replace_limits = BoolProperty(default=False)
    bpy.types.Scene.rd_bake_selected_only = BoolProperty(default=True)

    if "rd_report_items" not in bpy.context.scene:
        bpy.context.scene["rd_report_items"] = "[]"


def unregister():
    del bpy.types.Scene.rd_bake_selected_only
    del bpy.types.Scene.rd_replace_limits
    del bpy.types.Scene.rd_affect_all_bones
    del bpy.types.Scene.rd_pole_forward_axis
    del bpy.types.Scene.rd_pole_distance_factor
    del bpy.types.Scene.rd_skip_keepworld
    del bpy.types.Scene.rd_force_apply_transforms
    del bpy.types.Scene.rd_report_summary

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
