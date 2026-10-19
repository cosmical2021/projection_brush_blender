bl_info = {
    "name": "Real SpotLight Shader Projector & Brush Painter",
    "author": "Gemini Collaboration",
    "version": (3, 2, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar (N) > SpotLight Paint",
    "description": "Projects textures using real SPOT lamps with live drivers, bilinear smooth raycast brush, and one-click bake",
    "category": "Paint",
}

import bpy
import gpu
import math
import numpy as np
from bpy.app.handlers import persistent
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.interpolate import poly_3d_calc
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils

_is_updating_nodes = False


def get_builtin_shader(shader_type='UNIFORM_COLOR'):
    """Safe shader loader for Blender 4.x / 5.2+."""
    try:
        return gpu.shader.from_builtin(shader_type)
    except Exception:
        return gpu.shader.from_builtin('3D_UNIFORM_COLOR')


def add_light_prop_driver(target_socket_or_node, prop_path, array_index, light_data, data_path, expression):
    """Creates a real-time Blender Driver linking a Spot Light property to a Shader Node input socket."""
    try:
        if array_index >= 0:
            target_socket_or_node.driver_remove(prop_path, array_index)
            fcurve = target_socket_or_node.driver_add(prop_path, array_index)
        else:
            target_socket_or_node.driver_remove(prop_path)
            fcurve = target_socket_or_node.driver_add(prop_path)

        drv = fcurve.driver
        drv.type = 'SCRIPTED'
        var = drv.variables.new()
        var.name = "val"
        var.type = 'SINGLE_PROP'
        var.targets[0].id_type = 'LIGHT'
        var.targets[0].id = light_data
        var.targets[0].data_path = data_path
        drv.expression = expression
    except Exception:
        pass


def setup_spot_light_nodes(light_obj, image=None):
    """Configures the SPOT light's Cycles Shader Node Tree."""
    light_data = light_obj.data
    light_data.type = 'SPOT'
    light_data.use_nodes = True
    tree = light_data.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    out_node = nodes.new('ShaderNodeOutputLight')
    out_node.location = (400, 0)

    emit_node = nodes.new('ShaderNodeEmission')
    emit_node.location = (180, 0)
    emit_node.inputs['Strength'].default_value = 1.0

    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.name = "SPOT_PROJ_TEX"
    tex_node.label = "Projected Texture"
    tex_node.extension = 'CLIP'
    tex_node.location = (-120, 0)
    if image:
        tex_node.image = image

    sep_xyz = nodes.new('ShaderNodeSeparateXYZ')
    sep_xyz.location = (-750, 0)

    div_x = nodes.new('ShaderNodeMath')
    div_x.operation = 'DIVIDE'
    div_x.location = (-560, 120)

    div_y = nodes.new('ShaderNodeMath')
    div_y.operation = 'DIVIDE'
    div_y.location = (-560, -80)

    comb_xyz = nodes.new('ShaderNodeCombineXYZ')
    comb_xyz.location = (-380, 0)

    mapping = nodes.new('ShaderNodeMapping')
    mapping.name = "SPOT_MAPPING"
    mapping.vector_type = 'POINT'
    mapping.location = (-300, 0)
    mapping.inputs['Location'].default_value = (0.5, 0.5, 0.0)

    expr_scale = "-0.5 / max(0.005, tan(val * 0.5))"
    add_light_prop_driver(mapping.inputs['Scale'], "default_value", 0, light_data, "spot_size", expr_scale)
    add_light_prop_driver(mapping.inputs['Scale'], "default_value", 1, light_data, "spot_size", expr_scale)

    coord = nodes.new('ShaderNodeTexCoord')
    coord.location = (-950, 0)

    links.new(coord.outputs['Normal'], sep_xyz.inputs['Vector'])
    links.new(sep_xyz.outputs['X'], div_x.inputs[0])
    links.new(sep_xyz.outputs['Z'], div_x.inputs[1])
    links.new(sep_xyz.outputs['Y'], div_y.inputs[0])
    links.new(sep_xyz.outputs['Z'], div_y.inputs[1])
    links.new(div_x.outputs['Value'], comb_xyz.inputs['X'])
    links.new(div_y.outputs['Value'], comb_xyz.inputs['Y'])
    links.new(comb_xyz.outputs['Vector'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
    links.new(tex_node.outputs['Color'], emit_node.inputs['Color'])
    links.new(emit_node.outputs['Emission'], out_node.inputs['Surface'])


def get_spot_light_image(light_obj):
    if not light_obj or light_obj.type != 'LIGHT' or not light_obj.data.use_nodes:
        return None
    tex_node = light_obj.data.node_tree.nodes.get("SPOT_PROJ_TEX")
    if tex_node and tex_node.image:
        return tex_node.image
    for n in light_obj.data.node_tree.nodes:
        if n.type == 'TEX_IMAGE' and n.image:
            return n.image
    return None


def rebuild_target_material_preview(context):
    global _is_updating_nodes
    if _is_updating_nodes:
        return
    _is_updating_nodes = True

    try:
        scene = getattr(context, "scene", None)
        if not scene or not hasattr(scene, "spot_shader_paint"):
            return
        settings = scene.spot_shader_paint
        target = settings.target_obj
        if not target or target.type != 'MESH':
            return

        if not settings.canvas_image:
            img_name = f"{target.name}_PaintedCanvas"
            img = bpy.data.images.get(img_name)
            if not img:
                img = bpy.data.images.new(img_name, width=2048, height=2048, alpha=True)
                img.generated_color = (0.5, 0.5, 0.5, 1.0)
            settings.canvas_image = img

        if not target.data.materials:
            mat = bpy.data.materials.new(name=f"{target.name}_SpotPaint_Mat")
            mat.use_nodes = True
            target.data.materials.append(mat)
        else:
            mat = target.data.materials[0]
            mat.use_nodes = True

        tree = mat.node_tree
        nodes = tree.nodes
        links = tree.links

        for n in list(nodes):
            if n.name.startswith("SP_AUTO_"):
                nodes.remove(n)

        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not bsdf:
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            bsdf.location = (400, 0)
            out = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
            if not out:
                out = nodes.new('ShaderNodeOutputMaterial')
                out.location = (700, 0)
            links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

        canvas_node = nodes.get("SP_CANVAS_TEX")
        if not canvas_node:
            canvas_node = nodes.new('ShaderNodeTexImage')
            canvas_node.name = "SP_CANVAS_TEX"
            canvas_node.label = "Painted UV Canvas"
            canvas_node.location = (-600, 200)
        canvas_node.image = settings.canvas_image

        current_color_socket = canvas_node.outputs['Color']

        if settings.show_shader_preview and settings.preview_opacity > 0.001:
            for idx, item in enumerate(settings.projectors):
                light_obj = item.light_obj
                if not item.enabled or not light_obj or light_obj.type != 'LIGHT':
                    continue
                img = get_spot_light_image(light_obj)
                if not img:
                    continue

                y_off = -380 * (idx + 1)
                light_data = light_obj.data

                tc = nodes.new('ShaderNodeTexCoord')
                tc.name = f"SP_AUTO_TC_{idx}"
                tc.object = light_obj
                tc.location = (-1650, y_off)

                sep = nodes.new('ShaderNodeSeparateXYZ')
                sep.name = f"SP_AUTO_SEP_{idx}"
                sep.location = (-1450, y_off)
                links.new(tc.outputs['Object'], sep.inputs['Vector'])

                neg_z = nodes.new('ShaderNodeMath')
                neg_z.name = f"SP_AUTO_NEGZ_{idx}"
                neg_z.operation = 'MULTIPLY'
                neg_z.inputs[1].default_value = -1.0
                neg_z.location = (-1250, y_off - 120)
                links.new(sep.outputs['Z'], neg_z.inputs[0])

                in_front = nodes.new('ShaderNodeMath')
                in_front.name = f"SP_AUTO_FRONT_{idx}"
                in_front.operation = 'GREATER_THAN'
                in_front.inputs[1].default_value = 0.001
                in_front.location = (-1050, y_off - 240)
                links.new(neg_z.outputs['Value'], in_front.inputs[0])

                denom = nodes.new('ShaderNodeMath')
                denom.name = f"SP_AUTO_DENOM_{idx}"
                denom.operation = 'MULTIPLY'
                denom.location = (-1050, y_off - 80)
                links.new(neg_z.outputs['Value'], denom.inputs[0])
                add_light_prop_driver(
                    denom.inputs[1], "default_value", -1,
                    light_data, "spot_size", "max(0.005, tan(val * 0.5)) * 2.0"
                )

                div_u = nodes.new('ShaderNodeMath')
                div_u.name = f"SP_AUTO_DIVU_{idx}"
                div_u.operation = 'DIVIDE'
                div_u.location = (-860, y_off + 80)
                links.new(sep.outputs['X'], div_u.inputs[0])
                links.new(denom.outputs['Value'], div_u.inputs[1])

                div_v = nodes.new('ShaderNodeMath')
                div_v.name = f"SP_AUTO_DIVV_{idx}"
                div_v.operation = 'DIVIDE'
                div_v.location = (-860, y_off - 40)
                links.new(sep.outputs['Y'], div_v.inputs[0])
                links.new(denom.outputs['Value'], div_v.inputs[1])

                comb_centered = nodes.new('ShaderNodeCombineXYZ')
                comb_centered.name = f"SP_AUTO_COMBC_{idx}"
                comb_centered.location = (-860, y_off - 180)
                links.new(div_u.outputs['Value'], comb_centered.inputs['X'])
                links.new(div_v.outputs['Value'], comb_centered.inputs['Y'])

                rad_len = nodes.new('ShaderNodeVectorMath')
                rad_len.name = f"SP_AUTO_RAD_{idx}"
                rad_len.operation = 'LENGTH'
                rad_len.location = (-680, y_off - 180)
                links.new(comb_centered.outputs['Vector'], rad_len.inputs[0])

                cone_feather = nodes.new('ShaderNodeMapRange')
                cone_feather.name = f"SP_AUTO_CMASK_{idx}"
                cone_feather.clamp = True
                cone_feather.inputs['From Max'].default_value = 0.5
                cone_feather.inputs['To Min'].default_value = 1.0
                cone_feather.inputs['To Max'].default_value = 0.0
                cone_feather.location = (-480, y_off - 180)
                links.new(rad_len.outputs['Value'], cone_feather.inputs['Value'])
                add_light_prop_driver(
                    cone_feather.inputs['From Min'], "default_value", -1,
                    light_data, "spot_blend", "0.5 * (1.0 - max(0.001, val))"
                )

                add_u = nodes.new('ShaderNodeMath')
                add_u.name = f"SP_AUTO_ADDU_{idx}"
                add_u.operation = 'ADD'
                add_u.inputs[1].default_value = 0.5
                add_u.location = (-680, y_off + 80)
                links.new(div_u.outputs['Value'], add_u.inputs[0])

                add_v = nodes.new('ShaderNodeMath')
                add_v.name = f"SP_AUTO_ADDV_{idx}"
                add_v.operation = 'ADD'
                add_v.inputs[1].default_value = 0.5
                add_v.location = (-680, y_off - 40)
                links.new(div_v.outputs['Value'], add_v.inputs[0])

                comb_uv = nodes.new('ShaderNodeCombineXYZ')
                comb_uv.name = f"SP_AUTO_COMBUV_{idx}"
                comb_uv.location = (-480, y_off + 40)
                links.new(add_u.outputs['Value'], comb_uv.inputs['X'])
                links.new(add_v.outputs['Value'], comb_uv.inputs['Y'])

                proj_tex = nodes.new('ShaderNodeTexImage')
                proj_tex.name = f"SP_AUTO_TEX_{idx}"
                proj_tex.image = img
                proj_tex.extension = 'CLIP'
                proj_tex.interpolation = 'Linear'  # Smooth bilinear filtering
                proj_tex.location = (-300, y_off + 40)
                links.new(comb_uv.outputs['Vector'], proj_tex.inputs['Vector'])

                mask_mul1 = nodes.new('ShaderNodeMath')
                mask_mul1.name = f"SP_AUTO_MM1_{idx}"
                mask_mul1.operation = 'MULTIPLY'
                mask_mul1.location = (-260, y_off - 180)
                links.new(in_front.outputs['Value'], mask_mul1.inputs[0])
                links.new(cone_feather.outputs['Result'], mask_mul1.inputs[1])

                mask_mul2 = nodes.new('ShaderNodeMath')
                mask_mul2.name = f"SP_AUTO_MM2_{idx}"
                mask_mul2.operation = 'MULTIPLY'
                mask_mul2.location = (-80, y_off - 140)
                links.new(mask_mul1.outputs['Value'], mask_mul2.inputs[0])
                links.new(proj_tex.outputs['Alpha'], mask_mul2.inputs[1])

                mask_mul3 = nodes.new('ShaderNodeMath')
                mask_mul3.name = f"SP_AUTO_MM3_{idx}"
                mask_mul3.operation = 'MULTIPLY'
                mask_mul3.inputs[1].default_value = min(1.0, settings.preview_opacity * item.intensity)
                mask_mul3.location = (80, y_off - 140)
                links.new(mask_mul2.outputs['Value'], mask_mul3.inputs[0])

                mix_node = nodes.new('ShaderNodeMix')
                mix_node.name = f"SP_AUTO_MIX_{idx}"
                mix_node.data_type = 'RGBA'
                mix_node.location = (240, y_off + 60)
                links.new(mask_mul3.outputs['Value'], mix_node.inputs['Factor'])
                links.new(current_color_socket, mix_node.inputs['A'])
                links.new(proj_tex.outputs['Color'], mix_node.inputs['B'])

                current_color_socket = mix_node.outputs['Result']

        links.new(current_color_socket, bsdf.inputs['Base Color'])
    finally:
        _is_updating_nodes = False


@persistent
def depsgraph_spot_sync_handler(scene, depsgraph):
    if _is_updating_nodes or not hasattr(scene, "spot_shader_paint"):
        return
    settings = scene.spot_shader_paint
    if not settings.show_shader_preview or not settings.target_obj:
        return

    target = settings.target_obj
    if not target.data or not target.data.materials:
        return
    mat = target.data.materials[0]
    if not mat or not mat.use_nodes:
        return

    nodes = mat.node_tree.nodes
    needs_rebuild = False

    for idx, item in enumerate(settings.projectors):
        if not item.enabled or not item.light_obj or item.light_obj.type != 'LIGHT':
            continue
        lamp_img = get_spot_light_image(item.light_obj)
        mat_tex_node = nodes.get(f"SP_AUTO_TEX_{idx}")
        if lamp_img and mat_tex_node is None:
            needs_rebuild = True
            break
        if mat_tex_node and mat_tex_node.image != lamp_img:
            mat_tex_node.image = lamp_img

    if needs_rebuild:
        rebuild_target_material_preview(bpy.context)


def sample_image_bilinear(src_arr, s_w, s_h, u, v):
    """Bilinear texture sampling for smooth, non-pixelated projection lookup."""
    fx = u * (s_w - 1)
    fy = v * (s_h - 1)
    x0 = int(math.floor(fx))
    y0 = int(math.floor(fy))
    x1 = min(s_w - 1, x0 + 1)
    y1 = min(s_h - 1, y0 + 1)

    tx = fx - x0
    ty = fy - y0

    p00 = src_arr[y0, x0]
    p10 = src_arr[y0, x1]
    p01 = src_arr[y1, x0]
    p11 = src_arr[y1, x1]

    top = p00 * (1.0 - tx) + p10 * tx
    bot = p01 * (1.0 - tx) + p11 * tx
    return top * (1.0 - ty) + bot * ty


def evaluate_spot_lamp_at_3d_point(world_pt, world_norm, light_obj, bvh_tree, inv_target_mat):
    light_mat = light_obj.matrix_world
    light_orig = light_mat.to_translation()

    to_pt = world_pt - light_orig
    dist = to_pt.length
    if dist < 0.0001:
        return None
    ray_dir_world = to_pt / dist

    facing = world_norm.dot(-ray_dir_world)
    if facing <= 0.02:
        return None

    local_pt = light_mat.inverted() @ world_pt
    if local_pt.z >= -0.001:
        return None

    depth = -local_pt.z
    half_angle = light_obj.data.spot_size * 0.5
    tan_half = max(0.005, math.tan(half_angle))

    norm_x = local_pt.x / (depth * tan_half * 2.0)
    norm_y = local_pt.y / (depth * tan_half * 2.0)

    radial = math.hypot(norm_x, norm_y)
    if radial > 0.5:
        return None

    u = norm_x + 0.5
    v = norm_y + 0.5
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None

    if bvh_tree is not None:
        local_orig = inv_target_mat @ light_orig
        local_target = inv_target_mat @ world_pt
        local_vec = local_target - local_orig
        local_dist = local_vec.length
        if local_dist > 0.0001:
            local_dir = local_vec / local_dist
            hit_loc, _, _, _ = bvh_tree.ray_cast(local_orig, local_dir, local_dist - 0.003)
            if hit_loc is not None:
                return None

    blend = max(0.001, light_obj.data.spot_blend)
    norm_rad = radial * 2.0
    inner_edge = 1.0 - blend
    if norm_rad <= inner_edge:
        cone_falloff = 1.0
    else:
        cone_falloff = max(0.0, (1.0 - norm_rad) / blend)

    weight = facing * cone_falloff
    return u, v, weight


class SPOTSHADER_OT_add_spotlight(bpy.types.Operator):
    bl_idname = "spot_shader.add_spotlight"
    bl_label = "Add Real SpotLight Projector"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.spot_shader_paint
        if not settings.target_obj and context.active_object and context.active_object.type == 'MESH':
            settings.target_obj = context.active_object

        idx = len(settings.projectors) + 1
        light_data = bpy.data.lights.new(name=f"SpotProjector_{idx}", type='SPOT')
        light_data.energy = 500.0
        light_data.spot_size = math.radians(35.0)
        light_data.spot_blend = 0.25
        light_data.show_cone = True

        light_obj = bpy.data.objects.new(name=f"SpotProjector_{idx}", object_data=light_data)
        context.collection.objects.link(light_obj)

        if context.region_data:
            light_obj.matrix_world = context.region_data.view_matrix.inverted()

        setup_spot_light_nodes(light_obj, None)

        item = settings.projectors.add()
        item.name = light_obj.name
        item.light_obj = light_obj
        settings.active_index = len(settings.projectors) - 1

        rebuild_target_material_preview(context)
        self.report({'INFO'}, f"Created Driven SpotLight Projector: {light_obj.name}")
        return {'FINISHED'}


class SPOTSHADER_OT_remove_spotlight(bpy.types.Operator):
    bl_idname = "spot_shader.remove_spotlight"
    bl_label = "Remove SpotLight Projector"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.spot_shader_paint
        idx = settings.active_index
        if 0 <= idx < len(settings.projectors):
            settings.projectors.remove(idx)
            settings.active_index = max(0, idx - 1)
            rebuild_target_material_preview(context)
        return {'FINISHED'}


class SPOTSHADER_OT_snap_to_view(bpy.types.Operator):
    bl_idname = "spot_shader.snap_to_view"
    bl_label = "Snap SpotLight to Current View"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.spot_shader_paint
        idx = settings.active_index
        if 0 <= idx < len(settings.projectors):
            light_obj = settings.projectors[idx].light_obj
            if light_obj and context.region_data:
                light_obj.matrix_world = context.region_data.view_matrix.inverted()
        return {'FINISHED'}


class SPOTSHADER_OT_sync_material(bpy.types.Operator):
    bl_idname = "spot_shader.sync_material"
    bl_label = "Setup / Sync Driven Shader Nodes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rebuild_target_material_preview(context)
        self.report({'INFO'}, "Driven Shader Nodes synchronized!")
        return {'FINISHED'}


class SPOTSHADER_OT_bake_all_projections(bpy.types.Operator):
    """Bakes all active SpotLight texture projections onto the target UV canvas in one click"""
    bl_idname = "spot_shader.bake_projections"
    bl_label = "One-Click Bake All Projections to UV"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.spot_shader_paint
        target = settings.target_obj
        dest_img = settings.canvas_image

        if not target or not dest_img:
            self.report({'WARNING'}, "Select a Target Mesh and Canvas UV Texture first!")
            return {'CANCELLED'}

        active_projs = [p for p in settings.projectors if p.enabled and p.light_obj and p.light_obj.type == 'LIGHT']
        if not active_projs:
            self.report({'WARNING'}, "No active SpotLight projectors found with images assigned!")
            return {'CANCELLED'}

        # Load source images into cache
        src_cache = {}
        for item in active_projs:
            img = get_spot_light_image(item.light_obj)
            if img and img.name not in src_cache:
                sw, sh = img.size
                sbuf = np.empty(sw * sh * 4, dtype=np.float32)
                img.pixels.foreach_get(sbuf)
                src_cache[img.name] = (sbuf.reshape((sh, sw, 4)), sw, sh)

        if not src_cache:
            self.report({'WARNING'}, "Assign images to your SpotLights before baking!")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = target.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        uv_layer = eval_mesh.uv_layers.active.data if eval_mesh.uv_layers.active else None

        if not uv_layer:
            eval_obj.to_mesh_clear()
            self.report({'ERROR'}, "Target mesh has no active UV Map!")
            return {'CANCELLED'}

        bvh_tree = BVHTree.FromObject(target, depsgraph)
        inv_target = target.matrix_world.inverted()
        normal_mat = target.matrix_world.to_3x3()

        dst_w, dst_h = dest_img.size
        dst_pixels = np.empty(dst_w * dst_h * 4, dtype=np.float32)
        dest_img.pixels.foreach_get(dst_pixels)
        dst_pixels = dst_pixels.reshape((dst_h, dst_w, 4))

        hit_count = 0
        # Iterate over polygon loop triangles for complete surface baking
        eval_mesh.calc_loop_triangles()
        for tri in eval_mesh.loop_triangles:
            poly_idx = tri.polygon_index
            poly = eval_mesh.polygons[poly_idx]
            
            # Sample center or vertices of loop triangle / grid points
            # For robust one-click bake, let's sample the face center + vertices
            for l_idx in tri.loops:
                loop = eval_mesh.loops[l_idx]
                v_co = eval_mesh.vertices[loop.vertex_index].co
                uv_pt = uv_layer[l_idx].uv

                # Transform to world space
                world_pt = target.matrix_world @ v_co
                # Approximate vertex normal
                v_normal = eval_mesh.vertices[loop.vertex_index].normal
                world_norm = (normal_mat @ v_normal).normalized()

                accum_rgb = np.zeros(3, dtype=np.float32)
                total_w = 0.0

                for item in active_projs:
                    img = get_spot_light_image(item.light_obj)
                    if not img or img.name not in src_cache:
                        continue

                    res = evaluate_spot_lamp_at_3d_point(
                        world_pt, world_norm, item.light_obj, bvh_tree, inv_target
                    )
                    if res is None:
                        continue

                    u_proj, v_proj, weight = res
                    weight *= item.intensity
                    src_arr, s_w, s_h = src_cache[img.name]

                    px = sample_image_bilinear(src_arr, s_w, s_h, u_proj, v_proj)
                    eff_w = weight * px[3]
                    if eff_w > 0.001:
                        accum_rgb += px[:3] * eff_w
                        total_w += eff_w

                if total_w > 0.001:
                    final_rgb = accum_rgb / total_w
                    px_x = int(uv_pt.x * (dst_w - 1)) % dst_w
                    px_y = int(uv_pt.y * (dst_h - 1)) % dst_h
                    
                    dst_pixels[px_y, px_x, :3] = final_rgb
                    dst_pixels[px_y, px_x, 3] = 1.0
                    hit_count += 1

        dest_img.pixels.foreach_set(dst_pixels.ravel())
        dest_img.update()
        eval_obj.to_mesh_clear()

        self.report({'INFO'}, f"Successfully baked projections onto {hit_count} UV vertices/pixels!")
        return {'FINISHED'}


class SPOTSHADER_OT_paint_projection(bpy.types.Operator):
    """Interactive 3D Raycast Brush with smooth bilinear sampling"""
    bl_idname = "spot_shader.paint_projection"
    bl_label = "Start SpotLight Projection Brush"
    bl_options = {'REGISTER', 'UNDO'}

    _is_painting = False
    _bvh_tree = None
    _eval_mesh = None
    _dst_pixels = None
    _dst_size = (0, 0)
    _src_cache = {}
    _cursor_handler = None
    _mouse_pos = (0, 0)

    def draw_brush_ring(self):
        settings = bpy.context.scene.spot_shader_paint
        radius = settings.brush_radius_px
        cx, cy = self._mouse_pos
        segments = 40
        pts = [
            (cx + math.cos(2 * math.pi * i / segments) * radius,
             cy + math.sin(2 * math.pi * i / segments) * radius)
            for i in range(segments + 1)
        ]
        shader = get_builtin_shader('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
        shader.bind()
        shader.uniform_float("color", (1.0, 0.75, 0.15, 0.95))
        batch.draw(shader)
        gpu.state.blend_set('NONE')

    def paint_dab(self, context, mouse_x, mouse_y):
        settings = context.scene.spot_shader_paint
        target = settings.target_obj
        dest_img = settings.canvas_image
        radius_px = settings.brush_radius_px
        strength = settings.brush_strength

        region = context.region
        rv3d = context.region_data
        inv_target = target.matrix_world.inverted()
        normal_mat = target.matrix_world.to_3x3()

        dst_w, dst_h = self._dst_size
        uv_layer = self._eval_mesh.uv_layers.active.data

        # Denser step and bilinear interpolation for smooth, non-pixelated brushing
        step = max(1, int(radius_px / 14))
        modified = False

        for dy in range(-radius_px, radius_px + 1, step):
            for dx in range(-radius_px, radius_px + 1, step):
                r_dist = math.hypot(dx, dy)
                if r_dist > radius_px:
                    continue

                falloff = (1.0 - (r_dist / radius_px)) ** 1.5
                coord_2d = (mouse_x + dx, mouse_y + dy)

                view_orig = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord_2d)
                view_dir = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord_2d)

                local_orig = inv_target @ view_orig
                local_dir = (inv_target.to_3x3() @ view_dir).normalized()

                hit_loc, hit_norm, poly_idx, _ = self._bvh_tree.ray_cast(local_orig, local_dir)
                if hit_loc is None:
                    continue

                world_pt = target.matrix_world @ hit_loc
                world_norm = (normal_mat @ hit_norm).normalized()

                accum_rgb = np.zeros(3, dtype=np.float32)
                total_w = 0.0

                for item in settings.projectors:
                    light_obj = item.light_obj
                    if not item.enabled or not light_obj or light_obj.type != 'LIGHT':
                        continue
                    img = get_spot_light_image(light_obj)
                    if not img or img.name not in self._src_cache:
                        continue

                    res = evaluate_spot_lamp_at_3d_point(
                        world_pt, world_norm, light_obj, self._bvh_tree, inv_target
                    )
                    if res is None:
                        continue

                    u_proj, v_proj, weight = res
                    weight *= item.intensity
                    src_arr, s_w, s_h = self._src_cache[img.name]

                    # Smooth Bilinear Sampling
                    px = sample_image_bilinear(src_arr, s_w, s_h, u_proj, v_proj)

                    eff_w = weight * px[3]
                    if eff_w > 0.001:
                        accum_rgb += px[:3] * eff_w
                        total_w += eff_w

                if total_w <= 0.001:
                    continue

                final_rgb = accum_rgb / total_w

                poly = self._eval_mesh.polygons[poly_idx]
                v_coords = [self._eval_mesh.vertices[v].co for v in poly.vertices]
                bary_weights = poly_3d_calc(v_coords, hit_loc)

                uv_hit = Vector((0.0, 0.0))
                for loop_idx, w in zip(poly.loop_indices, bary_weights):
                    uv_hit += uv_layer[loop_idx].uv * w

                px_x = int(uv_hit.x * (dst_w - 1)) % dst_w
                px_y = int(uv_hit.y * (dst_h - 1)) % dst_h

                blend = min(1.0, falloff * strength * min(1.0, total_w))
                pad = max(1, int(step * 0.7))

                y0, y1 = max(0, px_y - pad), min(dst_h, px_y + pad + 1)
                x0, x1 = max(0, px_x - pad), min(dst_w, px_x + pad + 1)

                self._dst_pixels[y0:y1, x0:x1, :3] = (
                    self._dst_pixels[y0:y1, x0:x1, :3] * (1.0 - blend) + final_rgb * blend
                )
                self._dst_pixels[y0:y1, x0:x1, 3] = 1.0
                modified = True

        if modified:
            dest_img.pixels.foreach_set(self._dst_pixels.ravel())
            dest_img.update()

    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()
        self._mouse_pos = (event.mouse_region_x, event.mouse_region_y)

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} or event.alt:
            return {'PASS_THROUGH'}

        if event.type == 'LEFT_BRACKET' and event.value == 'PRESS':
            context.scene.spot_shader_paint.brush_radius_px = max(5, context.scene.spot_shader_paint.brush_radius_px - 5)
            return {'RUNNING_MODAL'}
        if event.type == 'RIGHT_BRACKET' and event.value == 'PRESS':
            context.scene.spot_shader_paint.brush_radius_px = min(250, context.scene.spot_shader_paint.brush_radius_px + 5)
            return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.finish(context)
            return {'FINISHED'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self._is_painting = True
                self.paint_dab(context, event.mouse_region_x, event.mouse_region_y)
                return {'RUNNING_MODAL'}
            elif event.value == 'RELEASE':
                self._is_painting = False
                return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE' and self._is_painting:
            self.paint_dab(context, event.mouse_region_x, event.mouse_region_y)
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        settings = context.scene.spot_shader_paint
        if not settings.target_obj and context.active_object and context.active_object.type == 'MESH':
            settings.target_obj = context.active_object

        rebuild_target_material_preview(context)

        target = settings.target_obj
        dest_img = settings.canvas_image
        if not target or not dest_img:
            self.report({'WARNING'}, "Select a Target Mesh first!")
            return {'CANCELLED'}

        self._src_cache.clear()
        for item in settings.projectors:
            if item.enabled and item.light_obj and item.light_obj.type == 'LIGHT':
                img = get_spot_light_image(item.light_obj)
                if img and img.name not in self._src_cache:
                    sw, sh = img.size
                    sbuf = np.empty(sw * sh * 4, dtype=np.float32)
                    img.pixels.foreach_get(sbuf)
                    self._src_cache[img.name] = (sbuf.reshape((sh, sw, 4)), sw, sh)

        if not self._src_cache:
            self.report({'WARNING'}, "Assign an Image Texture to at least one SpotLight first!")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = target.evaluated_get(depsgraph)
        self._eval_mesh = eval_obj.to_mesh()
        if not self._eval_mesh.uv_layers.active:
            eval_obj.to_mesh_clear()
            self.report({'ERROR'}, "Target Mesh has no active UV Map!")
            return {'CANCELLED'}

        self._bvh_tree = BVHTree.FromObject(target, depsgraph)

        w, h = dest_img.size
        self._dst_size = (w, h)
        buf = np.empty(w * h * 4, dtype=np.float32)
        dest_img.pixels.foreach_get(buf)
        self._dst_pixels = buf.reshape((h, w, 4))

        self._cursor_handler = bpy.types.SpaceView3D.draw_handler_add(
            self.draw_brush_ring, (), 'WINDOW', 'POST_PIXEL'
        )
        settings.is_painting = True
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "SpotLight Brush Active: Left-Click to paint, [ / ] size, Right-Click/ESC to finish")
        return {'RUNNING_MODAL'}

    def finish(self, context):
        context.scene.spot_shader_paint.is_painting = False
        if self._cursor_handler:
            bpy.types.SpaceView3D.draw_handler_remove(self._cursor_handler, 'WINDOW')
            self._cursor_handler = None
        if context.scene.spot_shader_paint.target_obj:
            depsgraph = context.evaluated_depsgraph_get()
            context.scene.spot_shader_paint.target_obj.evaluated_get(depsgraph).to_mesh_clear()
        self._bvh_tree = None
        self._src_cache.clear()


def on_shader_preview_update(self, context):
    rebuild_target_material_preview(context)


class SpotLightProjectorItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Name", default="SpotProjector")
    enabled: bpy.props.BoolProperty(
        name="Enabled",
        default=True,
        update=on_shader_preview_update
    )
    light_obj: bpy.props.PointerProperty(
        name="Spot Light",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'LIGHT' and obj.data.type == 'SPOT',
        update=on_shader_preview_update
    )
    intensity: bpy.props.FloatProperty(
        name="Mix Weight",
        default=1.0,
        min=0.0,
        max=2.0,
        subtype='FACTOR',
        update=on_shader_preview_update
    )


class SpotShaderPaintSettings(bpy.types.PropertyGroup):
    target_obj: bpy.props.PointerProperty(
        name="Target Mesh",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH',
        update=on_shader_preview_update
    )
    canvas_image: bpy.props.PointerProperty(
        name="Canvas UV Texture",
        type=bpy.types.Image,
        update=on_shader_preview_update
    )
    projectors: bpy.props.CollectionProperty(type=SpotLightProjectorItem)
    active_index: bpy.props.IntProperty(name="Active SpotLight", default=0)

    show_shader_preview: bpy.props.BoolProperty(
        name="Live Shader Preview on Mesh",
        default=True,
        description="Shows the SpotLight texture projection directly inside the Target Mesh's material shader",
        update=on_shader_preview_update
    )
    preview_opacity: bpy.props.FloatProperty(
        name="Shader Preview Opacity",
        default=0.55,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=on_shader_preview_update
    )

    brush_radius_px: bpy.props.IntProperty(name="Brush Radius (px)", default=45, min=5, max=250)
    brush_strength: bpy.props.FloatProperty(name="Brush Strength", default=0.85, min=0.05, max=1.0, subtype='FACTOR')
    is_painting: bpy.props.BoolProperty(name="Is Painting", default=False)


class SPOTSHADER_UL_projector_list(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(item, "light_obj", text="", icon='LIGHT_SPOT')
        img = get_spot_light_image(item.light_obj) if item.light_obj else None
        row.label(text=img.name if img else "Select Texture Below", icon='IMAGE_DATA' if img else 'INFO')
        row.prop(item, "intensity", text="Mix")


class SPOTSHADER_PT_main_panel(bpy.types.Panel):
    bl_label = "SpotLight Shader Projector"
    bl_idname = "SPOTSHADER_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SpotLight Paint"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.spot_shader_paint

        # --- BRUSH & BAKE BOX AT THE TOP ---
        box_paint = layout.box()
        box_paint.label(text="SpotLight Raycast Brush & Bake", icon='BRUSH_DATA')
        col_p = box_paint.column(align=True)
        col_p.prop(settings, "brush_radius_px")
        col_p.prop(settings, "brush_strength")
        col_p.separator()
        if not settings.is_painting:
            row_op = col_p.row(align=True)
            row_op.scale_y = 1.4
            row_op.operator("spot_shader.paint_projection", text="Start Smooth Brush Paint", icon='VPAINT_HLT')
        else:
            col_p.label(text="PAINTING ACTIVE (Right-Click / ESC to stop)", icon='REC')

        col_p.separator()
        row_bake = col_p.row(align=True)
        row_bake.scale_y = 1.3
        row_bake.operator("spot_shader.bake_projections", text="One-Click Bake All Projections", icon='RENDER_STILL')

        # --- 1. TARGET & CANVAS ---
        box1 = layout.box()
        box1.label(text="1. Target Mesh & UV Canvas", icon='MESH_DATA')
        box1.prop(settings, "target_obj")
        box1.prop(settings, "canvas_image")
        box1.operator("spot_shader.sync_material", text="Setup / Sync Driven Shader Nodes", icon='NODE_MATERIAL')

        # --- 2. SPOTLIGHT PROJECTORS ---
        box2 = layout.box()
        box2.label(text="2. SpotLight Projectors (Live Driven)", icon='LIGHT_SPOT')
        row = box2.row()
        row.template_list("SPOTSHADER_UL_projector_list", "", settings, "projectors", settings, "active_index", rows=3)
        col = row.column(align=True)
        col.operator("spot_shader.add_spotlight", icon='ADD', text="")
        col.operator("spot_shader.remove_spotlight", icon='REMOVE', text="")
        col.separator()
        col.operator("spot_shader.snap_to_view", icon='VIEW_CAMERA', text="")

        if 0 <= settings.active_index < len(settings.projectors):
            active_item = settings.projectors[settings.active_index]
            light_obj = active_item.light_obj
            if light_obj and light_obj.type == 'LIGHT':
                sub = box2.box()
                sub.label(text=f"SpotLight: {light_obj.name}", icon='LIGHT_SPOT')

                if not light_obj.data.use_nodes:
                    setup_spot_light_nodes(light_obj)
                tex_node = light_obj.data.node_tree.nodes.get("SPOT_PROJ_TEX")
                if tex_node:
                    sub.template_ID(tex_node, "image", open="image.open")

                col_cone = sub.column(align=True)
                col_cone.prop(light_obj.data, "spot_size", text="Cone Angle (Live Driven)")
                col_cone.prop(light_obj.data, "spot_blend", text="Cone Softness (Live Driven)")
                col_cone.prop(light_obj.data, "energy", text="Cycles Lamp Power")
                col_cone.prop(light_obj.data, "show_cone", text="Show 3D Spot Cone")

        # --- 3. SHADER PREVIEW ON MESH ---
        box3 = layout.box()
        box3.label(text="3. Live Shader Projection Preview", icon='SHADING_RENDERED')
        box3.prop(settings, "show_shader_preview")
        if settings.show_shader_preview:
            box3.prop(settings, "preview_opacity", text="Overlay Opacity")


classes = (
    SpotLightProjectorItem,
    SpotShaderPaintSettings,
    SPOTSHADER_UL_projector_list,
    SPOTSHADER_OT_add_spotlight,
    SPOTSHADER_OT_remove_spotlight,
    SPOTSHADER_OT_snap_to_view,
    SPOTSHADER_OT_sync_material,
    SPOTSHADER_OT_bake_all_projections,
    SPOTSHADER_OT_paint_projection,
    SPOTSHADER_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.spot_shader_paint = bpy.props.PointerProperty(type=SpotShaderPaintSettings)
    if depsgraph_spot_sync_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_spot_sync_handler)


def unregister():
    if depsgraph_spot_sync_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_spot_sync_handler)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.spot_shader_paint


if __name__ == "__main__":
    register()