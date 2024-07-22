import os
import cv2
import time
import torch
import viser
from copy import deepcopy
from viser.transforms import SE3, SO3
import numpy as np
from omegaconf import OmegaConf

from model import GaussianModel, render, render_chn
from model.openseg_predictor import OpenSeg
from model.lseg_predictor import LSeg
from model.render_utils import render_palette
from scene import Scene
from utils.system_utils import searchForMaxIteration, set_seed
from utils.camera_utils import get_camera_viser
from utils.sh_utils import RGB2SH, SH2RGB
from dataset.scannet.scannet_constants import COLORMAP


def to_hex(color):
    return "{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_text(vocabulary, prefix_prompt=""):
    texts = [prefix_prompt + x.lower().replace("_", " ") for x in vocabulary]
    return texts


def main(config):
    with torch.no_grad():
        # Load 3D Gaussians
        scene_config = deepcopy(config)
        if config.model.dynamic:
            scene_config.scene.scene_path = os.path.join(config.scene.scene_path, "0")
        scene = Scene(scene_config.scene)
        gaussians = GaussianModel(config.model.sh_degree)

        if config.model.model_dir:
            if config.model.dynamic:
                gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), 0)
            else:
                if config.model.load_iteration == -1:
                    loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, "point_cloud"))
                else:
                    loaded_iter = config.model.load_iteration
                print("Loading trained model at iteration {}".format(loaded_iter))
                gaussians.load_ply(
                    os.path.join(
                        config.model.model_dir,
                        "point_cloud",
                        f"iteration_{loaded_iter}",
                        "point_cloud.ply",
                    )
                )
        else:
            raise NotImplementedError
        
        # Load semantic embeddings
        fusion = torch.load(config.render.fusion_dir)
        features, mask = fusion["feat"].float().cuda(), fusion["mask_full"].cuda()
        original_opacity = gaussians._opacity.detach().clone()
        original_scale = gaussians._scaling.detach().clone()
        original_color = gaussians._features_dc.detach().clone()
        original_coord = gaussians._xyz.detach().clone()

        # Create text encoding model
        if config.render.model_2d == "lseg": # 512dim CLIP
            text_model = LSeg(None)
        else: # 768dim CLIP
            text_model = OpenSeg(None, "ViT-L/14@336px")

        features_save = torch.zeros((mask.shape[0], text_model.embedding_dim)).float().cuda()
        features_save[mask] = features

        # Initialize colormap and camera
        colormap = COLORMAP
        colormap_hex = [to_hex(e) for e in colormap]
        colormap_cuda = torch.tensor(colormap).cuda()

        bg_color = [1, 1, 1] if config.scene.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        scene_camera = scene.getTrainCameras()[0]
        width, height = scene_camera.image_width, scene_camera.image_height
        w2c = scene_camera.world_view_transform.cpu().numpy().transpose()
        
        # Initialize viser
        server = viser.ViserServer()
        server.world_axes.visible = False
        need_update = False
        need_color_compute = True
        tab_group = server.add_gui_tab_group()

        # Settings tab
        with tab_group.add_tab("Settings", viser.Icon.SETTINGS):
            gui_render_mode = server.add_gui_button_group("Render mode", ("RGB", "Depth", "Semantic", "Relevancy"))
            render_mode = "RGB"
            gui_near_slider = server.add_gui_slider("Depth near", min=0, max=3, step=0.2, initial_value=1.5)
            gui_far_slider = server.add_gui_slider("Depth far", min=6, max=20, step=0.5, initial_value=6)
            gui_scale_slider = server.add_gui_slider("Gaussian scale", min=0.01, max=1, step=0.01, initial_value=1)
            resolution_scale_group = server.add_gui_button_group("Resolution scale", ("0.5x", "1x", "2x", "4x"))
            gui_background_checkbox = server.add_gui_checkbox("Remove background", False)
            gui_up_checkbox = server.add_gui_checkbox("Lock up direction", False)

            gui_prompt_input = server.add_gui_text(
                "Text prompt (divided by comma)",
                "wall,floor,cabinet,bed,chair,sofa,table,door,window,bookshelf,picture,counter,desk,curtain,refrigerator,shower curtain,toilet,sink,bathtub",
                # "wall,floor,sofa,table,television,plant,bookshelf,piano,door,speaker,slippers,bottle",
            )

            gui_prompt_button = server.add_gui_button("Apply text prompt")

        with tab_group.add_tab("Editing", viser.Icon.SETTINGS):
            gui_edit_mode = server.add_gui_button_group("Edit mode", ("Remove", "Color", "Size", "Move"))
            edit_mode = "Remove"
            gui_edit_input = server.add_gui_text("Edit prompt (divided by comma)", "")
            gui_preserve_input = server.add_gui_text("Preserve prompt (divided by comma)", "")
            gui_editing_button = server.add_gui_button("Apply editing prompt")

        # Colormap tab
        with tab_group.add_tab("Colormap", viser.Icon.COLOR_FILTER):
            gui_markdown = server.add_gui_markdown("")

        # Button callbacks
        @gui_render_mode.on_click
        def _(_) -> None:
            nonlocal render_mode
            render_mode = gui_render_mode.value

        @gui_edit_mode.on_click
        def _(_) -> None:
            nonlocal edit_mode
            nonlocal need_color_compute
            edit_mode = gui_edit_mode.value
            need_color_compute = True

        @gui_prompt_button.on_click
        def _(_) -> None:
            nonlocal need_color_compute
            need_color_compute = True

        @gui_editing_button.on_click
        def _(_) -> None:
            nonlocal need_color_compute
            need_color_compute = True

        @gui_scale_slider.on_update
        def _(_) -> None:
            nonlocal need_color_compute
            need_color_compute = True

        @resolution_scale_group.on_click
        def _(_) -> None:
            nonlocal need_color_compute
            need_color_compute = True

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            print("new client!")
            nonlocal w2c
            c2w_transform = SE3.from_matrix(w2c).inverse()
            client.camera.wxyz = c2w_transform.wxyz_xyz[:4]  # np.array([1.0, 0.0, 0.0, 0.0])
            client.camera.position = c2w_transform.wxyz_xyz[4:]

            # This will run whenever we get a new camera!
            @client.camera.on_update
            def _(_: viser.CameraHandle) -> None:
                nonlocal need_update
                need_update = True

        # Main render function. Render if camera moves or settings change.
        if config.model.dynamic:
            start_time = time.time()
            num_timesteps = config.model.num_timesteps
        while True:
            if config.model.dynamic:
                passed_time = time.time() - start_time
                passed_frames = passed_time * config.render.dynamic_fps
                t = int(passed_frames % num_timesteps)
            if need_color_compute:
                # Compute text embeddings and relevancy
                labelset = ["other"] + get_text(gui_prompt_input.value.split(","))
                text_features = text_model.extract_text_feature(labelset).float()
                sim = torch.einsum("cq,dq->dc", text_features, features)
                label_soft = sim.softmax(dim=1)
                label_hard = torch.nn.functional.one_hot(sim.argmax(dim=1), num_classes=label_soft.shape[1]).float()
                soft_save = torch.zeros((mask.shape[0], label_soft.shape[1])).float().cuda()
                soft_save[mask] = label_hard
                # sim[sim < 0] = -2
                label = sim.argmax(dim=1)
                colors = colormap_cuda[label] / 255
                color_save = torch.zeros((mask.shape[0], 3)).float().cuda()
                color_save[mask] = colors

                # Reset colormap tab
                color_mapping = list(zip(labelset, colormap_hex))
                content_head = "| | |\n|:-:|:-|"
                content_body = "".join(
                    [
                        f"\n|![color](https://via.placeholder.com/5x5/{color}/ffffff?text=+)|{label_name}||"
                        for label_name, color in color_mapping
                    ]
                )
                gui_markdown.content = content_head + content_body

                # Shape control
                height = int(float(resolution_scale_group.value[:-1]) * scene_camera.image_height)
                width = int(float(resolution_scale_group.value[:-1]) * scene_camera.image_width)

                # Scene editing control
                if gui_edit_input.value != "":
                    len_edit = len(gui_edit_input.value.split(","))
                    edit_features = text_model.extract_text_feature(
                        ["other"] + gui_edit_input.value.split(",") + gui_preserve_input.value.split(",")
                    ).float()
                    sim = torch.einsum("cq,dq->dc", edit_features, features)
                    sim[sim < 0] = -2
                    label = sim.argmax(dim=1)
                    gaussians._opacity[:] = original_opacity
                    gaussians._features_dc[:] = original_color
                    gaussians._scaling[:] = original_scale
                    gaussians._xyz[:] = original_coord

                    edit_mask = (label > 0) * (label <= len_edit)
                    if edit_mode == "Remove":
                        tmp = gaussians._opacity[mask]
                        tmp[edit_mask] = -9999
                        gaussians._opacity[mask] = tmp
                    elif edit_mode == "Color":
                        tmp = gaussians._features_dc[mask]
                        tmp_rgb = SH2RGB(tmp[edit_mask])  # [n_points, 1, 3]
                        tmp_rgb = 1 - tmp_rgb
                        tmp_rgb = torch.clamp(tmp_rgb, 0, 1)
                        tmp[edit_mask] = RGB2SH(tmp_rgb)
                        gaussians._features_dc[mask] = tmp
                    elif edit_mode == "Size":
                        tmp = gaussians._scaling[mask]
                        tmp[edit_mask] *= 2
                        gaussians._scaling[mask] = tmp
                        tmp = gaussians._xyz[mask]
                        tmp[edit_mask] *= 2
                        gaussians._xyz[mask] = tmp
                    elif edit_mode == "Move":
                        tmp = gaussians._xyz[mask]
                        tmp[edit_mask] += 1
                        gaussians._xyz[mask] = tmp
                else:
                    gaussians._opacity[:] = original_opacity
                    gaussians._features_dc[:] = original_color
                    gaussians._scaling[:] = original_scale
                    gaussians._xyz[:] = original_coord

                need_color_compute = False

            # Render for each client
            for client in server.get_clients().values():
                client_info = client.camera
                w2c_matrix = (
                    SE3.from_rotation_and_translation(SO3(client_info.wxyz), client_info.position).inverse().as_matrix()
                )
                if not gui_up_checkbox.value:
                    client.camera.up_direction = SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
                new_camera = get_camera_viser(
                    scene_camera,
                    w2c_matrix[:3, :3].transpose(),
                    w2c_matrix[:3, 3],
                    client_info.fov,
                    client_info.aspect,
                )
                new_camera.cuda()
                if config.model.dynamic:
                    gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), t)
                if render_mode == "RGB":
                    output = render(
                        new_camera,
                        gaussians,
                        config.pipeline,
                        background,
                        scaling_modifier=gui_scale_slider.value,
                        override_shape=(width, height),
                        foreground=gaussians.is_fg if gui_background_checkbox.value else None,
                    )
                    rendering = output["render"].cpu().numpy().transpose(1, 2, 0)
                elif render_mode == "Depth":
                    output = render(
                        new_camera,
                        gaussians,
                        config.pipeline,
                        background,
                        scaling_modifier=gui_scale_slider.value,
                        override_shape=(width, height),
                        foreground=gaussians.is_fg if gui_background_checkbox.value else None,
                    )
                    rendering = output["depth"].cpu().numpy().transpose(1, 2, 0)
                    rendering = np.clip(
                        (rendering - gui_near_slider.value) * 255 / (gui_far_slider.value - gui_near_slider.value),
                        0,
                        255,
                    )
                    rendering = cv2.cvtColor(rendering.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                elif render_mode == "Semantic":
                    output = render_chn(
                        new_camera,
                        gaussians,
                        config.pipeline,
                        background,
                        scaling_modifier=gui_scale_slider.value,
                        num_channels=soft_save.shape[1],
                        override_color=soft_save,
                        override_shape=(width//2, height//2),
                        foreground=gaussians.is_fg if gui_background_checkbox.value else None,
                    )
                    sim = output["render"]
                    label = sim.argmax(dim=0).cpu()
                    sem = render_palette(label, colormap_cuda.reshape(-1))
                    rendering = sem.cpu().numpy().transpose(1, 2, 0)
                else:  # relevancy
                    output = render(
                        new_camera,
                        gaussians,
                        config.pipeline,
                        background,
                        scaling_modifier=gui_scale_slider.value,
                        override_color=color_save,
                        override_shape=(width//2, height//2),
                        foreground=gaussians.is_fg if gui_background_checkbox.value else None,
                    )
                    rendering = output["render"].cpu().numpy().transpose(1, 2, 0)
                client.set_background_image(rendering)


if __name__ == "__main__":
    config = OmegaConf.load("./config/view_scannet.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)
    main(config)
