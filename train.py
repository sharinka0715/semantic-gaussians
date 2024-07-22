import os
import uuid
import torch
from tqdm import tqdm
from functools import partial
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model import GaussianModel
from model.renderer import render
from scene import Scene
from utils.system_utils import set_seed
from utils.loss_utils import l1_loss, ssim, psnr


def init_dir(config):
    if not config.train.exp_name:
        unique_str = str(uuid.uuid4())
        config.model.model_dir = os.path.join("./output", unique_str[0:10])
    else:
        config.model.model_dir = f"./output/{config.train.exp_name}"

    print("Output folder: {}".format(config.model.model_dir))
    os.makedirs(config.model.model_dir, exist_ok=True)
    with open(os.path.join(config.model.model_dir, "config.yaml"), "w") as fp:
        OmegaConf.save(config, fp)

    os.makedirs(os.path.join(config.model.model_dir, "tb_logs"), exist_ok=True)
    writer = SummaryWriter(os.path.join(config.model.model_dir, "tb_logs"))
    return writer


def eval(args, iteration, scene: Scene, render_partial, writer):
    torch.cuda.empty_cache()
    eval_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {
            "name": "train",
            "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
        },
    )

    for config in eval_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            for idx, viewpoint in enumerate(config["cameras"]):
                viewpoint.cuda()
                image = torch.clamp(render_partial(viewpoint)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if writer and (idx < 5):
                    writer.add_images(
                        "view_{}/render".format(viewpoint.image_name),
                        image[None],
                        global_step=iteration,
                    )
                    if iteration == args.train.test_iterations[0]:
                        writer.add_images(
                            "view_{}/ground_truth".format(viewpoint.image_name),
                            gt_image[None],
                            global_step=iteration,
                        )
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
            writer.add_scalar(f"{config['name']}/l1_loss", l1_test, iteration)
            writer.add_scalar(f"{config['name']}/psnr", psnr_test, iteration)

    torch.cuda.empty_cache()


def train(config):
    scene = Scene(config.scene)
    gaussians = GaussianModel(config.model.sh_degree)
    first_iter = 0

    # if config.model.model_dir:
    #     if config.model.load_iteration == -1:
    #         loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, "point_cloud"))
    #     else:
    #         loaded_iter = config.model.load_iteration
    #     print("Loading trained model at iteration {}".format(loaded_iter))
    #     gaussians.load_ply(os.path.join(config.model.model_dir, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply"))
    # else:

    gaussians.create_from_pcd(scene.pcd, scene.cameras_extent, config.model.random_init)
    writer = init_dir(config)

    gaussians.training_setup(config.train)

    bg_color = [1, 1, 1] if config.scene.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, config.train.iterations), desc="Training progress")
    loader = DataLoader(
        scene.getTrainCameras(),
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x,
        num_workers=config.train.num_workers,
    )
    data_iter = iter(loader)
    first_iter += 1

    for iteration in range(first_iter, config.train.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        try:
            viewpoint_cam = next(data_iter)[0]
        except StopIteration:
            data_iter = iter(loader)
            viewpoint_cam = next(data_iter)[0]
        viewpoint_cam.cuda()

        # Render
        bg = torch.rand((3), device="cuda") if config.train.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, config.pipeline, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image #.cuda()
        if config.train.cut_edge:
            h, w = image.shape[1:3]
            ch, cw = h // 100, w // 100
            Ll1 = l1_loss(image[:, ch:-ch, cw:-cw], gt_image[:, ch:-ch, cw:-cw])
            loss = (1.0 - config.train.lambda_dssim) * Ll1 + config.train.lambda_dssim * (
                1.0 - ssim(image[:, ch:-ch, cw:-cw], gt_image[:, ch:-ch, cw:-cw])
            )
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - config.train.lambda_dssim) * Ll1 + config.train.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Densification
            if iteration < config.train.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > config.train.densify_from_iter and iteration % config.train.densification_interval == 0:
                    size_threshold = 20 if iteration > config.train.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        config.train.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % config.train.opacity_reset_interval == 0 or (
                    config.scene.white_background and iteration == config.train.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < config.train.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad()

            # Logging
            writer.add_scalar("train/l1_loss", Ll1.item(), iteration)
            writer.add_scalar("train/total_loss", loss.item(), iteration)
            writer.add_scalar("train/iter_time", iter_start.elapsed_time(iter_end), iteration)
            writer.add_scalar("train/total_points", gaussians.get_xyz.shape[0], iteration)
            writer.add_histogram("train/opacity_histogram", gaussians.get_opacity, iteration)

            # Evaluation
            if iteration in config.train.test_iterations:
                eval(
                    config,
                    iteration,
                    scene,
                    partial(render, pc=gaussians, pipe=config.pipeline, bg_color=bg),
                    writer,
                )

            # Saving gaussians
            if iteration in config.train.save_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                point_cloud_path = os.path.join(config.model.model_dir, "point_cloud/iteration_{}".format(iteration))
                gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            # if (iteration in config.train.checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), config.model.model_dir + "/chkpnt" + str(iteration) + ".pth")

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 20 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(20)
            if iteration == config.train.iterations:
                progress_bar.close()


if __name__ == "__main__":
    config = OmegaConf.load("./config/official_train.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)
    train(config)
