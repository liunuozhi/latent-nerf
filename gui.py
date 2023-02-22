from src.latent_nerf.configs.train_config import TrainConfig
from src.latent_nerf.training.trainer import Trainer
from src.latent_nerf.models.network_grid import NeRFNetwork
from src.latent_nerf.training.nerf_dataset import NeRFDataset
from src.utils import make_path, tensor2numpy
from src.latent_nerf.models.render_utils import get_rays

import pyrallis
from scipy.spatial.transform import Rotation as R
import torch
import dearpygui.dearpygui as dpg
import numpy as np


class GUITrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nerf = self.init_nerf().eval()
        self.load_checkpoint('./weights.pth', model_only=True)
        self.dataloaders = self.init_dataloaders()

    def init_nerf(self):
        return NeRFNetwork(self.cfg.render).to(self.device)

    # def calc_text_embeddings(self):
    #     return None

    def init_dataloaders(self):
        val_loader = NeRFDataset(self.cfg.render,
                                 device=self.device,
                                 type='val',
                                 H=self.cfg.render.eval_h,
                                 W=self.cfg.render.eval_w,
                                 size=self.cfg.log.eval_size).dataloader()
        return {'val': val_loader}

    def render_output(self, pose, H, W):
        # dataloader = self.dataloaders['val']
        # for i, data in enumerate(dataloader):

        # fixed focal
        cx = H / 2
        cy = W / 2
        fovy_range = self.cfg.render.fovy_range
        fov = (fovy_range[1] + fovy_range[0]) / 2
        focal = H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = torch.FloatTensor([focal, focal, cx, cy]).to(self.device)
        pose = torch.FloatTensor(pose).to(self.device)

        data = get_rays(pose[None], intrinsics, H, W, -1)
        data['H'] = torch.tensor(H).to(self.device)
        data['W'] = torch.tensor(W).to(self.device)

        with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
            preds, preds_depth, preds_normals = self.eval_render(data)

        pred, pred_depth, pred_normals = tensor2numpy(preds[0]), tensor2numpy(preds_depth[0]), tensor2numpy(
            preds_normals[0])
        return pred, pred_depth, pred_normals


class OrbitCamera:
    def __init__(self, img_wh, r):
        self.W, self.H = img_wh
        self.radius = r
        self.center = np.zeros(3)
        self.rot = np.eye(3)

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(0.05 * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])


class NGPGUI:
    def __init__(self, trainer):
        self.trainer = trainer
        img_wh = (1024, 1024)
        self.cam = OrbitCamera(img_wh, 2.5)
        self.W, self.H = img_wh
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)

        # placeholders
        self.dt = 0
        self.mean_samples = 0

        self.register_dpg()

    def render_cam(self, cam):
        rgb, depth, normal = self.trainer.render_output(cam.pose, 128, 128)
        print(rgb.shape)
        return rgb

    def register_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title="ngp_pl", width=self.W, height=self.H, resizable=False)

        ## register texture ##
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture")

        ## register window ##
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)

        def callback_depth(sender, app_data):
            self.img_mode = 1 - self.img_mode

        ## control window ##
        with dpg.window(label="Control", tag="_control_window", width=200, height=150):
            dpg.add_slider_float(label="exposure", default_value=0.2,
                                 min_value=1 / 60, max_value=32, tag="_exposure")
            dpg.add_button(label="show depth", tag="_button_depth",
                           callback=callback_depth)
            dpg.add_separator()
            dpg.add_text('no data', tag="_log_time")
            dpg.add_text('no data', tag="_samples_per_ray")

        ## register camera handler ##
        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.orbit(app_data[1], app_data[2])

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.scale(app_data)

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.pan(app_data[1], app_data[2])

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        ## Avoid scroll bar in the window ##
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        ## Launch the gui ##
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            dpg.set_value("_texture", self.render_cam(self.cam))
            dpg.render_dearpygui_frame()


@pyrallis.wrap()
def main(cfg: TrainConfig):
    gui_trainer = GUITrainer(cfg)
    return gui_trainer


if __name__ == '__main__':
    trainer = main()
    NGPGUI(trainer).render()
    dpg.destroy_context()
