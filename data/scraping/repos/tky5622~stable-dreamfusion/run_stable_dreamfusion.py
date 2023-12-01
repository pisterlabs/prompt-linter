import torch
import argparse
import pandas as pd
import sys

from nerf.provider import NeRFDataset
from nerf.utils import *

# torch.autograd.set_detect_anomaly(True)
from argparse import Namespace

def main(
    workspace='', #workspace path
    file='', 
    text='', 
    negative='',

    # iters=2000,
    # lr=0.001,
    # dt_lr=0.001,
    # parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray")
    # parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla")
    test=False,
    six_views=True,
    eval_interval=1,
    test_interval=100,
    seed=None,
    image=None,
    image_config=None,
    known_view_interval=4,
    IF=True,
    guidance=['SD'],
    guidance_scale=100,
    save_mesh=True, #"export an obj mesh with texture")
    mcubes_resolutio=256, #help="mcubes resolution for extracting mesh")
    decimate_target=5e4, #help="target face number for mesh decimation")
    dmtet=False,
    tet_grid_size=128, #help="tet grid size")
    init_with='', #help="ckpt to init dmtet")
    lock_geo=False, # help="disable dmtet to learn geometry")

    ## Perp-Neg options
    perpneg=False, # help="use perp_neg")
    negative_w=-2, # help="The scale of the weights of negative prompts. A larger value will help to avoid the Janus problem, but may cause flat faces. Vary between 0 to -4, depending on the prompt")
    front_decay_factor=2, #help="decay factor for the front prompt")
    side_decay_factor=10, #help="decay factor for the side prompt")

    ### training options
    iters=10000, #help="training iters")
    lr=1e-3, #help="max learning rate")
    ckpt='latest', # help="possible options are ['latest', 'scratch', 'best', 'latest_model']")
    cuda_ray=False, #help="use CUDA raymarching instead of pytorch")
    taichi_ray=False,
    max_steps=1024, # help="max num steps sampled per ray (only valid when using --cuda_ray)")
    num_steps=64, #help="num steps sampled per ray (only valid when not using --cuda_ray)")
    upsample_steps=32, #help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    update_extra_interval=16, #help="iter interval to update extra status (only valid when using --cuda_ray)")
    max_ray_batch=4096, #help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    latent_iter_ratio=0.2, #help="training iters that only use albedo shading")
    albedo_iter_ratio=0, #help="training iters that only use albedo shading")
    min_ambient_ratio=0.1, #help="minimum ambient ratio to use in lambertian shading")
    textureless_ratio=0.2, #help="ratio of textureless shading")
    jitter_pose=False, #action='store_true', help="add jitters to the randomly sampled camera poses")
    jitter_center=0.2, #help="amount of jitter to add to sampled camera pose's center (camera location)")
    jitter_target=0.2, #help="amount of jitter to add to sampled camera pose's target (i.e. 'look-at')")
    jitter_up=0.02, #help="amount of jitter to add to sampled camera pose's up-axis (i.e. 'camera roll')")
    uniform_sphere_rate=0, #help="likelihood of sampling camera location uniformly on the sphere surface area")
    grad_clip=-1, #help="clip grad of all grad to this limit, negative value disables it")
    grad_clip_rgb=-1, #help="clip grad of rgb space grad to this limit, negative value disables it")
    # model options
    bg_radius=1.4, #help="if positive, use a background model at sphere(bg_radius)")
    density_activation='exp',# choices=['softplus', 'exp'], help="density activation function")
    density_thresh=10, #help="threshold for density grid to be occupied")
    blob_density=5, #help="max (center) density for the density blob")
    blob_radius=0.2, #help="control the radius for the density blob")
    # network backbone
    backbone='grid', #choices=['grid_tcnn', 'grid', 'vanilla', 'grid_taichi'], help="nerf backbone")
    optim='adan', #choices=['adan', 'adam'], help="optimizer")
    sd_version='2.1', #choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    hf_key=None, #help="hugging face Stable diffusion model key")
    # try this if CUDA OOM
    fp16=False, #help="use float16 for training")
    vram_O=False, # help="optimization for low VRAM usage")
    # rendering resolution in training, increase these for better quality / decrease these if CUDA OOM even if --vram_O enabled.
    w=64,  #help="render width for NeRF in training")
    h=64, #help="render height for NeRF in training")
    known_view_scale=1.5, #help="multiply --h/w by this for known view rendering")
    known_view_noise_scale=2e-3, #help="random camera noise added to rays_o and rays_d")
    dmtet_reso_scale=8, #help="multiply --h/w by this for dmtet finetuning")
    batch_size=1, #help="images to render per batch using NeRF")

    ### dataset options
    bound=1, #help="assume the scene is bounded in box(-bound, bound)")
    dt_gamma=0, #help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    min_near=0.01, #help="minimum near distance for camera")

    radius_range=[3.0, 3.5], #help="training camera radius range")
    theta_range=[45, 105], #help="training camera range along the polar angles (i.e. up and down). See advanced.md for details.")
    phi_range=[-180, 180], #help="training camera range along the azimuth angles (i.e. left and right). See advanced.md for details.")
    fovy_range=[10, 30], #help="training camera fovy range")

    default_radius=3.2, #help="radius for the default view")
    default_polar=90, #help="polar for the default view")
    default_azimuth=0, #help="azimuth for the default view")
    default_fovy=20, #help="fovy for the default view")

    progressive_view=False, #action='store_true', help="progressively expand view sampling range from default to full")
    progressive_view_init_ratio=0.2, #help="initial ratio of final range, used for progressive_view")
    
    progressive_level=False, #help="progressively increase gridencoder's max_level")

    angle_overhead=30, #help="[0, angle_overhead] is the overhead region")
    angle_front=60, #help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    t_range=[0.02, 0.98], #help="stable diffusion time steps range")
    dont_override_stuff=False, #',action='store_true', help="Don't override t_range, etc.")


    ### regularizations
    lambda_entropy=1e-3, #help="loss scale for alpha entropy")
    lambda_opacity=0, #help="loss scale for alpha value")
    lambda_orient=1e-2, #help="loss scale for orientation")
    lambda_tv=0, #help="loss scale for total variation")
    lambda_wd=0, #help="loss scale")

    lambda_mesh_normal=0.5, #help="loss scale for mesh normal smoothness")
    lambda_mesh_laplacian=0.5, #help="loss scale for mesh laplacian")

    lambda_guidance=1, #help="loss scale for SDS")
    lambda_rgb=1000, #help="loss scale for RGB")
    lambda_mask=500, #help="loss scale for mask (alpha)")
    lambda_normal=0, #help="loss scale for normal map")
    lambda_depth=10, #help="loss scale for relative depth")
    lambda_2d_normal_smooth=0, #help="loss scale for 2D normal image smoothness")
    lambda_3d_normal_smooth=0, #help="loss scale for 3D normal image smoothness")

    
    save_guidance=False, #action='store_true', help="save images of the per-iteration NeRF renders, added noise, denoised (i.e. guidance), fully-denoised. Useful for debugging, but VERY SLOW and takes lots of memory!")
    save_guidance_interval=10, #help="save guidance every X step")

    
    gui=False, #action='store_true', help="start a GUI")
    W=800, #help="GUI width")
    H=800, #help="GUI height")
    radius=5, #help="default GUI camera radius from center")
    fovy=20, #help="default GUI camera fovy")
    light_theta=60, #help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    light_phi=0, #help="default GUI light direction in [0, 360), azimuth")
    max_spp=1, #help="GUI rendering max sample per pixel")

    zero123_config='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml',#, help="config file for zero123")
    zero123_ckpt='pretrained/zero123/zero123-xl.ckpt', #, help="ckpt for zero123")
    zero123_grad_scale='angle', #, help="whether to scale the gradients based on 'angle' or 'None'")

    dataset_size_train=100, #help="Length of train dataset i.e. # of iterations per epoch")
    dataset_size_valid=8, #help="# of frames to render in the turntable video in validation")
    dataset_size_test=100, #help="# of frames to render in the turntable video at test time")

    exp_start_iter=None, #', te, help="start iter # for experiment, to calculate progressive_view and progressive_level")
    exp_end_iter=None,#', typ help="end iter # for experiment, to calculate progressive_view and progressive_level")

    # 以下に必要なすべてのパラメータを続けてください
):
    
    opt = Namespace(
    workspace=workspace, #workspace path
    file=file, 
    text=text, 
    negative=negative,

    # iters=iters, #help="training iterations")
    # lr=lr, #help="learning rate")
    # dt_lr=dt_lr, #help="dt learning rate")
    # parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray")
    # parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla")
    test=test,
    six_views=six_views,
    eval_interval=eval_interval,
    test_interval=test_interval,
    seed=seed,
    image=image,
    image_config=image_config,
    known_view_interval=known_view_interval,
    IF=IF,
    guidance=guidance,
    guidance_scale=guidance_scale,
    save_mesh=save_mesh, #"export an obj mesh with texture")
    mcubes_resolutio=mcubes_resolutio, #help="mcubes resolution for extracting mesh")
    decimate_target=decimate_target, #help="target face number for mesh decimation")
    dmtet=dmtet, #help="use dmtet")
    tet_grid_size=tet_grid_size, #help="tet grid size")
    init_with=init_with, #help="ckpt to init dmtet")
    lock_geo=lock_geo, # help="disable dmtet to learn geometry")

    ## Perp-Neg options
    perpneg=perpneg, # help="use perp_neg")
    negative_w=negative_w, # help="The scale of the weights of negative prompts. A larger value will help to avoid the Janus problem, but may cause flat faces. Vary between 0 to -4, depending on the prompt")
    front_decay_factor=front_decay_factor, #help="decay factor for the front prompt")
    side_decay_factor=side_decay_factor, #help="decay factor for the side prompt")

    ### training options
    iters=iters, #help="training iters")
    lr=lr, #help="max learning rate")
    ckpt=ckpt, # help="possible options are ['latest', 'scratch', 'best', 'latest_model']")
    cuda_ray=cuda_ray, #help="use CUDA raymarching instead of pytorch")
    taichi_ray=taichi_ray, #help="use taichi raymarching instead of pytorch")
    max_steps=max_steps, # help="max num steps sampled per ray (only valid when using --cuda_ray)")
    num_steps=num_steps, #help="num steps sampled per ray (only valid when not using --cuda_ray)")
    upsample_steps=upsample_steps, #help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    update_extra_interval=update_extra_interval, #help="iter interval to update extra status (only valid when using --cuda_ray)")
    max_ray_batch=max_ray_batch, #help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    latent_iter_ratio=latent_iter_ratio, #help="training iters that only use albedo shading")
    albedo_iter_ratio=albedo_iter_ratio, #help="training iters that only use albedo shading")
    min_ambient_ratio=min_ambient_ratio, #help="minimum ambient ratio to use in lambertian shading")
    textureless_ratio=textureless_ratio, #help="ratio of textureless shading")
    jitter_pose=jitter_pose, #action='store_true', help="add jitters to the randomly sampled camera poses")
    jitter_center=jitter_center, #help="amount of jitter to add to sampled camera pose's center (camera location)")
    jitter_target=jitter_target, #help="amount of jitter to add to sampled camera pose's target (i.e. 'look-at')")
    jitter_up=jitter_up, #help="amount of jitter to add to sampled camera pose's up-axis (i.e. 'camera roll')")
    uniform_sphere_rate=uniform_sphere_rate, #help="likelihood of sampling camera location uniformly on the sphere surface area")
    grad_clip=grad_clip, #help="clip grad of all grad to this limit, negative value disables it")
    grad_clip_rgb=grad_clip_rgb, #help="clip grad of rgb space grad to this limit, negative value disables it")
    # model options
    bg_radius=bg_radius, #help="if positive, use a background model at sphere(bg_radius)")
    density_activation=density_activation,# choices=['softplus', 'exp'], help="density activation function")
    density_thresh=density_thresh, #help="threshold for density grid to be occupied")
    blob_density=blob_density, #help="max (center) density for the density blob")
    blob_radius=blob_radius, #help="control the radius for the density blob")
    # network backbone
    backbone=backbone, #choices=['grid_tcnn', 'grid', 'vanilla', 'grid_taichi'], help="nerf backbone")
    optim=optim, #choices=['adan', 'adam'], help="optimizer")
    sd_version=sd_version, #choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    hf_key=hf_key, #help="hugging face Stable diffusion model key")
    # try this if CUDA OOM
    fp16=fp16, #help="use float16 for training")
    vram_O=vram_O, # help="optimization for low VRAM usage")
    # rendering resolution in training, increase these for better quality / decrease these if CUDA OOM even if --vram_O enabled.
    w=w,  #help="render width for NeRF in training")
    h=h, #help="render height for NeRF in training")
    known_view_scale=known_view_scale, #help="multiply --h/w by this for known view rendering")
    known_view_noise_scale=known_view_noise_scale, #help="random camera noise added to rays_o and rays_d")
    dmtet_reso_scale=dmtet_reso_scale, #help="multiply --h/w by this for dmtet finetuning")
    batch_size=batch_size, #help="images to render per batch using NeRF")

    ### dataset options
    bound=bound, #help="assume the scene is bounded in box(-bound, bound)")
    dt_gamma=dt_gamma , #help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    min_near=min_near, #help="minimum near distance for camera")

    radius_range=radius_range, #help="training camera radius range")
    theta_range=theta_range, #help="training camera range along the polar angles (i.e. up and down). See advanced.md for details.")
    phi_range=phi_range, #help="training camera range along the azimuth angles (i.e. left and right). See advanced.md for details.")
    fovy_range=fovy_range, #help="training camera fovy range")

    default_radius=default_radius, #help="radius for the default view")
    default_polar=default_polar, #help="polar for the default view")
    default_azimuth=default_azimuth, #help="azimuth for the default view")
    default_fovy=default_fovy, #help="fovy for the default view")

    progressive_view=progressive_view, #action='store_true', help="progressively expand view sampling range from default to full")
    progressive_view_init_ratio=progressive_view_init_ratio, #help="initial ratio of final range, used for progressive_view")
    
    progressive_level=progressive_level, #help="progressively increase gridencoder's max_level")

    angle_overhead=angle_overhead, #help="[0, angle_overhead] is the overhead region")
    angle_front=angle_front, #help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    t_range=t_range, #help="stable diffusion time steps range")
    dont_override_stuff=dont_override_stuff, #',action='store_true', help="Don't override t_range, etc.")


    ### regularizations
    lambda_entropy=lambda_entropy, #help="loss scale for alpha entropy")
    lambda_opacity=lambda_opacity, #help="loss scale for alpha value")
    lambda_orient=lambda_orient, #help="loss scale for orientation")
    lambda_tv=lambda_tv, #help="loss scale for total variation")
    lambda_wd=lambda_wd, #help="loss scale")

    lambda_mesh_normal=lambda_mesh_normal, #help="loss scale for mesh normal smoothness")
    lambda_mesh_laplacian=lambda_mesh_laplacian, #help="loss scale for mesh laplacian")

    lambda_guidance=lambda_guidance, #help="loss scale for SDS")
    lambda_rgb=lambda_rgb, #help="loss scale for RGB")
    lambda_mask=lambda_mask, #help="loss scale for mask (alpha)")
    lambda_normal=lambda_normal, #help="loss scale for normal map")
    lambda_depth=lambda_depth, #help="loss scale for relative depth")
    lambda_2d_normal_smooth=lambda_2d_normal_smooth, #help="loss scale for 2D normal image smoothness")
    lambda_3d_normal_smooth=lambda_3d_normal_smooth, #help="loss scale for 3D normal image smoothness")

    
    save_guidance=save_guidance, #action='store_true', help="save images of the per-iteration NeRF renders, added noise, denoised (i.e. guidance), fully-denoised. Useful for debugging, but VERY SLOW and takes lots of memory!")
    save_guidance_interval=save_guidance_interval, #help="save guidance every X step")

    
    gui=gui, #action='store_true', help="start a GUI")
    W=W, #help="GUI width")
    H=H, #help="GUI height")
    radius=radius, #help="default GUI camera radius from center")
    fovy=fovy, #help="default GUI camera fovy")
    light_theta=light_theta, #help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    light_phi=light_phi, #help="default GUI light direction in [0, 360), azimuth")
    max_spp=max_spp, #help="GUI rendering max sample per pixel")

    zero123_config=zero123_config,
    zero123_ckpt=zero123_ckpt,
    zero123_grad_scale=zero123_grad_scale, #, help="whether to scale the gradients based on 'angle' or 'None'")

    dataset_size_train=dataset_size_train, #help="Length of train dataset i.e. # of iterations per epoch")
    dataset_size_valid=dataset_size_valid, #help="# of frames to render in the turntable video in validation")
    dataset_size_test=dataset_size_test, #help="# of frames to render in the turntable video at test time")

    exp_start_iter=exp_start_iter, #', te, help="start iter # for experiment, to calculate progressive_view and progressive_level")
    exp_end_iter=exp_end_iter,#', typ help="end iter # for experiment, to calculate progressive_view and progressive_level")
    )
    # 以下のコードはそのまま残します...

    # See https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
    class LoadFromFile (argparse.Action):
        def __call__ (self, parser, namespace, values, option_string = None):
            with values as f:
                # parse arguments in the file and store them in the target namespace
                parser.parse_args(f.read().split(), namespace)

    parser = argparse.ArgumentParser()

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True

    elif opt.O2:
        opt.fp16 = True
        opt.backbone = 'vanilla'
        opt.progressive_level = True

    if opt.IF:
        if 'SD' in opt.guidance:
            opt.guidance.remove('SD')
            opt.guidance.append('IF')
        opt.latent_iter_ratio = 0 # must not do as_latent

    opt.images, opt.ref_radii, opt.ref_polars, opt.ref_azimuths, opt.zero123_ws = [], [], [], [], []
    opt.default_zero123_w = 1

    opt.exp_start_iter = opt.exp_start_iter or 0
    opt.exp_end_iter = opt.exp_end_iter or opt.iters

    # parameters for image-conditioned generation
    if opt.image is not None or opt.image_config is not None:

        if opt.text is None:
            # use zero123 guidance model when only providing image
            opt.guidance = ['zero123']
            if not opt.dont_override_stuff:
                opt.fovy_range = [opt.default_fovy, opt.default_fovy] # fix fov as zero123 doesn't support changing fov
                opt.guidance_scale = 5
                opt.lambda_3d_normal_smooth = 10
        else:
            # use stable-diffusion when providing both text and image
            opt.guidance = ['SD', 'clip']
            
            if not opt.dont_override_stuff:
                opt.guidance_scale = 10
                opt.t_range = [0.2, 0.6]
                opt.known_view_interval = 2
                opt.lambda_3d_normal_smooth = 20
            opt.bg_radius = -1

        # smoothness
        opt.lambda_entropy = 1
        opt.lambda_orient = 1

        # latent warmup is not needed
        opt.latent_iter_ratio = 0
        if not opt.dont_override_stuff:
            opt.albedo_iter_ratio = 0
            
            # make shape init more stable
            opt.progressive_view = True
            opt.progressive_level = True

        if opt.image is not None:
            opt.images += [opt.image]
            opt.ref_radii += [opt.default_radius]
            opt.ref_polars += [opt.default_polar]
            opt.ref_azimuths += [opt.default_azimuth]
            opt.zero123_ws += [opt.default_zero123_w]

        if opt.image_config is not None:
            # for multiview (zero123)
            conf = pd.read_csv(opt.image_config, skipinitialspace=True)
            opt.images += list(conf.image)
            opt.ref_radii += list(conf.radius)
            opt.ref_polars += list(conf.polar)
            opt.ref_azimuths += list(conf.azimuth)
            opt.zero123_ws += list(conf.zero123_weight)
            if opt.image is None:
                opt.default_radius = opt.ref_radii[0]
                opt.default_polar = opt.ref_polars[0]
                opt.default_azimuth = opt.ref_azimuths[0]
                opt.default_zero123_w = opt.zero123_ws[0]

    # reset to None
    if len(opt.images) == 0:
        opt.images = None

    # default parameters for finetuning
    if opt.dmtet:

        opt.h = int(opt.h * opt.dmtet_reso_scale)
        opt.w = int(opt.w * opt.dmtet_reso_scale)
        opt.known_view_scale = 1

        if not opt.dont_override_stuff:            
            opt.t_range = [0.02, 0.50] # ref: magic3D

        if opt.images is not None:

            opt.lambda_normal = 0
            opt.lambda_depth = 0

            if opt.text is not None and not opt.dont_override_stuff:
                opt.t_range = [0.20, 0.50]

        # assume finetuning
        opt.latent_iter_ratio = 0
        opt.albedo_iter_ratio = 0
        opt.progressive_view = False
        # opt.progressive_level = False

    # record full range for progressive view expansion
    if opt.progressive_view:
        if not opt.dont_override_stuff:
            # disable as they disturb progressive view
            opt.jitter_pose = False
            
        opt.uniform_sphere_rate = 0
        # back up full range
        opt.full_radius_range = opt.radius_range
        opt.full_theta_range = opt.theta_range
        opt.full_phi_range = opt.phi_range
        opt.full_fovy_range = opt.fovy_range

    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'grid_tcnn':
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == 'grid_taichi':
        opt.cuda_ray = False
        opt.taichi_ray = True
        import taichi as ti
        from nerf.network_grid_taichi import NeRFNetwork
        taichi_half2_opt = True
        taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
        if taichi_half2_opt:
            taichi_init_args["half2_vectorization"] = True
        ti.init(**taichi_init_args)
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)

    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt).to(device)

    if opt.dmtet and opt.init_with != '':
        if opt.init_with.endswith('.pth'):
            # load pretrained weights to init dmtet
            state_dict = torch.load(opt.init_with, map_location=device)
            model.load_state_dict(state_dict['model'], strict=False)
            if opt.cuda_ray:
                model.mean_density = state_dict['mean_density']
            model.init_tet()
        else:
            # assume a mesh to init dmtet (experimental, not working well now!)
            import trimesh
            mesh = trimesh.load(opt.init_with, force='mesh', skip_material=True, process=False)
            model.init_tet(mesh=mesh)

    print(model)

    if opt.six_views:
        guidance = None # no need to load guidance model at test

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        test_loader = NeRFDataset(opt, device=device, type='six_views', H=opt.H, W=opt.W, size=6).dataloader(batch_size=1)
        trainer.test(test_loader, write_video=False)

        if opt.save_mesh:
            trainer.save_mesh()

    elif opt.test:
        guidance = None # no need to load guidance model at test

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)
            trainer.test(test_loader)

            if opt.save_mesh:
                trainer.save_mesh()

    else:

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()

        if opt.optim == 'adan':
            from optimizer import Adan
            # Adan usually requires a larger LR
            optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else: # adam
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        if opt.backbone == 'vanilla':
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
        else:
            scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
            # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        guidance = nn.ModuleDict()

        if 'SD' in opt.guidance:
            from guidance.sd_utils import StableDiffusion
            guidance['SD'] = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key, opt.t_range)

        if 'IF' in opt.guidance:
            from guidance.if_utils import IF
            guidance['IF'] = IF(device, opt.vram_O, opt.t_range)

        if 'zero123' in opt.guidance:
            from guidance.zero123_utils import Zero123
            guidance['zero123'] = Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)

        if 'clip' in opt.guidance:
            from guidance.clip_utils import CLIP
            guidance['clip'] = CLIP(device)

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)

        trainer.default_view_data = train_loader._data.get_default_view_data()

        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader(batch_size=1)
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, test_loader, max_epoch)

            if opt.save_mesh:
                trainer.save_mesh()
