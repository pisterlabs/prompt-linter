import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_cond_uncond_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

def create_argparser():
    '''
    计算差值用到的参数
    '''
    calc_args=dict(
        n_samples_per_time=1,
        num_classes=1000,
        interval=100,
    )
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=16,
        use_ddim=False,
        cond_model_path="",
        uncond_model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(calc_args)
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def model_load_state_dict(model,model_path,use_fp16):
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    
    if use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model
def sample_fn_return_xt_list_by_psample(
        diffusion,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        xt_list = []
        for sample in diffusion.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            xt_list.append(sample['sample'])
        return xt_list
def convert_x_t_to_images_for_vaild(x_t,name='test.png'):
    import cv2
    img=(x_t[0]*127.5)+127.5
    img=img.cpu().detach().int().clamp(0, 255).permute(1, 2, 0)
    img=img.numpy()
    cv2.imwrite(name,img)
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    cond_model,uncond_model, diffusion = create_cond_uncond_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    #load cond and load uncond
    logger.log("loading cond_model and uncond_model...")
    cond_model=model_load_state_dict(cond_model,args.cond_model_path,args.use_fp16)
    uncond_model=model_load_state_dict(uncond_model,args.uncond_model_path,args.use_fp16)
    
    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    
    logger.log("calu diff...")
    # def cond_fn(x, t, y=None):
    #     assert y is not None
    #     with th.enable_grad():
    #         x_in = x.detach().requires_grad_(True)
    #         logits = classifier(x_in, t)
    #         log_probs = F.log_softmax(logits, dim=-1)
    #         selected = log_probs[range(len(logits)), y.view(-1)]
    #         return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
    def cond_model_fn(x, t, y=None):
        assert y is not None
        return cond_model(x, t, y)
    def uncond_model_fn(x, t, y=None):
        assert y is not None
        return uncond_model(x, t, None)
        
    b,c,h,w=1,3,256,256
    g_shape = (b,c,h,w)
    error_over_time={}
    for i in range(args.n_samples_per_time):
        c=i%args.num_classes
        print('run {} sample. used c is {} '.format(i,c))
        model_kwargs = {}
        model_kwargs["y"] = th.tensor([c], dtype=th.long, device=dist_util.dev()).view(1,)
        
        #使用uncond的方法进行采样,得到Xt->X0的全部过程.
        #即只和时间有关系，计算eps(x, NULL)中的 x_T ~ x_0
        '''
        这部分来得到x0~X_T 是通过计算eps(x, NULL)吗？不太确定。
        p_sample.pdf 里面写到：
        x_0_to_T=gd.sample(c,cond_scale=0)
        ->
        cfg: 
        model(x_t,t,null)+cond_scale*(model(x,t,c)-model(x,t,null))
        =model(x_t,t,null)
        
        所以暂时写做:
        x0~X_T = uncond_model(x_t, t, y=None)# uncond_model: 256x256_diffusion_uncond.pt
        '''
        xt_list=sample_fn_return_xt_list_by_psample(
            diffusion,
            uncond_model_fn,
            (1, 3, 256, 256),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None,    #对应 eps(x, NULL)
            device=dist_util.dev(),
        )
        # copy from openai. GaussianDiffusion.p_sample_loop_progressive: line 544
        indices = list(range(diffusion.num_timesteps))[::-1] #T,T-1,...,1,0
        for i in indices:
            
            x_t=xt_list[diffusion.num_timesteps-i-1]
            t=th.tensor([i] * g_shape[0], device=dist_util.dev())
            if t not in error_over_time.keys():
                error_over_time[int(t[0].cpu())]=[]
            p_t=classifier(x_t,diffusion._scale_timesteps(t)) #这里diffusion._scale_timesteps(t)=t
            p_t=th.nn.functional.softmax(p_t, dim=-1)
            
            '''
            这里不太清楚,是用哪个模型? 
            暂时写作:
            eps(x|NULL)=uncond_model(x_t,t,None) # uncond_model: 256x256_diffusion_uncond.pt
            '''
            with th.no_grad():
                eps_null=uncond_model(x_t,t,None)
            mean_eps_c=th.zeros_like(eps_null)
            interval=args.interval
            #为了加速，将全部数据叠加在bs上，然后一次性计算
            x_t=x_t.repeat(interval,1,1,1)
            t=t.repeat(interval).view(-1)
            for c_i in range(args.num_classes//interval):
                assert args.num_classes%interval==0

                model_kwargs["y"] =th.arange(interval*c_i,interval*(c_i+1),dtype=th.long, device=dist_util.dev()).view(-1)
                '''
                这里也不太明白,是用哪个模型? 
                暂时写作:
                eps(x|c)=cond_model(x_t,t,y) # cond_model: 256x256_diffusion.pt
                '''
                with th.no_grad():
                    eps_c=cond_model(x_t,t,**model_kwargs)
                score=p_t[0][interval*c_i:interval*(c_i+1)].view(interval,1,1,1)
                mean_eps_c+=th.sum(eps_c*score,dim=0)
            with th.no_grad():
                error=th.nn.functional.l1_loss(eps_null,mean_eps_c)
            error_over_time[int(t[0].cpu())].append(error.cpu())
            print('error:',error)
              
    print(error_over_time)
    axis_x=[]
    axis_y=[]
    for t,errors in error_over_time.items():
        axis_x.append(int(t))
        axis_y.append(np.mean(errors))
        #y_error_bar=np.std(errors)
    plt.plot(axis_x,axis_y)
    plt.gca().invert_xaxis()
    
    
    plt.savefig('test.png')
        # print(len(xt_list))
        # convert_x_t_to_images_for_vaild(xt_list[0],name='test2_{}.png'.format('0'))
        # convert_x_t_to_images_for_vaild(xt_list[-1],name='test2_{}.png'.format('-1'))
        
        
        
        
if __name__ == "__main__":
    main()