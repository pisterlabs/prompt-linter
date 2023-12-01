import torch
from openai_realted.unet_from_openai import EncoderUNetModel

import torch
from models.diff_model import diff_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import click
import argparse
import numpy as np
from tqdm import tqdm
from pytorch_lightning import seed_everything
def config():
    # several hyperparameters for models
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument("--loadDir" , type=str,default='pretrained_models', help="Location of the models to load in.", required=False)
    parser.add_argument("--loadFile", type=str,default='model_438e_550000s.pkl', help="Name of the .pkl model file to load in. Ex: model_358e_450000s.pkl", required=False)
    parser.add_argument("--loadDefFile", type=str,default='model_params_438e_550000s.json', help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl", required=False)

    # Generation parameters
    parser.add_argument("--step_size", type=int, default=1, help="Step size when generating. A step size of 10 with a model trained on 1000 steps takes 100 steps to generate. Lower is faster, but produces lower quality images.", required=False)
    parser.add_argument("--DDIM_scale", type=int, default=0, help="Must be >= 0. When this value is 0, DDIM is used. When this value is 1, DDPM is used. A low scalar performs better with a high step size and a high scalar performs better with a low step size.", required=False)
    parser.add_argument("--device", type=str, default="cuda", help="Device to put the model on. use \"gpu\" or \"cpu\".", required=False)
    parser.add_argument("--guidance", type=int, default=4, help="Classifier guidance scale which must be >= 0. The higher the value, the better the image quality, but the lower the image diversity.", required=False)
    parser.add_argument("--class_label", type=int, default=1, help="0-indexed class value. Use -1 for a random class and any other class value >= 0 for the other classes. FOr imagenet, the class value range from 0 to 999 and can be found in data/class_information.txt", required=False)
    parser.add_argument("--corrected", type=bool, default=False, help="True to put a limit on generation, False to not put a litmit on generation. If the model is generating images of a single color, then you may need to set this flag to True. Note: This restriction is usually needed when generating long sequences (low step size) Note: With a higher guidance w, the correction usually messes up generation.", required=False)

    # Output parameters
    parser.add_argument("--out_imgname", type=str, default="fig1.png", help="Name of the file to save the output image to.", required=False)
    parser.add_argument("--out_gifname", type=str, default="diffusion1.gif", help="Name of the file to save the output image to.", required=False)
    parser.add_argument("--gif_fps", type=int, default=10, help="FPS for the output gif.", required=False)

    args = parser.parse_args()
    return args
def create_classifier(state_dict_path):
    #分类器的初始化参数
    attention_ds = []
    classifier_attention_resolutions= "16,8,4"
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(32 // int(res))
    channel_mult= (1, 2, 3, 4)
    classifier=EncoderUNetModel(
        image_size=64,
        in_channels=3,
        model_channels=128,
        out_channels=1000,
        num_res_blocks=4,
        use_fp16=False,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        num_head_channels=64,
        use_scale_shift_norm=True,
        use_new_attention_order=False,
        resblock_updown=True,
        pool='attention',
    )
    classifier.load_state_dict(torch.load(state_dict_path),strict=True)
    return classifier
def convert_x_t_to_images_for_vaild(x_t,name='test.png'):
    import cv2
    img=(x_t[0]*127.5)+127.5
    img=img.cpu().detach().int().clamp(0, 255).permute(1, 2, 0)
    img=img.numpy()
    cv2.imwrite(name,img)
def infer(args):
    loadDir=args.loadDir
    loadFile=args.loadFile
    loadDefFile=args.loadDefFile
    step_size=args.step_size
    DDIM_scale=args.DDIM_scale
    device=args.device
    w=args.guidance
    class_label=args.class_label
    corrected=args.corrected
    out_imgname=args.out_imgname
    out_gifname=args.out_gifname
    gif_fps=args.gif_fps
    
    
    ### Model Creation

    # Create a dummy model
    """
    这里的设置是默认的,后续会通过loadModel使用的json进行修改
    step_size为间隔,
    step_size=1 -->T=1000
    step_size=20 -->T=50
    """
    
    
    
    ############################################################### define all important args:################################################################################
    '''
    define all important args:
    '''
    
    num_classes=1000
    classifier_path='pretrained_models/64x64_classifier.pt'#from guided_diffusion : https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt
    seed=42
    
    #### 常修改参数
    interval=1000 #为了迅速计算每个类别的贡献，将其叠加至bs维度，叠加的长度为interval,interval=1000约50G显存
    n_samples_per_time=1000
    step_size=20 #实际迭代步长为 1000//step_size
    
    error_over_time={}
   # error_over_time=np.zeros(shape=(1000,1000))
    ################################################################ define diffusion model################################################################################
    model=diff_model(inCh=3, embCh=3, chMult=1, num_blocks=1,
                blk_types=["res", "res"], T=100000, beta_sched="cosine", t_dim=100, device=device, 
                c_dim=100, num_classes=1000, 
                atn_resolution=16, dropoutRate=0.0, 
                step_size=step_size, DDIM_scale=DDIM_scale,
    )
    model.loadModel(loadDir, loadFile, loadDefFile)
    ################################################################ define classifier################################################################################
    classifier=create_classifier(classifier_path).to(device)
        
    ###############################################################define loss #######################################################################################
    L1_loss=torch.nn.L1Loss()
    L1_loss.to('cuda')
    ############################################################### define random seed####################################################################################

    #for reproduce the result
    seed_everything(seed)
    import time
    for i in range(n_samples_per_time):
        c=i%num_classes
        # Sample the model
        print('run {} sample. used c is {} '.format(i,c))
        noise, x_0_to_T,t_list_ddpm = model.sample_imgs(1, c, w, True, False, True, corrected)
        '''
        x_0_to_T=[x_t,x_t-1,x_t-2...x_0]
        '''
        for t_i in tqdm(range(0,min(1001,len(x_0_to_T)))):
            
            x_t=x_0_to_T[t_i]
            t=t_list_ddpm[t_i]
            if t not in error_over_time.keys():
                error_over_time[t]=[]
                #error_over_time[t]=torch.zeros(size=(1,))
            if not isinstance(t,torch.Tensor):
                t=torch.tensor(t).repeat(x_t.shape[0]).to(torch.long)
            x_t = x_t.to(device)
            t = t.to(device)
            p_t=classifier(x_t,t)
            p_t=torch.nn.functional.softmax(p_t, dim=-1)
            torch_zero=torch.tensor([0])
            torch_one=torch.tensor([1])
            #无条件噪声eps_t
            with torch.no_grad():
                eps_t_null,_=model.forward(x_t,t,torch_zero,torch_one)#return noise_t_un, v_t_un
           
            #有条件噪声eps_t_cond
            mean_eps_t_cond=torch.zeros_like(eps_t_null)
            #t=t.repeat(interval).view(-1)
            x_t=x_t.repeat(interval,1,1,1)
            
            for c_i in range(num_classes//interval): #[0...9,10..19]
                assert num_classes%interval==0
                input_condition=torch.arange(interval*c_i,interval*(c_i+1)).view(-1)
                
                with torch.no_grad():
                    eps_t_c,_=model.forward(x_t,t,input_condition,torch_zero)
                # for z in range(interval):
                #     mean_eps_t_cond+=eps_t_c[z]*p_t[0][interval*c_i+z]
                score=p_t[0][interval*c_i:interval*(c_i+1)].view(interval,1,1,1)
                mean_eps_t_cond+=torch.sum(eps_t_c*score,dim=0)
                # with torch.no_grad():
                #     eps_t_c,_=model.forward(x_t,t,torch.tensor([int(c_i)]),torch_zero)
                #mean_eps_t_cond+=eps_t_c*p_t[0][c_i]
            
            with torch.no_grad():
                error=L1_loss(eps_t_null,mean_eps_t_cond)
           
            error_over_time[int(t[0].cpu())].append(error.cpu())
           
    print(error_over_time)
    axis_x=[]
    axis_y=[]
    for t,errors in error_over_time.items():
        axis_x.append(int(t))
        axis_y.append(np.mean(errors))
        #y_error_bar=np.std(errors)
    plt.plot(axis_x,axis_y)
    plt.gca().invert_xaxis()
    
    plt.savefig('1res_seed_{}_samples_{}_step_{}_interval_{}.png'.format(seed,n_samples_per_time,step_size,interval))
    #model.calc_diff(1)
    # Convert the sample image to 0->255
    # and show it
    # plt.close('all')
    # plt.axis('off')
    # noise = torch.clamp(noise.cpu().detach().int(), 0, 255)
    # for img in noise:
    #     plt.imshow(img.permute(1, 2, 0))
    #     plt.savefig(out_imgname, bbox_inches='tight', pad_inches=0, )
    #     plt.show()

    # # Image evolution gif
    # plt.close('all')
    # fig, ax = plt.subplots()
    # ax.set_axis_off()
    # for i in range(0, len(imgs)):
    #     title = plt.text(imgs[i].shape[0]//2, -5, f"t = {i}", ha='center')
    #     imgs[i] = [plt.imshow(imgs[i], animated=True), title]
    # animate = animation.ArtistAnimation(fig, imgs, interval=1, blit=True, repeat_delay=1000)
    # animate.save(out_gifname, writer=animation.PillowWriter(fps=gif_fps))
    
    
    
    
    
if __name__ == '__main__':
    args=config()
    infer(args)