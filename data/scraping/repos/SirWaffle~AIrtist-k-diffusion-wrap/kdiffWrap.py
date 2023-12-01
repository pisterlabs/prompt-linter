import gc
import io
from logging import exception
import math
import sys

sys.path.append('./../k-diffusion')
sys.path.append('./../guided-diffusion')
sys.path.append('./../v-diffusion-pytorch')


import k_diffusion as K
from PIL import Image
import torch
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
from loguru import logger
from torchvision.utils import make_grid
from torch import autocast
from contextlib import contextmanager, nullcontext
import torch.onnx 
from torch import nn

#from ldm.util import instantiate_from_config
import random

import cutouts
import paramsGen
import utilFuncs
import clipWrap
import modelWrap
import CompVisRDMModel
import OpenAIUncondModel
import CompVisSDModel
import CompVisSDOnnxModel
import onnxSampling

class KDiffWrap:
    def __init__(self, deviceName:str = 'cuda:0'):
        #torch
        self.torchDevice = torch.device(deviceName)
        print('Using device:', self.torchDevice, flush=True)

    def DeleteModel(self, model:modelWrap.ModelWrap):
        model.model = model.model.cpu()
        del model.model
        del model
        gc.collect()
        return None

    def DeleteClipModel(self, model:clipWrap.ClipWrap):
        model.model = model.model.cpu()
        del model.model
        del model
        gc.collect()
        return None        


    def CreateModel(self, modelName:str) ->modelWrap.ModelWrap:
        
        if modelName.lower() == "sd-v1-3-full-ema":
            modelwrapper = CompVisSDModel.CompVisSDModel()
            modelwrapper.model_path = "E:/MLModels/stableDiffusion/sd-v1-3-full-ema.ckpt" 
        elif modelName.lower() == "sd-v1-4":
            modelwrapper = CompVisSDModel.CompVisSDModel()
            modelwrapper.model_path = "E:/MLModels/stableDiffusion/sd-v1-4.ckpt"  
        elif modelName.lower() == "sd-v1-3":
            modelwrapper = CompVisSDModel.CompVisSDModel()
            modelwrapper.model_path = "E:/MLModels/stableDiffusion/sd-v1-3.ckpt"  
        elif modelName.lower() == "sd-v1-2":
            modelwrapper = CompVisSDModel.CompVisSDModel()
            modelwrapper.model_path = "E:/MLModels/stableDiffusion/sd-v1-2.ckpt"  
        elif modelName.lower() == "sd-v1-1":
            modelwrapper = CompVisSDModel.CompVisSDModel()
            modelwrapper.model_path = "E:/MLModels/stableDiffusion/sd-v1-1.ckpt"   


        elif modelName.lower() == "rdm":
            modelwrapper = CompVisRDMModel.CompVisRDMModel()


        elif modelName.lower() == "oai-uncond":
            modelwrapper = OpenAIUncondModel.OpenAIUncondModel()

            
        elif modelName.lower() == "sd-v1-4-onnx-fp32":
            modelwrapper = CompVisSDOnnxModel.CompVisSDOnnxModel(torchdtype = torch.float32)
            modelwrapper.model_path = "E:/onnxOut/sd-v1-4-fp32-cuda-auto/model.onnx"    
        elif modelName.lower() == "sd-v1-4-onnx-fp16":
            modelwrapper = CompVisSDOnnxModel.CompVisSDOnnxModel(torchdtype = torch.float16)
            modelwrapper.model_path = "E:/onnxOut/sd-v1-4-fp16-cuda-auto.onnx"    
        elif modelName.lower() == "onnx-test":
            modelwrapper = CompVisSDOnnxModel.CompVisSDOnnxModel(torchdtype = torch.float16)
            modelwrapper.model_path = "E:/onnxOut/test-sd-v1-4-fp16-auto.onnx"        

        else:
            raise exception("invalid model")
            

        modelwrapper.modelName = modelName
        modelwrapper.ModelLoadSettings()
        modelwrapper.LoadModel(self.torchDevice)

        return modelwrapper


    def CreateClipModel(self, modelName:str) -> clipWrap.ClipWrap:
        # CLIP model settings
        clipwrapper = clipWrap.ClipWrap()

        clipwrapper.modelName = modelName

        if modelName.lower() == "vit-b-16":
            clipwrapper.modelPath = "E:/MLModels/clip/ViT-B-16.pt"

        elif modelName.lower() == "vit-l-14-336":
            clipwrapper.modelPath = "E:/MLModels/clip/ViT-L-14-336px.pt"

        elif modelName.lower() == "vit-l-14":
            clipwrapper.modelPath = "E:/MLmodels/clip/ViT-L-14.pt"     

        clipwrapper.ModelLoadSettings()
        clipwrapper.LoadModel(self.torchDevice)

        return clipwrapper



    def pad_or_truncate(some_list, target_len):
        return some_list[:target_len] + [0]*(target_len - len(some_list))



    def internal_run(self, genParams:paramsGen.ParamsGen, cw:clipWrap.ClipWrap, mw:modelWrap.ModelWrap): 

        #load or unload aesthetic model, which requires a clip model to be passed in
        if genParams.aesthetics_scale != 0 and cw != None:
            if cw.aestheticModel.amodel == None:
                cw.LoadAestheticsModel(self.torchDevice)
        elif cw != None and cw.aestheticModel != None:
            cw.aestheticModel.amodel = None            

        gc.collect()

        #make sure if we have an image prompt, theres one for each sample
        if genParams.image_prompts != None and len(genParams.image_prompts) > 0 and len(genParams.image_prompts) < genParams.num_images_to_sample:
            while len(genParams.image_prompts) < genParams.num_images_to_sample:
                genParams.image_prompts.append( genParams.image_prompts[0])

        #match number of promtps to number of sample
        if genParams.prompts != None:
            #add more to match num samples
            if len(genParams.prompts) < genParams.num_images_to_sample:
                while len(genParams.prompts) < genParams.num_images_to_sample:
                    genParams.prompts.append( genParams.prompts[0])
            
            #truncate excess
            elif len(genParams.prompts) > genParams.num_images_to_sample + 1:
                genParams.prompts = genParams.prompts[0:genParams.num_images_to_sample]

            #if we have 1 more than num samples, treat it as a | b | c | d | e = a + b, a + c, a + d, a + e
            elif len(genParams.prompts) == genParams.num_images_to_sample + 1:
                #append the first entry to the other entries, make a new list
                new_list = []
                first_entry = genParams.prompts[0]
                count = 1
                for x in genParams.prompts:
                    if count > 1:
                        new_list.append(first_entry + ' ' + x)
                    count += 1
                
                genParams.prompts = new_list
                

        seeds = []
        if genParams.seed is not None:
            torch.manual_seed(genParams.seed)
        else:
            for i in range(genParams.num_images_to_sample):
                seeds.append(torch.seed())
            print('Using Seeds: '.join("%i, " % item for item in seeds))

        #cheat for now
        device = self.torchDevice

        #get the prompts to use for clip
        clip_prompts = None
        cfg_prompts = None

        if mw.default_guiding == 'CFG':
            cfg_prompts = genParams.prompts
        if mw.default_guiding == 'CLIP' or cw != None: #genParams.clip_guidance_scale != 0:
            clip_prompts = genParams.prompts

        #if there are cfg prompts, whats in prompts is going to be for clip
        if genParams.CFGprompts and len(genParams.CFGprompts[0]) > 2:
            cfg_prompts = genParams.CFGprompts
        
        # if clip prompts are defined, use that
        if genParams.CLIPprompts and len(genParams.CLIPprompts[0]) > 2:
            clip_prompts = genParams.CLIPprompts

        clipguided = False
        if cw != None:
            if genParams.clip_guidance_scale != 0 or genParams.aesthetics_scale != 0:
                if clip_prompts != None:
                    clipguided = True
            if clipguided == False:
                clip_prompts = None

        print('========= PARAMS ==========')
        print("Model: " + str(mw.model_path))
        if cw != None:
            print("Clip Model:" + str(cw.modelName))
        attrs = vars(genParams)
        # now dump this in some way or another
        print(', '.join("%s: %s" % item for item in attrs.items()))
        print('========= /PARAMS ==========')


        torch.cuda.empty_cache()
        #causes a crash ...
        #torch.backends.cudnn.benchmark = True

        modelCtx = mw.CreateModelInstance(self.torchDevice, cw, genParams, clipguided)

        modelCtx = mw.RequestImageSize(modelCtx, genParams.image_size_x, genParams.image_size_y)

        if cfg_prompts != None:
            print("using CFG denoiser, prompts: " + str(cfg_prompts))
            modelCtx = mw.CreateCFGDenoiser(modelCtx, cw, cfg_prompts, genParams.conditioning_scale, genParams)

        if clip_prompts != None:
            print("using CLIP denoiser, prompts: " + str(clip_prompts))
            modelCtx = mw.CreateClipGuidedDenoiser(modelCtx, cw, clip_prompts, genParams, device)

        modelCtx = mw.GetSigmas(modelCtx, genParams, device)

        print("got sigmas: " + str(modelCtx.sigmas))


        ############
        # start gen loop
        ################


        if cw != None:
            if genParams.cutoutMethod.lower() == 'grid':
                modelCtx.make_cutouts = cutouts.MakeGridCutouts(cw.clip_size, genParams.cutn, genParams.cut_pow)
            else:
                modelCtx.make_cutouts = cutouts.MakeCutoutsRandom(cw.clip_size, genParams.cutn, genParams.cut_pow)


        side_x = modelCtx.image_size_x
        side_y = modelCtx.image_size_y

        if genParams.image_prompts != None:
            for prompt in genParams.image_prompts:
                path, weight = utilFuncs.parse_prompt(prompt)
                img = Image.open(utilFuncs.fetch(path)).convert('RGB')
                img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)

                gridCuts = cutouts.MakeGridCutouts(cw.clip_size, 1, genParams.cut_pow)
                #make_cutouts(TF.to_tensor(img)[None].to(device))
                batch = gridCuts(TF.to_tensor(img)[None].to(device))
                embed = cw.model.encode_image(cw.normalize(batch)).float()
                modelCtx.target_clip_embeds.append(embed)
                modelCtx.clip_weights.extend([weight / genParams.cutn] * genParams.cutn)

            #we need to set the conditional phrase here...
            if mw.using_compvisLDM == True:
                #expects like [1,1,768], output from encode image above is [1,768]
                embed = embed[None,:].to(device)
                c = embed
                uc = torch.zeros_like(c)
                modelCtx.extra_args = {'cond': c, 'uncond': uc, 'cond_scale': genParams.conditioning_scale}





        if modelCtx.target_clip_embeds != None and len(modelCtx.target_clip_embeds) > 0:
            modelCtx.target_clip_embeds = torch.cat(modelCtx.target_clip_embeds)
            modelCtx.clip_weights = torch.tensor(modelCtx.clip_weights, device=device)
            if modelCtx.clip_weights.sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            modelCtx.clip_weights /= modelCtx.clip_weights.sum().abs()

        init = None
        if genParams.init_image is not None:
            print("using init image: " + genParams.init_image)
            init = Image.open(utilFuncs.fetch(genParams.init_image)).convert('RGB')
            init = init.resize((side_x, side_y), Image.Resampling.LANCZOS)
            init = TF.to_tensor(init).to(device)[None] * 2 - 1

            init = mw.EncodeInitImage(init).to(device)

        if init is not None:
            modelCtx.sigmas = modelCtx.sigmas[modelCtx.sigmas <= genParams.sigma_start]



        #modelCtx.target_clip_embeds = target_embeds
        #modelCtx.clip_weights = weights
        modelCtx.init = init       



        def callback(info, imgData = None):
            i = info['i'] 
            if info['i'] % 25 == 0:
                tqdm.write(f'Step {info["i"]} of {len(modelCtx.sigmas) - 1}, sigma {info["sigma"]:g}:')

            if info['i'] != 0 and info['i'] % genParams.saveEvery == 0:
                denoised = mw.DecodeImage(imgData)
                nrow = math.ceil(denoised.shape[0] ** 0.5)
                grid = utils.make_grid(denoised, nrow, padding=0)
                filename = f'step_{i}.png'
                K.utils.to_pil_image(grid).save(filename)


        utilFuncs.log_torch_mem("Starting sample loops")
        
        def doSamples(sm: str):
            
            if len(seeds) > 1:
                imgTensors = []
                for i in range(genParams.num_images_to_sample):
                    torch.manual_seed(seeds[i])
                    randImg = torch.randn([1, modelCtx.modelWrap.channels, 
                                    modelCtx.image_tensor_size_y, modelCtx.image_tensor_size_x], device=device) * modelCtx.sigmas[0]

                    imgTensors.append(randImg)

                x = torch.cat(imgTensors,0)
            else:                
                x = torch.randn([genParams.num_images_to_sample, modelCtx.modelWrap.channels, 
                                modelCtx.image_tensor_size_y, modelCtx.image_tensor_size_x], device=device) * modelCtx.sigmas[0]

            
            if init is not None:
                x += init

            #hacky ONNX test
            if hasattr(mw, 'ONNX') and mw.ONNX == True:
                #since we load the original model into CPU memory, and ONNX onto device, sort out where tensor are living...
                x = x.to(device)
                modelCtx.sigmas = modelCtx.sigmas.to(device)
                modelCtx.extra_args["cond"] = modelCtx.extra_args["cond"].to(device)
                modelCtx.extra_args["uncond"] = modelCtx.extra_args["uncond"].to(device)
                modelCtx.extra_args["cond_scale"] = torch.FloatTensor([modelCtx.extra_args["cond_scale"]]).to(device)

                if modelCtx.modelWrap.tensordtype == torch.float16:
                    x = x.half()
                    modelCtx.sigmas = modelCtx.sigmas.half()
                    modelCtx.extra_args["cond"] = modelCtx.extra_args["cond"].half()
                    modelCtx.extra_args["uncond"] = modelCtx.extra_args["uncond"].half()
                    modelCtx.extra_args["cond_scale"] = modelCtx.extra_args["cond_scale"].half()

                x_0 = onnxSampling.sample_lms_ONNX(mw.ONNXSession, x, modelCtx.sigmas, callback=callback, extra_args=modelCtx.extra_args)
                #x_0 = onnxSampling.sample_lms_ONNX_with_binding(mw.ONNXSession, x, modelCtx.sigmas, bindingType = modelCtx.modelWrap.tensordtype, callback=callback, extra_args=modelCtx.extra_args)

            elif  sm == "heun":
                print("sampling: HUEN")
                x_0 = K.sampling.sample_heun(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, s_churn=20, callback=callback, extra_args=modelCtx.extra_args)
            elif sm == "lms":
                print("sampling: LMS")
                x_0 = K.sampling.sample_lms(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, callback=callback,extra_args=modelCtx.extra_args)
            elif sm == "euler":
                print("sampling: EULER")
                x_0 = K.sampling.sample_euler(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, s_churn=20, callback=callback,extra_args=modelCtx.extra_args)
            elif sm == "euler_a":
                print("sampling: EULER_A")
                x_0 = K.sampling.sample_euler_ancestral(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, callback=callback,extra_args=modelCtx.extra_args)
            elif sm == "dpm_2":
                print("sampling: DPM_2")
                x_0 = K.sampling.sample_dpm_2(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, s_churn=20, callback=callback,extra_args=modelCtx.extra_args)
            elif sm == "dpm_2_a":
                print("sampling: DPM_2_A")
                x_0 = K.sampling.sample_dpm_2_ancestral(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, callback=callback,extra_args=modelCtx.extra_args)
            else:
                print("ERROR: invalid sampling method, defaulting to LMS")
                x_0 = K.sampling.sample_lms(modelCtx.kdiffModelWrap, x, modelCtx.sigmas, callback=callback,extra_args=modelCtx.extra_args)
                
            return x_0



        if str(device) == 'cpu':
            precision_device = "cpu"
        else:
            precision_device = "cuda"

        precision_scope = autocast if hasattr(modelCtx, 'precision') and modelCtx.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope(precision_device):
                if hasattr(mw.model, 'ema_scope'):
                    with mw.model.ema_scope():
                        samples = doSamples(genParams.sampleMethod.lower())
                else:
                    samples = doSamples(genParams.sampleMethod.lower())

        ############
        # end gen loop
        ################
        with precision_scope(precision_device):
            samples = mw.DecodeImage(samples)
            #seemed fine without this next part...not sure if its needed?
            #makes it look washed out
            #samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

        #TODO: combine all the samples into a grid somehow
        imgsList = []
        for i, out in enumerate(samples):
            imgsList.append( K.utils.to_pil_image(out) )
            #grid = K.utils.to_pil_image(out)

        if clipguided == True:
            denoised = modelCtx.kdiffModelWrap.orig_denoised

            denoised = mw.DecodeImage(denoised)

            nrow = math.ceil(denoised.shape[0] ** 0.5)
            grid = make_grid(denoised, nrow, padding=0)
            grid = K.utils.to_pil_image(grid)
        else:
            #nrow = math.ceil(denoised.shape[0] ** 0.5)
            rows = 8
            if genParams.num_images_to_sample <= 4:
                rows = 2
            elif genParams.num_images_to_sample <= 9:
                rows = 3
            elif genParams.num_images_to_sample <= 16:
                rows = 4
            grid = make_grid(samples, rows, padding=0)
            grid = K.utils.to_pil_image(grid)

        return grid, imgsList, seeds
