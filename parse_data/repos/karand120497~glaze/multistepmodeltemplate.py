class Template:
    def __init__(self, device):
        print(f"Initializing Template to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_id = ""
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id)
        self.lora_weights = "file location for weights in safetensor"                                                    torch_dtype=self.torch_dtype)
        self.pipe = self.convert(self.model_id,self.lora_weights)
        self.pipe.to(device)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

    @prompts(name="One-liner for your functionality",
             description="Detailed description for your functionality")
    def inference(self, text):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        prompt = text + ', ' + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Template, Input Text: {text}, Output Image: {image_filename}")
        return image_filename
    
    def convert(base_model_path, checkpoint_path, LORA_PREFIX_UNET="lora_unet", LORA_PREFIX_TEXT_ENCODER="lora_te", alpha=0.75):

    # load base model
        pipeline = StableDiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float32)

    # load LoRA weight from .safetensors
        state_dict = load_file(checkpoint_path)

        visited = []

    # directly update weight in diffusers model
        for key in state_dict:

        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
            if ".alpha" in key or key in visited:
                continue

            if "text" in key:
                layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
            else:
                layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipeline.unet

        # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            pair_keys = []
            if "lora_down" in key:
                pair_keys.append(key.replace("lora_down", "lora_up"))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_up = state_dict[pair_keys[0]].to(torch.float32)
                weight_down = state_dict[pair_keys[1]].to(torch.float32)
                curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
            for item in pair_keys:
                visited.append(item)

        return pipeline
