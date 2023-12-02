
from ..constants import GUIDANCE_SCALE_DEFAULT, NUM_INFERENCE_STEPS_DEFAULT

def improve_prompt(prompt, level="high"):

    detail_levels = {
        "high":  "highly detailed, realistic, unreal engine, octane render, vray, houdini render, quixel megascans, depth of field, 8k uhd, raytracing, lumen reflections, ultra realistic, cinema4d, studio quality",
        "alpha": "intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by gaston bussiere and alphonse mucha",
        "beta": "cinematic lighting, photorealistic, ornate, intricate, realistic, detailed, volumetric light and shadow, hyper HD, octane render, unreal engine insanely detailed and intricate, hypermaximalist",
        "gamma": "by tim okamura, victor nizovtsev, greg rutkowski, noah bradley. trending on artstation, 8k, masterpiece, graffiti paint, fine detail, full of color, intricate detail, golden ratio illustration",
    }

    return prompt.strip() + " " + detail_levels[level]


ALL_OPTIONS = {
    "n": "(int) repeat image gen with same settings n amount of times",
    "gs": "(int) set guidance scale",
    "steps": "(int) set number of steps",
    "models": "(str) choose from a preset saved model",
    "detailed": "(str) choose from preset saved prompts",
    "custom": "(str) choose from custom model (any diffuser model from huggingface"
}


class Options:

    def parse_options(self):

        if "n" in self.options:
            self.repeat = int(self.options["n"])

        if "gs" in self.options:
            self.guidance_scale = int(self.options["gs"])
        else:
            self.guidance_scale = GUIDANCE_SCALE_DEFAULT
        
        
        if "steps" in self.options:
            self.num_inference_steps = int(self.options["steps"])
        else:
            self.num_inference_steps = NUM_INFERENCE_STEPS_DEFAULT

        if "custom" in self.options:
            self.model_id = self.options["custom"]
            self.add_response("custom model chosen: " + self.model_id)

        if "detailed" in self.options:
            if self.options["detailed"] == "all":
                self.all_details = True
            else:
                self.prompt = improve_prompt(self.prompt, self.options["detailed"])
            self.add_response("custom details: True")

        self.add_response("repeat: " + str(self.repeat))
        self.add_response("gs: " + str(self.guidance_scale))
        self.add_response("num_inference_steps: " + str(self.num_inference_steps))
        

    def add_response(self, info):
        self.response += "\n" + info

    @classmethod
    def show_all_options(cls):
        all_options = ""
        for k,v in ALL_OPTIONS.items():
            all_options += f'{k}: {v}\n'
        
        return all_options
