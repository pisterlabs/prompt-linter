import replicate
import cohere

REPLICATE_API_TOKEN = 'e93017c2a366e89cf2df8f8ccd0699902f40d449'
COHERE_API_TOKEN = 'hpzIRAK5VD6lPMYv7OdXBKCwxvknVf9TPnDOvHWZ'


def cohere_summarize(prompt, api_key):
    co = cohere.Client(api_key)

    response = co.generate(
        prompt=prompt,
    )

    return response


def stable_diffusion_request(prompt, api_token=REPLICATE_API_TOKEN):
    # set REPLICATE_API_TOKEN=e93017c2a366e89cf2df8f8ccd0699902f40d449
    replicate.Client(api_token=REPLICATE_API_TOKEN)

    model = replicate.models.get("stability-ai/stable-diffusion")
    version = model.versions.get(
        "f178fa7a1ae43a9a9af01b833b9d2ecf97b1bcb0acfd2dc5dd04895e042863f1")

    inputs = {
        'prompt': "detailed, 4K, simple, Create an image about: " + prompt,
        'negative-prompt': "no words",
        'width': 768,
        'height': 768,
        'prompt_strength': 0.8,
        'num_outputs': 1,
        'num_inference_steps': 50,
        'guidance_scale': 7.5,
        'scheduler': "DPMSolverMultistep",
    }

    return version.predict(**inputs)


prompt = "a 20% off deal at my ice cream store for chocolate ice cream this weekend."
abstract = "Despite recent progress in generative image modeling, successfully generating high-resolution, diverse samples from complex datasets such as ImageNet remains an elusive goal. To this end, we train Generative Adversarial Networks at the largest scale yet attempted, and study the instabilities specific to such scale. We find that applying orthogonal regularization to the generator renders it amenable to a simple 'truncation trick,' allowing fine control over the trade-off between sample fidelity and variety by reducing the variance of the Generator's input. Our modifications lead to models which set the new state of the art in class-conditional image synthesis. When trained on ImageNet at 128x128 resolution, our models (BigGANs) achieve an Inception Score (IS) of 166.5 and Frechet Inception Distance (FID) of 7.4, improving over the previous best IS of 52.52 and FID of 18.6."

print(stable_diffusion_request(prompt))
# print(cohere_summarize(abstract, COHERE_API_TOKEN))
