from stackview import jupyter_displayable_output

@jupyter_displayable_output(library_name='darth-d', help_url='https://github.com/haesleinhuepf/darth-d')
def create(prompt:str=None, image_width:int=1024, image_height:int=1024, num_images:int=1, model:str="dall-e-3", style:str='vivid', quality:str='standard'):
    """Create an image from scratch using OpenAI's DALL-E 2 or 3.

    Parameters
    ----------
    prompt: str, text explaining what the image should show
    num_images: int, optional
        ignored for dall-e-3
    model: str, optional
        "dall-e-2", "dall-e-3"
    image_width: int, optional
        must be 256, 512 or 1024 for dall-e-2 or 1024, 1792 for dall-e-3
    image_height: int, optional
        must be 256, 512 or 1024 for dall-e-2 or 1024, 1792 for dall-e-3
    style: str, optional
        "vivid" or "natural", ignored for dall-e-2
    quality: str, optional
        "standard" or "hd", ignored for dall-e-2

    See Also
    --------
    https://platform.openai.com/docs/guides/images/generations

    Returns
    -------
    single 2D image or 3D image with the first dimension = num_images
    """
    from openai import OpenAI
    from ._utilities import images_from_url_responses

    client = OpenAI()

    size_str = f"{image_width}x{image_height}"

    kwargs = {}
    if model == "dall-e-3":
        kwargs['style'] = style
        kwargs['quality'] = quality

    response = client.images.generate(
      prompt=prompt,
      n=num_images,
      model=model,
      size=size_str,
        **kwargs
    )


    # bring result in right format
    return images_from_url_responses(response)
