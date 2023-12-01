from stackview import jupyter_displayable_output

@jupyter_displayable_output(library_name='darth-d', help_url='https://github.com/haesleinhuepf/darth-d')
def vary(input_image, image_width:int=1024, image_height:int=1024, num_images:int=1):
    """Varies an image using OpenAI's DALL-E 2.

    Parameters
    ----------
    input_image: 2D image, potentially RGB
    num_images: int, optional
        ignored for dall-e-3
    model: str, optional
        "dall-e-2", "dall-e-3"
    image_width: int, optional
        must be 256, 512 or 1024 for dall-e-2
    image_height: int, optional
        must be 256, 512 or 1024 for dall-e-2

    See Also
    --------
    https://platform.openai.com/docs/guides/images/variations

    Returns
    -------
    single 2D image or 3D image with the first dimension = num_images
    """
    from openai import OpenAI
    from ._utilities import numpy_to_bytestream
    from ._utilities import images_from_url_responses
    
    from stackview._image_widget import _img_to_rgb
    from warnings import warn

    model: str = "dall-e-2"

    warn("Using the vary function on scientific images could be seen as scientific misconduct. Handle this function with care.")
    client = OpenAI()

    size_str = f"{image_width}x{image_height}"

    response = client.images.create_variation(
      image=numpy_to_bytestream(_img_to_rgb(input_image)),
      n=num_images,
      size=size_str,
    )

    # bring result in right format
    return images_from_url_responses(response, input_image.shape)
    