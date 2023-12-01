from stackview import jupyter_displayable_output
from warnings import warn

@jupyter_displayable_output(library_name='darth-d', help_url='https://github.com/haesleinhuepf/darth-d')
def replace(input_image, mask = None, prompt:str = "A similar pattern like in the rest of the image", image_width:int=1024, image_height:int=1024, num_images:int = 1):
    """
    Replace a masked region in an image with a new pattern as described in a prompt using OpenAI's DALL-E 2.
    In case no mask is given, the entire image is replaced in a two-step process:
    First the upper left and lower bottom half of the image is replaced. Afterwards the other two quadrants.
    This may lead to artifacts at quadrant borders.

    Parameters
    ----------
    input_image: 2D image, potentially RGB
    mask: 2D image, optional
    prompt: str, optional
    num_images: int, optional
    image_width: int, optional
        must be 256, 512 or 1024 for dall-e-2
    image_height: int, optional
        must be 256, 512 or 1024 for dall-e-2

    See Also
    --------
    https://platform.openai.com/docs/guides/images/edits

    Returns
    -------
    single 2D image or 3D image with the first dimension = num_images
    """
    from ._utilities import numpy_to_bytestream
    from stackview._image_widget import _img_to_rgb
    from skimage import transform
    import numpy as np
    from openai import OpenAI
    from ._utilities import images_from_url_responses
    from warnings import warn

    model: str = "dall-e-2"

    warn("Using the replace function on scientific images could be seen as scientific misconduct. Handle this function with care.")

    if mask is None:
        # In case no mask is given, make one with a 2x2 checker board pattern
        mask = np.zeros(input_image.shape[:2], dtype=np.uint8)
        mask[:int(mask.shape[0] / 2), :int(mask.shape[1] / 2)] = 1
        mask[int(mask.shape[0] / 2):, int(mask.shape[1] / 2):] = 1

        # replace two quadrants of the image
        half_replaced = replace(input_image=input_image, mask=mask, prompt=prompt, image_width=image_width, image_height=image_height, num_images=num_images, model=model)
        mask_inverse = ((mask == 0) * 1).astype(dtype=np.uint8)

        # replace the other two quadrants
        if num_images > 1:
            # in case multiple images were requested, we need to process the individual half-replaced images
            replaced = np.asarray([
                replace(input_image=half_replaced_image, mask=mask_inverse, prompt=prompt, image_width=image_width,
                        image_height=image_height, num_images=1, model=model) for half_replaced_image in half_replaced])
        else:
            replaced = replace(input_image=half_replaced, mask=mask_inverse, prompt=prompt, image_width=image_width,
                        image_height=image_height, num_images=num_images, model=model)

        return replaced

    # in case mask is given

    # we rescale image and mask to the specified size
    resized_image = transform.resize(input_image, (image_height, image_width), anti_aliasing=True)
    resized_mask = transform.resize(mask, (image_height, image_width), anti_aliasing=False)

    resized_image_rgb = _img_to_rgb(resized_image)
    resized_image_rgb = (resized_image_rgb * 255 / resized_image_rgb.max()).astype(np.uint8)

    # masked = np.swapaxes(np.swapaxes(np.asarray([(resized_mask == 0)] * 4), 0, 2), 0,1)
    masked = (np.swapaxes(np.swapaxes(np.asarray([
        resized_image_rgb[:, :, 0],
        resized_image_rgb[:, :, 1],
        resized_image_rgb[:, :, 2],
        (resized_mask == 0) * 255]), 0, 2), 0, 1)).astype(np.uint8)

    # actual request to OpenAI's DALL-E 2/3
    client = OpenAI()

    response = client.images.edit(
      image=numpy_to_bytestream(resized_image_rgb),
      mask=numpy_to_bytestream(masked),
      prompt=prompt,
      n=num_images,
      size=f"{image_width}x{image_height}",
    )

    # bring result in right format
    return images_from_url_responses(response, input_image.shape)
    