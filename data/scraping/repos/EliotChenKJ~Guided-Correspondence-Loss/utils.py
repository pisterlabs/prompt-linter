import PIL.Image as Image
import random
import torch
import torch.nn.functional as F
import math
import torchvision
import matplotlib.pyplot as plt
from scipy import ndimage

def isNone(x):
    return type(x) is type(None)

def decode_image(path, size=None, type='RGB'):
    """ Load and resize the input texture """

    img = Image.open(path).convert(type)
    width, height = img.size
    max_size = width
    if max_size > height:
        max_size = height
    trans_list = ([torchvision.transforms.Resize(size, Image.LANCZOS)] if size else []) + [torchvision.transforms.ToTensor()]
    trans = torchvision.transforms.Compose(trans_list)
    img = trans(img)
    img = img.unsqueeze(0)
    return img


def save_image(path, image_tensor):
    n, c, h, w = image_tensor.shape
    if c == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)
    if n > 1:
        image_tensor = image_tensor.permute(1, 2, 0, 3).reshape(1, c, h, -1)
    plt.imsave(path, image_tensor.permute(0, 2, 3, 1).detach().cpu().float().numpy()[0])


def initialize_from_progression(target_guid, source_guid, source_image):
    """
    Get init image from guidance according to correspondence
        Input:
        - target_guid: target guidance image [1, h_g, w_g]
        - source_guid: source guidance image [1, h_s, w_s]
        - source_image: source image [c, h_s, w_s]
        Output:
        - target_image: target image [c, h_g, w_g]
    """
    # source guidance value2Indexs
    value2Indexs = [None] * 256
    values = []
    _, hs, ws = source_guid.shape
    for _h in range(hs):
        for _w in range(ws):
            _c = int(source_guid[0, _h, _w])
            if value2Indexs[_c]:
                value2Indexs[_c].append([_h, _w])
            else:
                value2Indexs[_c] = [[_h, _w]]
                values.append(_c)

    # target image initiation
    _, hg, wg = target_guid.shape
    c = source_image.shape[0]
    target_image = torch.zeros(c, hg, wg)
    for _h in range(hg):
        for _w in range(wg):
            _c = int(target_guid[0, _h, _w])
            if value2Indexs[_c]:
                _idxs = value2Indexs[_c]
            else:
                _idxs = value2Indexs[min(values, key=lambda x: abs(x - _c))]
            _rh, _rw = _idxs[int(random.random() * (len(_idxs) - 1))]
            target_image[:, _h, _w] = source_image[:, _rh, _rw]

    return target_image


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """
    angle = math.radians(angle)

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a np / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[-1], image.shape[2])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if (width > image_size[0]):
        width = image_size[0]

    if (height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[:, :, y1:y2, x1:x2]


def generate_coordinate(tensor):
    n, c, h, w = tensor.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, h),
        torch.linspace(0, 1, w),
    )
    coordinate = torch.stack([grid_y, grid_x])
    return coordinate.to(tensor.device)[None].expand(n, 2, h, w)


def distance_transform(tensor):
    device = tensor.device
    tensor = tensor.cpu() > 0
    distance = ndimage.distance_transform_edt(tensor)
    distance = torch.tensor(distance).type(torch.float32).to(device)
    distance = (distance - distance.min()) / (distance.max() - distance.min())
    return distance


def visualize_progression_map(guidance, output_folder, filename):
    for index in range(guidance.shape[0]):
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.imshow(guidance.cpu()[index].permute(1, 2, 0))
        plt.savefig(output_folder + f'/{filename}-{index}.jpg', bbox_inches='tight', pad_inches=0)


def visualize_orientation_map(guidance, output_folder, filename, image=None, step=16):
    for index in range(guidance.shape[0]):
        orient = guidance[index].cpu()
        x = torch.linspace(step // 2, guidance.shape[-1] - step // 2, step)
        y = torch.linspace(step // 2, guidance.shape[-2] - step // 2, step)
        x = x[None, :].repeat(y.shape[-1], 1)
        y = y[:, None].repeat(1, x.shape[-1])

        u, v = orient[0], orient[1]
        u = u[y.long(), x.long()]
        v = v[y.long(), x.long()]

        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.gca().invert_yaxis()
        if not isNone(image):
            plt.imshow(image.cpu()[index].permute(1, 2, 0))
        plt.quiver(x, y, u, -v, headwidth=0, pivot='mid', color='red')
        plt.savefig(output_folder + f'/{filename}-{index}.jpg', bbox_inches='tight', pad_inches=0)
