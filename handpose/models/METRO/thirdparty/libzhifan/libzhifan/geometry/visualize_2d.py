import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

import numpy as np


def draw_proj_cuboid_image(image,
                           proj_cuboid: np.ndarray,
                           color='green',
                           thickness=4):
    """
    TODO consider different colors for different sides?

    Args:
        image: PIL.Image.Image or (h, w, 3) ndarray
        proj_cuboid: [8, 2] in absolute
        color:
        thickness:

    Returns: same as image

    """
    is_array = True
    if isinstance(image, Image.Image):
        is_array = False
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        raise ValueError(f"Data `image` with type {type(image)} not understood.")
    draw = ImageDraw.Draw(image)
    points = [(proj_cuboid[i][0], proj_cuboid[i][1]) for i in range(8)]
    # Front
    lines = [
        points[0], points[1], points[2], points[3], points[0]
    ]
    draw.line(lines, width=thickness, fill=color)
    # Rear
    lines = [
        points[4], points[5], points[6], points[7], points[4]
    ]
    draw.line(lines, width=thickness, fill=color)
    # # (0, 4), (1, 5), (2, 6), (3, 7)
    draw.line([points[0], points[4]], width=thickness//2, fill=color)
    draw.line([points[1], points[5]], width=thickness//2, fill=color)
    draw.line([points[2], points[6]], width=thickness//2, fill=color)
    draw.line([points[3], points[7]], width=thickness//2, fill=color)
    if is_array:
        return np.asarray(image)
    else:
        return image


def draw_box2d_image(image,
                     bnd_box: np.ndarray,
                     color='green',
                     thickness=4):
    """

    Args:
        image: PIL.Image.Image or (h, w, 3) ndarray
        bnd_box: [y1, x1, y2, x2] in absolute
        color:
        thickness:

    Returns: same as image
    """
    is_array = True
    if isinstance(image, Image.Image):
        is_array = False
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        raise ValueError(f"Data `image` with type {type(image)} not understood.")
    draw = ImageDraw.Draw(image)
    top, left, bottom, right = bnd_box
    draw.line([(left, top), (right, top),
               (right, bottom), (left, bottom), (left, top)],
              width=thickness, fill=color)
    if is_array:
        return np.asarray(image)
    else:
        return image


def draw_line_image(image,
                    line: np.ndarray,
                    color='green',
                    thickness=4):
    """

    Args:
        image:
        line: [ [x_min, y_min], [x_max, y_max] ]
        color:
        thickness:

    Returns: same as image

    """
    is_array = True
    if isinstance(image, Image.Image):
        is_array = False
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        raise ValueError(f"Data `image` with type {type(image)} not understood.")
    draw = ImageDraw.Draw(image)
    left, top = line[0]
    right, bottom = line[1]
    draw.line([(left, top), (right, bottom)],
              width=thickness, fill=color)
    if is_array:
        return np.asarray(image)
    else:
        return image


def draw_pivots_image(image,
                      pivots: np.ndarray,
                      thickness=4):
    """
    draw pivots with X-Red, Y-Green, Z-Blue

    Args:
        image:
        pivots: [ [h_start, v_start],
                  [X_h, X_v],
                  [Y_h, Y_v],
                  [Z_h, Z_v] ] , h(horizontal), v(vertical)
        thickness:

    Returns: same as image
    """
    is_array = True
    if isinstance(image, Image.Image):
        is_array = False
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        raise ValueError(f"Data `image` with type {type(image)} not understood.")
    draw = ImageDraw.Draw(image)

    h, v = pivots[0]
    x_h, x_v = pivots[1]
    y_h, y_v = pivots[2]
    z_h, z_v = pivots[3]

    draw.line([(h, v), (x_h, x_v)],
              width=thickness, fill='red')
    draw.line([(h, v), (y_h, y_v)],
              width=thickness, fill='green')
    draw.line([(h, v), (z_h, z_v)],
              width=thickness, fill='blue')
    if is_array:
        return np.asarray(image)
    else:
        return image


def draw_dots_image(image,
                    dots: np.ndarray,
                    color='green',
                    thickness=4):
    """

    Args:
        image: Image or ndarray
        dots: (n, 2)
        color:
        thickness: radius equals 2 * thickness

    Returns: same as image
    """
    is_array = True
    if isinstance(image, Image.Image):
        is_array = False
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        raise ValueError(f"Data `image` with type {type(image)} not understood.")
    draw = ImageDraw.Draw(image)
    r = 2 * thickness
    for (x, y) in dots:
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    if is_array:
        return np.asarray(image)
    else:
        return image
