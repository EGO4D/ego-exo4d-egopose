"""
    These function do not return value, they modify image in place instead.
"""

import numpy as np
import PIL
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2


# ORDER : bbox coordinate order
# NORM : normalized or absolute
ORDER = 'xyxy'  # one of {'xyxy', 'yxyx', 'xywh'}
NORM = False  # True for google
setup_flag = False

# The following to is for drawing class specific bbox (diff color, display
#   name), currently support mmdetection format.
# CLASSES : display class name
# COLORS : (Optional) overload default colors
# FONT_SIZE: display font size
CLASSES = None
COLORS = list(PIL.ImageColor.colormap.keys())
FONT_SIZE = 12


def setup(order='xyxy',
          norm=False,
          classes=None,
          font_size=12,
          colors=None,
          colors_overload=None):
    """
    Args:
        order: str, 'xyxy' or 'yxyx'
        norm: bool
        classes: list of str
        font_size: int
        colors: display colors list
        colors_overload: list of str, will overload default color order.
                         e.g. ['lightgreen', 'red']
    """
    global ORDER
    global NORM
    global CLASSES
    global COLORS
    global FONT_SIZE
    global setup_flag
    setup_flag = True
    ORDER = order
    NORM = norm
    CLASSES = classes
    FONT_SIZE = font_size
    if colors is not None:
        COLORS = colors
    # swap color order to meet overload requiremnt
    if colors_overload is not None:
        colors = COLORS
        for i, clr in enumerate(colors_overload):
            ind = colors.index(clr)
            colors[i], colors[ind] = colors[ind], colors[i]
        COLORS = colors


def draw_box_image(image,
                   ymin,
                   xmin,
                   ymax,
                   xmax,
                   color='red',
                   thickness=4,
                   display_str_list=()):
    """Adds a bounding box to an image.

    Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box. [0,1]
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default is 4.
        display_str_list: list o strings to display in box
                            (each to be shown on its own line).
    """
    if not setup_flag:
        print("WARN: setup() not run, \
              default order:{ORDER}, default normalize:{NORM}")
    assert ORDER in set({'xyxy', 'yxyx', 'xywh'})
    if ORDER == 'xyxy':
        xmin, ymin, xmax, ymax = ymin, xmin, ymax, xmax
    elif ORDER == 'yxyx':
        pass
    elif ORDER == 'xywh':
        xmin, ymin, box_w, box_h = ymin, xmin, ymax, xmax
        xmax, ymax = xmin + box_w, ymin + box_h
    # if (ymin >= ymax or xmin >= xmax):
    #     print(ymin, xmin, ymax, xmax)
    #     raise ValueError("ymin > ymax or xmin > xmax, format *is not* [y, x, w, h]" )
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if NORM:
        (left, top, right, bottom) = (im_width * xmin, im_height * ymin,
                                      im_width * xmax, im_height * ymax)
    else:
        (left, top, right, bottom) = (xmin, ymin, xmax, ymax)
    draw.line([(left, top), (right, top),
               (right, bottom), (left, bottom), (left, top)],
              width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 30)
    except IOError:
        font = ImageFont.load_default()
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(right - margin - text_width, text_bottom - text_height - margin),
             (right - margin, text_bottom)], fill=color)
        draw.text(
            (right - margin - text_width, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height + 0 * margin


def draw_box_image_array(image,
                         ymin,
                         xmin,
                         ymax,
                         xmax,
                         color='red',
                         thickness=4,
                         display_str_list=()):
    """Adds a bounding box to an image (numpy array).

    Args:
        image: a numpy array with shape [height,width,3].
        ymin: ymin of bounding box. [0,1]
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default is 4.
        display_str_list: list o strings to display in box
                            (each to be shown on its own line).
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_box_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list)
    # np.copyto(image,np.array(image_pil))
    return image_pil


def draw_bboxes_image(image,
                      boxes,
                      color='red',
                      thickness=4,
                      display_str_list_list=()):
    """Draws bounding boxes on image.

    Args:
        image: a PIL.Image object.
        boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
               normalized [0, 1].
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

    Raises:
        ValueError: if boxes is not a [N, 4] array
    """
    if np.ndim(boxes) != 2 or boxes.shape[1] != 4:
        raise ValueError('boxes should be [N,4] array, but got {}'.format(boxes.shape))
    for box_idx in range(boxes.shape[0]):
        (ymin, xmin, ymax, xmax) = boxes[box_idx]
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[box_idx]
        draw_box_image(image, ymin, xmin, ymax, xmax,
                                   color, thickness, display_str_list)


def draw_bboxes_image_array(image,
                            boxes,
                            color='red',
                            thickness=4,
                            display_str_list_list=()):
    """Draws bounding boxes on image (numpy array).

    Args:
        image: a numpy array with shape [height,width,3].
        boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
               normalized [0, 1].
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list_list: list of list of strings.
                               a list of strings for each bounding box.
                               The reason to pass a list of strings for a
                               bounding box is that it might contain
                               multiple labels.

    Raises:
        ValueError: if boxes is not a [N, 4] array
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bboxes_image(image_pil, boxes, color, thickness, display_str_list_list)
    # np.copyto(image,np.array(image_pil))
    return image_pil


def draw_dets_image_array_mmlab(image,
                                dets,
                                thickness=4,
                                force_color=None,
                                thr=None):
    """Draw colorful detection results on image array.

    Args:
        image: a numpy array with shape [height, width, 3].
        dets: a list of 2 dimensional numpy array of [N, 5],
              (xmin, ymin, xmax, ymax, score).
              A [N, 4] format will also work.
        thickness: line thickness. Default value is 4.
        force_color: 'str', if not None, all box use this color.
        thr: float or None
    """
    if not setup_flag:
        print("Warning, No setup for COLORS and CLASSES names, use default")
    if len(dets) > len(COLORS):
        print("Max {len(COLORS)} classes supported, \
              get {len(dets)} > {len(COLORS)}")
    img = image.copy()
    if thr is None:
        thr = -1.0
    for c, det in enumerate(dets):
        has_score = det.shape[1] >= 5
        det = det[det[:,-1]>thr]
        if CLASSES is not None and has_score:
            display_str_list_list = [
                [CLASSES[c], f"{d[-1]:.02f}"] for d in det
            ]
        elif CLASSES is None and has_score:
            display_str_list_list = [
                [f"{d[-1]:.02f}"] for d in det
            ]
        elif CLASSES is None and not has_score:
            display_str_list_list = [
                [CLASSES[c]] for d in det
            ]
        else:
            display_str_list_list = (())
        color = COLORS[c] if force_color is None else force_color
        img = draw_bboxes_image_array(
            image=img, boxes=det[..., :4], color=color,
            thickness=thickness, display_str_list_list=display_str_list_list)
    return img


def draw_gt_image_array_plain(image,
                              gts,
                              thickness=4):
    """Draw colorful ground truth on image array.

    Args:
        image: a numpy array with shape [height, width, 3].
        dets: a dimensional numpy array of [N, 5],
              (xmin, ymin, xmax, ymax, CLASS_INDEX).
        thickness: line thickness. Default value is 4.
    """
    if not setup_flag:
        print("Warning, No setup for COLORS and CLASSES names, use default")
    if gts[..., -1].max() > len(COLORS):
        print("Max {len(COLORS)} classes supported, \
              get {len(dets)} > {len(COLORS)}")
    img = image.copy()
    for gt in gts:
        if CLASSES is not None:
            display_str_list = [
                CLASSES[gt[-1]]
            ]
        else:
            display_str_list = [ str(gt[-1]) ]
        x1, y1, x2, y2, c = gt
        img = draw_box_image_array(
            image=img, ymin=x1, xmin=y1, ymax=x2, xmax=y2, color=COLORS[c],
            thickness=thickness, display_str_list=display_str_list)
    return img


def rescale_img_pixel(img):
    """ Rescale pixel value from [-1, 1] float to [0, 255] uint8.

    Args:
        img: ndarray, range from [-1, 1] float

    Return:
        ndarray, [0, 255] uint8
    """
    return np.uint8((img + 1) * 255. / 2)


def rescale_with_pad(img, width=500, ar=1.4):
    """ Rescale and pad image to fix shape.
        ar = w / h

    Args:
        img: ndarray
        width: int
        ar: float

    Returns:
        ndarray
    """
    img = np.asarray(img)
    ht, wd, chan = img.shape
    width = int(width)
    height = int(width / ar)
    fill = (0, 0, 0)
    result = np.full((height, width, chan), fill, dtype=np.uint8)

    scale_h = 1.0 * height / ht
    scale_w = 1.0 * width / wd
    scale = min(scale_h, scale_w)
    ht = int(scale * ht)
    wd = int(scale * wd)
    img = cv2.resize(img, (wd, ht))

    xx = (width - wd) // 2
    yy = (height - ht) // 2
    result[yy:yy+ht, xx:xx+wd] = img
    return result


def heatmap_on_image(heatmap, image, weight_hm=0.5, weight_img=None):
    """ Draw heatmap on image.
        output pixel = weight_hm * hm + weight_img * image

    Args:
        heatmap: [H, W] np.float32 image, value range [-1, 1],
            the value of heatmap will be scaled to [0, 255] and convert to uint8
        image: [H, W, 3] np.uint8.
        weight_hm: float
        weight_img: None or float, if None, weight_img = 1 - weight_hm

    Return:
        [H, W, 3]
    """
    img_h, img_w, _ = image.shape
    hm_resize = cv2.resize(heatmap, (img_w, img_h))
    hm_resize = rescale_img_pixel(hm_resize)
    hm_color = cv2.applyColorMap(hm_resize, cv2.COLORMAP_JET)
    if weight_img is None:
        weight_img = 1 - weight_hm
    heatmapped_img = cv2.addWeighted(hm_color, weight_hm, image, weight_img, 0)
    return heatmapped_img
