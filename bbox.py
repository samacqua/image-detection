from enum import Enum, auto
class BBox(Enum):
    """
    enumeration of different bbox types
    """
    XYWH = auto()   # [top left x, top left y, width, height]
    X1Y1X2Y2 = auto()   # [left x, top y, right x, bottom y]
    XMIDYMIDWH_NORM = auto()    # [center x (normalized), center y (noramlized), width (normalized), height (normalized)]

def convert_bbox(bbox, from_format, to_format, image_size=(800, 600)):
    """convert bounding box between any two formats"""

    if from_format == BBox.XYWH:
        x,y,w,h = bbox
    elif from_format == BBox.X1Y1X2Y2:
        x,y,x2,y2 = bbox
        w, h = x2-x, y2-y
    elif from_format == BBox.XMIDYMIDWH_NORM:
        x_mid_n, y_mid_n, w_n, h_n = bbox
        x_mid, w = x_mid_n * image_size[0], w_n * image_size[0]
        y_mid, h = y_mid_n * image_size[1], h_n * image_size[1]
        x = x_mid - w/2
        y = y_mid - h/2
    
    if to_format == BBox.XYWH:
        return [x,y,w,h]
    elif to_format == BBox.X1Y1X2Y2:
        return [x,y,x+w,y+h]
    elif to_format == BBox.XMIDYMIDWH_NORM:
        x_n, w_n = x/image_size[0], w/image_size[0]
        y_n, h_n = y/image_size[1], h/image_size[1]
        x_mid_n = x_n + w_n/2
        y_mid_n = y_n + h_n/2
        return [x_mid_n, y_mid_n, w_n, h_n]

def bbox_center(bbox, from_format, image_size=(800, 600)):
    """calculate the center of the bounding box"""
    x_center, y_center, *_ = convert_bbox(bbox, from_format, BBox.XMIDYMIDWH_NORM, image_size=image_size)
    return x_center, y_center

if __name__ == '__main__':
    xywh_bbox = [100, 90, 50, 40]
    print(convert_bbox(xywh_bbox, from_format=BBox.XYWH, to_format=BBox.X1Y1X2Y2))
    print(convert_bbox(xywh_bbox, from_format=BBox.XYWH, to_format=BBox.XMIDYMIDWH_NORM))