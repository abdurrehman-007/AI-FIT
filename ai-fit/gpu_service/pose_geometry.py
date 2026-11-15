"""Pixel-space pose anchors and a conservative output pose check.

These are geometric controls for the inpainting mask, not a trained pose adapter.
"""
import cv2
import numpy as np


def visible(point, minimum=.5):
    values=(point.x, point.y, point.visibility, point.presence)
    return bool(np.isfinite(values).all() and point.visibility >= minimum and point.presence >= minimum
                and 0 <= point.x <= 1 and 0 <= point.y <= 1)


def hand_protection(points, size, *, protect_wrist=False):
    width, height = size
    result=np.zeros((height,width),dtype=np.uint8)
    if points is None or len(points)!=33:
        return result
    shoulder_span=np.hypot((points[11].x-points[12].x)*width,(points[11].y-points[12].y)*height)
    if not np.isfinite(shoulder_span):
        return result
    radius=max(2,round(shoulder_span*.035))
    for elbow,indices in ((13,(15,17,19,21)),(14,(16,18,20,22))):
        # Do not paint a guessed hand when only an occluded wrist is available.
        if not visible(points[indices[0]]):
            continue
        coords=[(round(points[i].x*(width-1)),round(points[i].y*(height-1))) for i in indices if visible(points[i])]
        if protect_wrist and visible(points[elbow],.55):
            # Retain the wrist joint and nearby accessories at the cuff.
            wrist_radius = max(radius,round(shoulder_span*.065))
            cv2.circle(result,coords[0],wrist_radius,255,-1)
        if len(coords)<2:
            continue
        shape=np.zeros_like(result)
        cv2.fillConvexPoly(shape,cv2.convexHull(np.array(coords,dtype=np.int32)),255)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
        result=np.maximum(result,cv2.dilate(shape,kernel))
    return result


def pose_difference(before, after, size, category):
    """No alignment: source and output must retain coordinates in the same image.

    Thresholds are conservative engineering checks, not calibrated probabilities.
    """
    width,height=size
    if len(before)!=33 or len(after)!=33:
        raise ValueError('Pose comparison requires 33 landmarks.')
    shoulder=np.array([(before[11].x+before[12].x)*width/2,(before[11].y+before[12].y)*height/2])
    hip=np.array([(before[23].x+before[24].x)*width/2,(before[23].y+before[24].y)*height/2])
    scale=max(float(np.linalg.norm(hip-shoulder)),1.)
    indices=[11,12,13,14,15,16,23,24]
    if category in ('lower','overall'):
        indices += [25,26,27,28]
    movement={};missing=[]
    for i in indices:
        if not visible(before[i], .65 if i in (25,26,27,28) else .5):
            continue
        if not visible(after[i],.4):
            missing.append(i);continue
        movement[i]=float(np.hypot((before[i].x-after[i].x)*width,(before[i].y-after[i].y)*height)/scale)
    shifted=[i for i,v in movement.items() if v > (.22 if i in (13,14,15,16,27,28) else .28)]
    return {'passed':not missing and not shifted,'missing':missing,'shifted':shifted,'displacement':movement}


def leg_corridor(points, size):
    """Bound semantic-mask repair to the subject's lower legs, not scene people."""
    width,height=size;mask=np.zeros((height,width),dtype=np.uint8)
    if points is None or len(points)!=33:return mask
    span=abs(points[11].x-points[12].x)*width
    for knee,ankle in ((25,27),(26,28)):
        if not visible(points[knee],.4) or not visible(points[ankle],.4):continue
        start=(round(points[knee].x*(width-1)),round(points[knee].y*(height-1)))
        end=(round(points[ankle].x*(width-1)),round(points[ankle].y*(height-1)))
        cv2.line(mask,start,end,255,max(3,round(span*.40)))
    return mask


def foot_protection(points, size, labels):
    """Preserve footwear edges without retaining misclassified trouser legs."""
    width,height=size
    shoes=np.isin(labels,(18,19)).astype(np.uint8)*255
    span=width*.25
    if points is not None and len(points)==33 and visible(points[11],.4) and visible(points[12],.4):
        span=abs(points[11].x-points[12].x)*width
    radius=max(2,round(span*.07))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    result=cv2.dilate(shoes,kernel)
    if points is None or len(points)!=33:return result
    for heel,toe in ((29,31),(30,32)):
        if not visible(points[heel],.4) or not visible(points[toe],.65):continue
        a=(round(points[heel].x*(width-1)),round(points[heel].y*(height-1)))
        b=(round(points[toe].x*(width-1)),round(points[toe].y*(height-1)))
        cv2.line(result,a,b,255,max(3,radius*2))
    return result


def upper_arm_editable(points, size):
    """Allow the sleeve cap to change around reliable shoulder landmarks."""
    width,height = size
    mask = np.zeros((height,width),dtype=np.uint8)
    if len(points) != 33 or not all(visible(points[i], .65) for i in (11,12)):
        return mask
    span = abs(points[11].x-points[12].x)*width
    for shoulder, elbow in ((11,13),(12,14)):
        a,b = points[shoulder],points[elbow]
        if not visible(b,.65):
            continue
        start = (round(a.x*(width-1)),round(a.y*(height-1)))
        end = (round((a.x+.35*(b.x-a.x))*(width-1)),
               round((a.y+.35*(b.y-a.y))*(height-1)))
        cv2.line(mask,start,end,255,max(3,round(span*.20)))
    return mask


def sleeve_corridor(points, size):
    """Make the visible arm available to short- and long-sleeve inpainting."""
    width,height = size
    mask = np.zeros((height,width),dtype=np.uint8)
    if points is None or len(points)!=33 or not all(visible(points[i],.55) for i in (11,12)):
        return mask
    span = np.hypot((points[11].x-points[12].x)*width,
                    (points[11].y-points[12].y)*height)
    for indices in ((11,13,15),(12,14,16)):
        shoulder,elbow,wrist = (points[i] for i in indices)
        for a,b,width_fraction in ((shoulder,elbow,.52),(elbow,wrist,.42)):
            if visible(a,.40) and visible(b,.40):
                start = (round(a.x*(width-1)),round(a.y*(height-1)))
                end = (round(b.x*(width-1)),round(b.y*(height-1)))
                cv2.line(mask,start,end,255,max(3,round(span*width_fraction)))
    return mask
