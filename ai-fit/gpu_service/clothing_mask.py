"""CPU clothing segmentation and a garment-focused inference crop."""
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from model.SCHP import SCHP
from pose_geometry import hand_protection, leg_corridor, foot_protection, upper_arm_editable, sleeve_corridor


class ClothingMask:
    def __init__(self, checkpoint):
        self.parser = SCHP(str(checkpoint), device='cpu')

    def _labels(self, image):
        # The Windows SCHP PIL path expects BGR.
        bgr = Image.fromarray(np.asarray(image)[:, :, ::-1].copy())
        with torch.inference_mode():
            return np.asarray(self.parser(bgr))

    def restore_exposed_skin(self, original, preview):
        """Keep the source tone where the preview still shows exposed skin."""
        original_skin = np.isin(self._labels(original), (13, 14, 15))
        visible_skin = np.isin(self._labels(preview), (13, 14, 15)).astype(np.uint8)
        # Keep garment boundary pixels from the preview so new cuffs and
        # collars do not inherit skin from under the shirt.
        visible_skin = cv2.erode(visible_skin, np.ones((5, 5),dtype=np.uint8))
        shared_skin = (original_skin & (visible_skin>0)).astype(np.uint8)
        distance = cv2.distanceTransform(shared_skin, cv2.DIST_L2, 3)
        # Feather inward so the skin correction does not create a cutout
        # seam at the wrist or bleed into adjacent cloth.
        mask = Image.fromarray(np.clip(distance*(255/5),0,255).astype(np.uint8))
        return Image.composite(original,preview,mask)

    def garment_only(self, image, category):
        """Remove donor skin/head/background from a detected worn-garment image."""
        rgb=image.convert('RGB')
        with torch.inference_mode():
            labels=np.asarray(self.parser(Image.fromarray(np.asarray(rgb)[:,:,::-1].copy())))
        targets={'upper':(5,6,7,10,11), 'lower':(9,12), 'overall':(5,6,7,9,10,11,12)}[category]
        keep=np.isin(labels,targets)
        if np.count_nonzero(keep)<labels.size*.01:
            raise ValueError('Could not isolate the outfit from its model. Use a garment-only product photo.')
        pixels=np.asarray(rgb).copy();pixels[~keep]=255
        ys,xs=np.where(keep);pad=max(3,round((xs.max()-xs.min()+1)*.04))
        box=(max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(rgb.width,int(xs.max())+pad+1),min(rgb.height,int(ys.max())+pad+1))
        return Image.fromarray(pixels).crop(box)

    def prepare(self, person, category, size=(768, 1024), landmarks=None, include_protection=False, foreground=None):
        labels = self._labels(person)
        targets = {'upper': (5, 6, 7, 10, 11), 'lower': (9, 12, 16, 17),
                   'overall': (5, 6, 7, 9, 10, 11, 12, 16, 17)}[category]
        clothing = np.isin(labels, targets).astype(np.uint8) * 255
        repair=np.zeros_like(labels,dtype=bool)
        if category in ('lower','overall') and foreground is not None:
            # SCHP can label pale trousers as background. Repair only pixels
            # independently marked as this person's foreground and near legs.
            repair=(np.asarray(foreground)>.8)&(leg_corridor(landmarks,person.size)>0)&(labels==0)
            clothing[repair]=255
        if np.count_nonzero(clothing) < labels.size * .005:
            raise ValueError('Could not identify the clothing to replace. Use a clear front-facing photo.')
        ys, xs = np.where(clothing > 0)
        x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
        garment_width = x1-x0
        radius = max(7, round(garment_width * .05))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        hem_band = np.zeros_like(clothing, dtype=bool)
        if category == 'upper':
            # A button-up's split hem must not become a permanent opening in
            # a replacement T-shirt. Permit a continuous hem in this narrow band.
            hem_top = max(y0, y1-round(garment_width*.15))
            hem_band[hem_top:min(person.height, y1+radius+1)] = True
            hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
            filled = np.zeros_like(clothing)
            cv2.fillConvexPoly(filled, hull, 255)
            clothing[hem_band] = np.maximum(clothing[hem_band], filled[hem_band])
        mask = cv2.dilate(clothing, kernel)
        edge_repair=np.zeros_like(labels,dtype=bool)
        if category == 'overall' and foreground is not None:
            # Human parsing can miss pale fabric at the silhouette. Permit the
            # existing narrow dilation only where person segmentation agrees.
            edge_repair=(mask>0)&(np.asarray(foreground)>.5)&(labels==0)
        # Keep the face, hair and unrelated clothes out of the editable area.
        protected = np.isin(labels, (1, 2, 4, 13, 18, 19))
        if category == 'upper':
            protected |= np.isin(labels, (9, 12)) & ~hem_band
            protected |= np.isin(labels, (16, 17))
            arms = np.isin(labels, (14, 15))
            sleeves = arms & (sleeve_corridor(landmarks,person.size)>0)
            # Include the arm outline so a long sleeve can extend slightly
            # beyond bare skin. Keep the margin bounded by pose and scale.
            sleeve_radius = max(2,round(garment_width*.015))
            sleeve_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,(2*sleeve_radius+1,2*sleeve_radius+1))
            sleeve_edges = cv2.dilate(sleeves.astype(np.uint8),sleeve_kernel)>0
            mask[sleeve_edges & (labels==0)] = 255
            mask[sleeves] = 255
            if foreground is not None:
                # Preserve scene pixels outside a narrow band around the
                # person, even when the parser overestimates shirt edges.
                silhouette = (np.asarray(foreground)>.45).astype(np.uint8)
                allowed_background = cv2.dilate(silhouette,sleeve_kernel)>0
                protected |= (labels==0) & ~allowed_background
            rows = np.arange(person.height)[:, None]
            protected |= arms & ~sleeves & (rows > y0+(y1-y0)*.5)
        if category == 'lower': protected |= np.isin(labels, (5, 7, 14, 15))
        if category == 'overall':
            # Arms/hands anchor pose. Leg labels are editable for a full outfit:
            # pale trousers are frequently misclassified as exposed legs, and
            # retaining those pixels leaks the old colour into the new outfit.
            # Shoes remain protected above; output joints are checked separately.
            arm = np.isin(labels,(14,15))
            # Let sleeves extend across the upper arm, while retaining the
            # forearm and hand that establish the person's pose.
            if landmarks is not None:
                arm &= upper_arm_editable(landmarks, person.size) == 0
            protected |= arm
            protected |= (labels==0)&~repair&~edge_repair
        if category in ('lower','overall'):
            foot = foot_protection(landmarks,person.size,labels)>0
            if category == 'overall':
                foot &= ~np.isin(labels,(9,12))
            protected |= foot
        protected |= hand_protection(landmarks,person.size,protect_wrist=category=='upper')>0
        mask[protected] = 0
        # Focus resolution on the garment while retaining head/body context.
        left, right = x0-garment_width*.25, x1+garment_width*.25
        top, bottom = y0-(y1-y0)*.15, y1+(y1-y0)*.15
        if category == 'upper' and sleeves.any():
            sleeve_y,sleeve_x = np.where(sleeves & (mask>0))
            if sleeve_x.size:
                left,right = min(left,sleeve_x.min()-radius),max(right,sleeve_x.max()+radius+1)
                top,bottom = min(top,sleeve_y.min()-radius),max(bottom,sleeve_y.max()+radius+1)
        if category in ('upper', 'overall'):
            head_y = np.where(np.isin(labels, (2, 13)))[0]
            if head_y.size: top = min(top, head_y.min()-garment_width*.08)
        crop_w, crop_h = right-left, bottom-top
        crop_w = max(crop_w, crop_h*.75)
        crop_h = max(crop_h, crop_w/.75)
        scale = min(1., person.width/crop_w, person.height/crop_h)
        crop_w, crop_h = crop_w*scale, crop_h*scale
        left = max(0, min((left+right-crop_w)/2, person.width-crop_w))
        top = max(0, min((top+bottom-crop_h)/2, person.height-crop_h))
        box = (round(left), round(top), round(left+crop_w), round(top+crop_h))
        feather = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(2))
        feather_array = np.array(feather)
        feather_array[protected] = 0
        feather = Image.fromarray(feather_array)
        prepared=(person.crop(box).resize(size, Image.Resampling.LANCZOS),
                  feather.crop(box).resize(size, Image.Resampling.BILINEAR), box)
        if include_protection:
            # Retain full-resolution protection; small fingers can disappear
            # during the 384x512 roundtrip otherwise.
            return (*prepared,Image.fromarray(protected.astype(np.uint8)*255))
        return prepared


def restore_crop(output, original, mask, box, protected=None):
    size = (box[2]-box[0], box[3]-box[1])
    result = original.copy()
    alpha = np.asarray(mask.resize(size, Image.Resampling.BILINEAR).convert('L'))
    # Soft mask edges otherwise blend the original garment colour into the result.
    alpha = np.where(alpha >= 96, 255, 0).astype(np.uint8)
    result.paste(output.resize(size, Image.Resampling.LANCZOS), box[:2], Image.fromarray(alpha))
    if protected is not None:
        result.paste(original,(0,0),protected.convert('L'))
    return result
