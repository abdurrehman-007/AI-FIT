"""Preprocessing regression checks; no weights or GPU required."""
import sys
import types
import unittest
import numpy as np
from PIL import Image

# The Windows parser is supplied by the user's existing installation.
sys.modules.setdefault('model', types.ModuleType('model'))
fake_schp = types.ModuleType('model.SCHP')
fake_schp.SCHP = object
sys.modules['model.SCHP'] = fake_schp
from clothing_mask import ClothingMask, restore_crop


class ClothingMaskTests(unittest.TestCase):
    def test_restore_exposed_skin_keeps_new_sleeves(self):
        from unittest.mock import patch
        original = Image.new('RGB',(30,30),(204,138,110))
        preview = Image.new('RGB',(30,30),(44,77,65))
        source = np.full((30,30),14,dtype=np.uint8)
        output = source.copy()
        output[:,:15] = 5
        with patch.object(self.masker,'_labels',side_effect=[source,output]):
            result = self.masker.restore_exposed_skin(original,preview)
        self.assertEqual(result.getpixel((22,15)),original.getpixel((22,15)))
        self.assertEqual(result.getpixel((8,15)),preview.getpixel((8,15)))

    def setUp(self):
        self.labels = np.zeros((400, 300), dtype=np.uint8)
        self.labels[40:90, 120:180] = 13
        self.labels[100:260, 80:220] = 5
        self.labels[260:390, 90:210] = 9
        self.person = Image.new('RGB', (300, 400), (200, 80, 40))
        self.masker = ClothingMask.__new__(ClothingMask)
        self.masker.parser = lambda image: Image.fromarray(self.labels)

    def test_upper_crop_preserves_face_pants_and_background(self):
        person, mask, box = self.masker.prepare(self.person, 'upper')
        self.assertEqual(person.size, (768, 1024))
        result = restore_crop(Image.new('RGB', person.size, 'blue'), self.person, mask, box)
        self.assertEqual(result.size, self.person.size)
        for xy in ((150, 60), (150, 320), (20, 200)):
            self.assertEqual(result.getpixel(xy), self.person.getpixel(xy))
        self.assertEqual(result.getpixel((150, 180)), (0, 0, 255))

    def test_upper_preserves_background_next_to_shoulder(self):
        foreground = Image.fromarray((self.labels>0).astype(np.float32))
        _,mask,box,protected = self.masker.prepare(
            self.person,'upper',(384,512),foreground=foreground,include_protection=True)
        result = restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((75,120)),self.person.getpixel((75,120)))
        self.assertEqual(result.getpixel((90,120)),(0,0,255))

    def test_empty_clothing_does_not_fall_back_to_large_rectangle(self):
        self.labels[:] = 0
        with self.assertRaises(ValueError):
            self.masker.prepare(self.person, 'upper')

    def test_parser_receives_bgr(self):
        seen = []
        self.masker.parser = lambda image: (seen.append(image.getpixel((0, 0))) or Image.fromarray(self.labels))
        self.masker.prepare(self.person, 'upper')
        self.assertEqual(seen, [(40, 80, 200)])

    def test_split_hem_can_close_without_replacing_lower_trousers(self):
        self.labels[245:260, 140:160] = 9
        person, mask, box = self.masker.prepare(self.person, 'upper')
        result = restore_crop(Image.new('RGB', person.size, 'blue'), self.person, mask, box)
        self.assertEqual(result.getpixel((150, 252)), (0, 0, 255))
        self.assertEqual(result.getpixel((150, 320)), self.person.getpixel((150, 320)))

    def test_prepare_uses_requested_size(self):
        person, mask, box = self.masker.prepare(self.person, 'upper', (384, 512))
        self.assertEqual(person.size, (384, 512))
        self.assertEqual(mask.size, (384, 512))

    def test_hands_near_shirt_are_preserved(self):
        self.labels[215:270, 215:235] = 14
        person, mask, box = self.masker.prepare(self.person, 'upper')
        result = restore_crop(Image.new('RGB', person.size, 'blue'), self.person, mask, box)
        self.assertEqual(result.getpixel((220, 245)), self.person.getpixel((220, 245)))

    def test_upper_sleeve_can_reach_forearm_without_editing_hand_or_background(self):
        from types import SimpleNamespace
        points = [SimpleNamespace(x=0.,y=0.,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,210,110),(12,90,110),(13,230,200),(15,235,288),
                      (17,237,307),(19,240,304)):
            points[i] = SimpleNamespace(x=x/299,y=y/399,visibility=1.,presence=1.)
        self.labels[124:310,221:245] = 14
        _,mask,box,protected = self.masker.prepare(
            self.person,'upper',(384,512),landmarks=points,include_protection=True)
        result = restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((233,255)),(0,0,255))
        for point in ((238,301),(250,255),(150,320),(150,60)):
            self.assertEqual(result.getpixel(point),self.person.getpixel(point))

    def test_restore_does_not_blend_original_garment_colour(self):
        _, mask, box = self.masker.prepare(self.person, 'upper')
        faded = Image.new('L', mask.size, 80)
        faded.paste(mask.point(lambda v: 80 if v else 0))
        result = restore_crop(Image.new('RGB', mask.size, 'blue'), self.person, faded, box)
        self.assertEqual(result.getpixel((150, 180)), self.person.getpixel((150, 180)))


    def test_overall_skin_and_background_are_not_regenerated(self):
        self.labels[190:225,120:180]=14
        person,mask,box,protected=self.masker.prepare(self.person,'overall',(384,512),include_protection=True)
        result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
        for xy in ((150,205),(78,200),(150,60)):
            self.assertEqual(result.getpixel(xy),self.person.getpixel(xy))
        self.assertEqual(result.getpixel((150,300)),(0,0,255))

    def test_full_resolution_protection_survives_small_mask(self):
        self.labels[200,150]=14
        person,mask,box,protected=self.masker.prepare(self.person,'overall',(24,32),include_protection=True)
        result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((150,200)),self.person.getpixel((150,200)))

    def test_donor_extraction_removes_head_skin_and_background(self):
        self.labels[200:220,120:180]=14
        result=self.masker.garment_only(self.person,'overall')
        colors=set(map(tuple,np.asarray(result).reshape(-1,3)))
        self.assertEqual(colors,{(255,255,255),(200,80,40)})
        self.assertLess(result.height,self.person.height)
        self.labels[:]=0
        with self.assertRaises(ValueError):self.masker.garment_only(self.person,'overall')


    def test_leg_repair_requires_both_foreground_and_pose(self):
        from types import SimpleNamespace
        points=[SimpleNamespace(x=.5,y=.5,visibility=1.,presence=1.) for _ in range(33)]
        points[11].x=.3;points[12].x=.7
        for i in (25,26):points[i].y=.7
        for i in (27,28):points[i].y=.9
        self.labels[280:360,140:160]=0
        for probability,expected in ((0.,self.person.getpixel((150,320))),(1.,(0,0,255))):
            foreground=Image.fromarray(np.full((400,300),probability,dtype=np.float32))
            person,mask,box,protected=self.masker.prepare(self.person,'overall',(384,512),landmarks=points,include_protection=True,foreground=foreground)
            result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
            self.assertEqual(result.getpixel((150,320)),expected)
            self.assertEqual(result.getpixel((20,320)),self.person.getpixel((20,320)))


    def test_overall_replaces_leg_labelled_fabric_but_preserves_shoes(self):
        self.labels[280:340,120:180]=16
        self.labels[360:390,120:180]=18
        person,mask,box,protected=self.masker.prepare(self.person,'overall',(384,512),include_protection=True)
        result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((150,310)),(0,0,255))
        self.assertEqual(result.getpixel((150,375)),self.person.getpixel((150,375)))

    def test_overall_edge_repair_needs_person_foreground(self):
        for probability,expected in ((0.,self.person.getpixel((77,160))),(1.,(0,0,255))):
            foreground=Image.fromarray(np.full((400,300),probability,dtype=np.float32))
            person,mask,box,protected=self.masker.prepare(self.person,'overall',(384,512),include_protection=True,foreground=foreground)
            result=restore_crop(Image.new('RGB',person.size,'blue'),self.person,mask,box,protected)
            self.assertEqual(result.getpixel((77,160)),expected)
            self.assertEqual(result.getpixel((20,160)),self.person.getpixel((20,160)))

    def test_overall_sleeve_and_cuff_mask_preserves_forearm_and_shoe(self):
        from types import SimpleNamespace
        points = [SimpleNamespace(x=.5,y=.5,visibility=0.,presence=0.) for _ in range(33)]
        for i,x,y in ((11,.27,.26),(12,.73,.26),(13,.27,.47)):
            points[i] = SimpleNamespace(x=x,y=y,visibility=1.,presence=1.)
        self.labels[104:208,75:90] = 14
        self.labels[360:390,120:180] = 18
        _,mask,box,protected = self.masker.prepare(
            self.person,'overall',(384,512),landmarks=points,include_protection=True)
        result = restore_crop(Image.new('RGB',mask.size,'blue'),self.person,mask,box,protected)
        self.assertEqual(result.getpixel((83,114)),(0,0,255))
        self.assertEqual(result.getpixel((83,187)),self.person.getpixel((83,187)))
        self.assertEqual(result.getpixel((150,356)),(0,0,255))
        self.assertEqual(result.getpixel((150,366)),self.person.getpixel((150,366)))


if __name__ == '__main__': unittest.main()
