import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.preprocess-deps'))
from preprocessing import PhotoPreprocessor,InputRejected,validate_landmarks
class PreprocessingTests(unittest.TestCase):
    def points(self):
        p=[SimpleNamespace(x=.5,y=.5,visibility=1.,presence=1.) for _ in range(33)]
        p[11].x=.3;p[12].x=.7;p[11].y=p[12].y=.2;p[23].y=p[24].y=.6
        return p
    def test_valid_pose(self):
        for category in ('upper','lower','overall'):validate_landmarks([self.points()],category)
    def test_invalid_poses(self):
        for poses in ([],[self.points(),self.points()],[self.points()[:20]]):
            with self.assertRaises(InputRejected):validate_landmarks(poses,'upper')
    def test_missing_and_nonfinite_landmarks(self):
        for value in (0.,float('nan')):
            p=self.points();p[11].visibility=value
            with self.assertRaises(InputRejected):validate_landmarks([p],'upper')
    def test_category_and_lower_visibility(self):
        p=self.points();p[28].presence=0
        validate_landmarks([p],'upper')
        for category in ('lower','overall','bad'):
            with self.assertRaises(InputRejected):validate_landmarks([p],category)
    def test_existing_alpha_preserved(self):
        p=PhotoPreprocessor.__new__(PhotoPreprocessor)
        im=Image.new('RGBA',(128,128),(0,0,0,0));im.paste((20,40,200,255),(32,32,96,96))
        with patch('rembg.remove') as remove:
            out=p.clean_garment(im);remove.assert_not_called()
        self.assertEqual(out.getpixel((0,0)),(255,255,255));self.assertEqual(out.getpixel((out.width//2,out.height//2)),(20,40,200))
    def test_empty_garments_rejected(self):
        p=PhotoPreprocessor.__new__(PhotoPreprocessor)
        for im in (Image.new('RGB',(128,128),'white'),Image.new('RGBA',(128,128),(0,0,0,0)),Image.new('RGB',(32,32))):
            with self.assertRaises(InputRejected):p.clean_garment(im)
if __name__=='__main__':unittest.main()
