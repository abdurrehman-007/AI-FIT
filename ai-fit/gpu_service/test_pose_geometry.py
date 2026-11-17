import copy,unittest
from types import SimpleNamespace
from pose_geometry import hand_protection,pose_difference,foot_protection,neck_protection,body_width_difference
import numpy as np

class PoseGeometryTests(unittest.TestCase):
    def points(self):
        p=[SimpleNamespace(x=.5,y=.5,visibility=1.,presence=1.) for _ in range(33)]
        p[11].x=.3;p[12].x=.7;p[11].y=p[12].y=.2;p[23].y=p[24].y=.6
        p[15].x=.45;p[15].y=.48
        for i in (17,19,21):p[i].x=.48;p[i].y=.52
        return p
    def test_protects_hand_not_entire_arm(self):
        mask=hand_protection(self.points(),(300,400))
        self.assertGreater(mask[200,140],0);self.assertEqual(mask[100,90],0)

    def test_upper_wrist_guard_extends_protection_near_cuff(self):
        points = self.points()
        normal = hand_protection(points,(300,400))
        guarded = hand_protection(points,(300,400),protect_wrist=True)
        self.assertEqual(normal[185,135],0)
        self.assertEqual(guarded[185,135],255)
        self.assertEqual(guarded[160,135],0)
    def test_occluded_landmarks_not_guessed(self):
        p=self.points()
        for i in (15,16):p[i].visibility=0
        self.assertEqual(hand_protection(p,(300,400)).sum(),0)
    def test_identical_pose_passes(self):
        p=self.points();self.assertTrue(pose_difference(p,p,(300,400),'overall')['passed'])

    def test_neck_stretch_rejected_before_large_joint_displacement(self):
        points=self.points()
        points[9].y=points[10].y=.13
        after=copy.deepcopy(points)
        after[11].y=after[12].y=.25
        report=pose_difference(points,after,(300,400),'upper')
        self.assertEqual(report['shifted'],[])
        self.assertTrue(report['neck_changed'])
        self.assertFalse(report['passed'])

    def test_neck_guard_preserves_skin_without_preserving_shirt(self):
        labels=np.zeros((400,300),dtype=np.uint8)
        labels[50:85,140:160]=14
        labels[85:140,100:200]=5
        mask=neck_protection(self.points(),(300,400),labels)
        self.assertEqual(mask[75,150],255)
        self.assertEqual(mask[110,150],0)
        points=self.points();points[11].visibility=0
        self.assertEqual(neck_protection(points,(300,400),labels).sum(),0)

    def test_body_width_guard_detects_shrinking_with_identical_joints(self):
        from PIL import Image
        source=np.zeros((400,300),dtype=np.float32)
        source[50:300,90:210]=1
        narrow=np.zeros_like(source);narrow[50:300,120:180]=1
        report=body_width_difference(Image.fromarray(source),Image.fromarray(narrow),self.points(),(300,400))
        self.assertTrue(report['available'])
        self.assertFalse(report['passed'])
        self.assertTrue(body_width_difference(Image.fromarray(source),Image.fromarray(source),self.points(),(300,400))['passed'])

    def test_body_width_does_not_use_scene_people_for_width(self):
        from PIL import Image
        source=np.zeros((400,300),dtype=np.float32);source[50:300,90:210]=1
        other=source.copy();other[50:300,5:45]=1
        report=body_width_difference(Image.fromarray(source),Image.fromarray(other),self.points(),(300,400))
        self.assertEqual(report['ratios'],[1.,1.,1.,1.])

    def test_single_pixel_segmentation_hole_does_not_shrink_body(self):
        from PIL import Image
        source=np.zeros((400,300),dtype=np.float32);source[50:300,90:210]=1
        output=source.copy();output[136,150]=output[160,150]=0
        report=body_width_difference(Image.fromarray(source),Image.fromarray(output),self.points(),(300,400))
        self.assertTrue(report['passed'])
        self.assertEqual(report['ratios'],[1.,1.,1.,1.])
    def test_changed_wrist_and_ankle_rejected(self):
        for index in (15,27):
            p=self.points();after=copy.deepcopy(p);after[index].x+=.2
            self.assertFalse(pose_difference(p,after,(300,400),'overall')['passed'])
    def test_missing_visible_output_joint_rejected(self):
        p=self.points();after=copy.deepcopy(p);after[15].presence=0
        self.assertIn(15,pose_difference(p,after,(300,400),'upper')['missing'])

    def test_uncertain_ankle_does_not_override_reliable_wrist(self):
        p=self.points();p[27].visibility=.54;after=copy.deepcopy(p);after[27].x+=.2
        self.assertTrue(pose_difference(p,after,(300,400),'overall')['passed'])
        after[15].x+=.2
        self.assertFalse(pose_difference(p,after,(300,400),'overall')['passed'])

    def test_footwear_edge_protection_does_not_cover_trousers(self):
        labels=np.zeros((400,300),dtype=np.uint8);labels[365:380,140:160]=18
        mask=foot_protection(None,(300,400),labels)
        self.assertGreater(mask[363,150],0)
        self.assertEqual(mask[320,150],0)

    def test_unreliable_foot_landmarks_do_not_create_protection(self):
        p=self.points()
        for i in (29,30,31,32):p[i].visibility=.1
        self.assertEqual(foot_protection(p,(300,400),np.zeros((400,300),dtype=np.uint8)).sum(),0)
