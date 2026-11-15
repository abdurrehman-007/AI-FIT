import io,sys,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import local_windows as service
class LocalWindowsTests(unittest.TestCase):
    def test_health_without_cuda_is_503(self):
        with patch.object(service,'pipeline',None),patch.object(service.torch.cuda,'is_available',return_value=False):
            response=service.app.test_client().get('/health')
        self.assertEqual(response.status_code,503);self.assertFalse(response.json['ready']);self.assertIsNone(response.json['gpu'])
    def test_authorization_and_startup(self):
        client=service.app.test_client()
        self.assertEqual(client.post('/infer').status_code,401)
        with patch.object(service,'pipeline',None):
            self.assertEqual(client.post('/infer',headers={'Authorization':'Bearer '+service.KEY}).status_code,503)
    def test_route_preserves_garment_alpha(self):
        prep=Mock();prep.validate_person.return_value=([],None);prep.clean_garment.side_effect=ValueError('stop before inference')
        person=io.BytesIO();Image.new('RGB',(128,128)).save(person,format='PNG');person.seek(0)
        garment=io.BytesIO();Image.new('RGBA',(128,128),(20,30,40,0)).save(garment,format='PNG');garment.seek(0)
        with patch.object(service,'pipeline',Mock()),patch.object(service,'clothing_mask',Mock()),patch.object(service,'preprocessor',prep):
            response=service.app.test_client().post('/infer',data={'person':(person,'p.png'),'garment':(garment,'g.png')},headers={'Authorization':'Bearer '+service.KEY})
        self.assertEqual(response.status_code,422)
        im=prep.clean_garment.call_args.args[0];self.assertEqual(im.mode,'RGBA');self.assertEqual(im.getpixel((0,0))[3],0)
        self.assertFalse(service.lock.locked())
    def test_route_uses_pose_and_rejects_changed_output(self):
        from preprocessing import InputRejected
        prep=Mock();points=[object()]*33;prep.validate_person.return_value=(points,None)
        prep.clean_garment.return_value=Image.new('RGB',(128,128),'white')
        prep.garment_has_person.return_value=False
        prep.check_output_pose.side_effect=InputRejected('pose changed')
        masker=Mock();masker.prepare.return_value=(Image.new('RGB',(128,128)),Image.new('L',(128,128),255),(0,0,128,128),Image.new('L',(128,128)))
        pipeline=Mock(return_value=[Image.new('RGB',(128,128),'blue')]);pipeline.last_safety_blocked=False
        def photo():
            stream=io.BytesIO();Image.new('RGB',(128,128)).save(stream,format='PNG');stream.seek(0);return stream
        with patch.object(service,'pipeline',pipeline),patch.object(service,'clothing_mask',masker),patch.object(service,'preprocessor',prep):
            response=service.app.test_client().post('/infer',data={'person':(photo(),'p.png'),'garment':(photo(),'g.png'),'category':'overall'},headers={'Authorization':'Bearer '+service.KEY})
        self.assertEqual(response.status_code,422)
        self.assertIs(masker.prepare.call_args.kwargs['landmarks'],points)
        self.assertFalse(service.lock.locked())
    def test_cuda_runtime_oom_is_retryable_and_releases_lock(self):
        def image():
            stream=io.BytesIO();Image.new('RGB',(128,128)).save(stream,format='PNG');stream.seek(0);return stream
        for error,status in [(RuntimeError('CUDA error: out of memory'),503),(RuntimeError('tensor shape mismatch'),500)]:
            with self.subTest(status=status):
                prep=Mock();prep.validate_person.side_effect=error
                with patch.object(service,'pipeline',Mock()),patch.object(service,'clothing_mask',Mock()),patch.object(service,'preprocessor',prep),patch.object(service.torch.cuda,'empty_cache') as clear,patch.object(service.app.logger,'exception'):
                    response=service.app.test_client().post('/infer',data={'person':(image(),'p.png'),'garment':(image(),'g.png')},headers={'Authorization':'Bearer '+service.KEY})
                self.assertEqual(response.status_code,status)
                self.assertEqual(clear.call_count,1 if status==503 else 0)
                self.assertFalse(service.lock.locked())
                self.assertEqual(response.mimetype,'application/json')
if __name__=='__main__':unittest.main()
