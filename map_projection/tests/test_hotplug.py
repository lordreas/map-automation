from pathlib import Path
import tempfile
import unittest
from map_projection.hotplug import Monitor,Recovery


class Engine:
    gpu=False
    output_status={}
    def set_gpu(self,value):self.gpu=value


class Display:
    def __init__(self,fail=False,gpu=True):self.config=None;self.closed=False;self.gpu=gpu;self.fail=fail;self.status={'renderer':'test'}
    def configure(self,config):
        if self.fail:raise RuntimeError('EDID not ready')
        self.config=dict(config)
    def close(self):self.closed=True


class HotplugTests(unittest.TestCase):
    def test_connector_disconnect_reconnect_and_edid_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'card0-HDMI-A-1';path.mkdir();(path/'status').write_text('disconnected')
            (path/'edid').write_bytes(b'');monitor=Monitor(folder,events=False)
            s=monitor.poll();self.assertTrue(s['known']);self.assertFalse(s['connected'])
            (path/'status').write_text('connected');(path/'edid').write_bytes(b'monitor1')
            s=monitor.poll();self.assertTrue(s['connected']);self.assertTrue(s['changed'])
            self.assertFalse(monitor.poll()['changed'])
            (path/'edid').write_bytes(b'monitor2');self.assertTrue(monitor.poll()['changed'])

    def test_short_replug_is_detected_even_with_identical_edid(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'card0-HDMI-A-1';path.mkdir();(path/'status').write_text('connected')
            monitor=Monitor(folder,events=False);monitor.poll()
            class Events:
                sent=False
                def recv(self,_):
                    if self.sent:raise BlockingIOError()
                    self.sent=True;return b'SUBSYSTEM=drm\0HOTPLUG=1\0'
            monitor.socket=Events()
            self.assertTrue(monitor.poll()['changed'])
            self.assertFalse(monitor.poll()['changed'])

    def test_boot_without_hdmi_and_restore_same_config_after_replug(self):
        class Connector:
            connected=False;changed=False
            def poll(self):return {'known':True,'connected':self.connected,'changed':self.changed,'connectors':[]}
            def close(self):pass
        connector=Connector();engine=Engine();made=[]
        def factory():
            display=Display();made.append(display);return display
        recovery=Recovery(factory,engine,monitor=connector);config={'width':3840,'height':2160,'refresh':60}
        self.assertIsNone(recovery.step(0,config));self.assertEqual(made,[])
        self.assertFalse(engine.output_status['connected'])
        connector.connected=True;connector.changed=True
        display=recovery.step(.3,config);self.assertIsNotNone(display);self.assertTrue(engine.gpu)
        connector.changed=False;self.assertIs(recovery.step(.4,config),display)
        connector.connected=False;connector.changed=True
        self.assertIsNone(recovery.step(.6,config));self.assertTrue(display.closed)
        connector.connected=True
        restored=recovery.step(.9,config)
        self.assertIsNot(restored,display);self.assertEqual(restored.config,config)
        recovery.close()

    def test_failed_init_and_failed_presentation_retry_without_config_change(self):
        engine=Engine();made=[]
        def factory():
            display=Display(fail=len(made)==0);made.append(display);return display
        recovery=Recovery(factory,engine);config={'width':3840}
        self.assertIsNone(recovery.step(0,config));self.assertTrue(engine.output_status['recovering'])
        self.assertIsNone(recovery.step(.5,config));self.assertEqual(len(made),1)
        display=recovery.step(1.1,config);self.assertIsNotNone(display)
        recovery.failed(RuntimeError('surface lost'),2)
        self.assertTrue(display.closed);self.assertIsNone(recovery.step(2.5,config))
        self.assertIsNotNone(recovery.step(3.1,config));self.assertFalse(engine.output_status['recovering'])

    def test_auto_falls_back_if_compute_api_is_unavailable(self):
        engine=Engine()
        recovery=Recovery(lambda:Display(fail=True),engine,fallback=lambda:Display(gpu=False))
        display=recovery.step(0,{'width':3840})
        self.assertIsNotNone(display);self.assertFalse(display.gpu)
        self.assertIn('gpu_fallback',engine.output_status)

    def test_vendor_restart_runs_before_touching_stale_driver_resources(self):
        class Connector:
            connected=True;changed=False
            def poll(self):return {'known':True,'connected':self.connected,'changed':self.changed,'connectors':[]}
        class Restart(Exception):pass
        def restart():raise Restart()
        connector=Connector();display=Display();engine=Engine()
        recovery=Recovery(lambda:display,engine,monitor=connector,restart=restart)
        recovery.step(0,{'width':3840})
        for moment,connected in ((.3,False),(.6,True)):
            connector.connected=connected;connector.changed=True
            with self.assertRaises(Restart):recovery.step(moment,{'width':3840})
            self.assertFalse(display.closed)


if __name__=='__main__':unittest.main()
