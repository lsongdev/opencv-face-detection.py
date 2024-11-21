import os
import cv2
import time
import zeep
import queue
import urllib.parse
from threading import Thread
from onvif import ONVIFCamera

def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue

zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue

class FaceDetector:
    def __init__(self):
        print("加载人脸检测模型...")
        
        self.model_file = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        self.config_file = "models/deploy.prototxt"
        
        if not os.path.exists('models'):
            os.makedirs('models')
            
        if not os.path.exists(self.model_file):
            print("下载人脸检测模型...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                self.model_file
            )
            
        if not os.path.exists(self.config_file):
            print("下载模型配置文件...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                self.config_file
            )

        self.net = cv2.dnn.readNet(self.model_file, self.config_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.conf_threshold = 0.5
        self.frame_skip = 2

    def detect_faces(self, frame):
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), 
            [104, 117, 123], 
            False, False
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                w = x2 - x1
                h = y2 - y1
                
                faces.append((x1, y1, w, h, confidence))
        
        return faces

class FaceTracker:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.center_x = None
        self.center_y = None
        self.center_zone = 100
        self.last_move_time = time.time()
        self.move_cooldown = 0.5
        self.frame_count = 0
        self.last_face = None
        self.lost_frames = 0
        self.max_lost_frames = 10

    def draw_analysis_results(self, frame, face, analysis):
        """在画面上显示分析结果"""
        if analysis is None:
            return
            
        x, y, w, h, _ = face
        
        # 设置文本参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # 准备显示的信息
        lines = [
            f"Age: {analysis['age']:.0f}",
            f"Gender: {analysis['gender']}",
            f"Emotion: {analysis['emotion']}",
            f"Race: {analysis['race']}"
        ]
        
        # 计算文本框的大小
        padding = 5
        line_height = 20
        box_height = len(lines) * line_height + 2 * padding
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y-box_height), (x+w, y), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # 绘制文本
        for i, line in enumerate(lines):
            y_pos = y - box_height + padding + i * line_height + 15
            cv2.putText(frame, line, (x+5, y_pos), 
                       font, font_scale, (255, 255, 255), thickness)

    def get_direction(self, face_center_x, face_center_y):
        if face_center_x is None or face_center_y is None:
            return None
            
        dx = face_center_x - self.center_x
        dy = face_center_y - self.center_y

        if abs(dx) <= self.center_zone and abs(dy) <= self.center_zone:
            return 'center'

        if dx < -self.center_zone:
            h_dir = 'left'
        elif dx > self.center_zone:
            h_dir = 'right'
        else:
            h_dir = ''

        if dy < -self.center_zone:
            v_dir = 'top'
        elif dy > self.center_zone:
            v_dir = 'bottom'
        else:
            v_dir = ''

        if h_dir and v_dir:
            return f"{v_dir}_{h_dir}"
        elif h_dir:
            return h_dir
        elif v_dir:
            return v_dir
        else:
            return 'center'

    def process_frame(self, frame):
        if self.center_x is None:
            height, width = frame.shape[:2]
            self.center_x = width // 2
            self.center_y = height // 2

        self.frame_count += 1
        direction = None
        current_time = time.time()
        
        faces = self.face_detector.detect_faces(frame)
        
        if faces:
            self.lost_frames = 0
            face = max(faces, key=lambda x: x[4])
            x, y, w, h, conf = face
            
            face_center_x = x + w//2
            face_center_y = y + h//2
            self.last_face = face
            
            # 绘制人脸框和中心点
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)
            
            if current_time - self.last_move_time >= self.move_cooldown:
                direction = self.get_direction(face_center_x, face_center_y)
                if direction and direction != 'center':
                    self.last_move_time = current_time
        else:
            self.lost_frames += 1
            direction = None
            if current_time - self.last_move_time >= self.move_cooldown:
                if self.lost_frames >= self.max_lost_frames:
                    self.last_face = None

        # 绘制参考线和区域
        cv2.rectangle(frame, 
                     (self.center_x - self.center_zone, self.center_y - self.center_zone),
                     (self.center_x + self.center_zone, self.center_y + self.center_zone),
                     (0, 255, 0), 1)
        
        h, w = frame.shape[:2]
        cv2.line(frame, (w//3, 0), (w//3, h), (128, 128, 128), 1)
        cv2.line(frame, (2*w//3, 0), (2*w//3, h), (128, 128, 128), 1)
        cv2.line(frame, (0, h//3), (w, h//3), (128, 128, 128), 1)
        cv2.line(frame, (0, 2*h//3), (w, 2*h//3), (128, 128, 128), 1)

        return frame, direction


class VideoStreamReader:
    def __init__(self, url):
        self.url = url
        self.frame_queue = queue.Queue(maxsize=2)
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        cap = cv2.VideoCapture(self.url)
        while not self.stopped:
            if not cap.isOpened():
                print("无法打开视频流")
                break
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
        cap.release()

    def read(self):
        return self.frame_queue.get()

    def stop(self):
        self.stopped = True

class PTZController:
    def __init__(self, ip, port, user, password):
        print(f"\n初始化 PTZ 控制器... IP: {ip}, Port: {port}")
        self.mycam = ONVIFCamera(ip, port, user, password)
        
        # 初始化 PTZ 服务
        self.ptz = self.mycam.create_ptz_service()
        self.media = self.mycam.create_media_service()
        self.media_profile = self.media.GetProfiles()[0]
        
        # 获取 PTZ 配置
        request = self.ptz.create_type('GetConfiguration')
        request.PTZConfigurationToken = self.media_profile.PTZConfiguration.token
        self.ptz_configuration = self.ptz.GetConfiguration(request)
        
        # 获取 PTZ 配置选项
        request = self.ptz.create_type('GetConfigurationOptions')
        request.ConfigurationToken = self.media_profile.PTZConfiguration.token
        self.ptz_configuration_options = self.ptz.GetConfigurationOptions(request)
        
        # 获取流地址
        stream_setup = {
            'Stream': 'RTP-Unicast',
            'Transport': {
                'Protocol': 'RTSP'
            }
        }
        self.stream_uri = self.media.GetStreamUri({
            'StreamSetup': stream_setup,
            'ProfileToken': self.media_profile.token
        })
        
        # 构建带认证的RTSP URL
        parsed_uri = urllib.parse.urlparse(self.stream_uri.Uri)
        self.rtsp_url = f"rtsp://{user}:{password}@{parsed_uri.hostname}:{parsed_uri.port}{parsed_uri.path}?{parsed_uri.query}"
        print(f"RTSP URL: {self.rtsp_url}")
        
        # 创建移动请求
        self.request = self.ptz.create_type('ContinuousMove')
        self.request.ProfileToken = self.media_profile.token
        self.request.Velocity = {
            'PanTilt': {'x': 0, 'y': 0},
            'Zoom': {'x': 0}
        }
        
    def get_status(self):
        """更新当前位置状态"""
        return self.ptz.GetStatus({'ProfileToken': self.media_profile.token})

    def get_position(self):
        """获取当前位置"""
        return self.get_status().Position

    def absolute_move(self, pan=None, tilt=None, zoom=None):
        """
        移动到指定的绝对位置
        :param pan: 水平角度 (-1.0 到 1.0)
        :param tilt: 垂直角度 (-1.0 到 1.0)
        :param zoom: 缩放级别 (0.0 到 1.0)
        """
        try:
            # 创建绝对移动请求
            request = self.ptz.create_type('AbsoluteMove')
            request.ProfileToken = self.media_profile.token

            # 创建位置对象
            pos = self.get_position()
            if pan is not None or tilt is not None:
                if pan is not None:
                    pos.PanTilt.x = max(self.XMIN, min(self.XMAX, pan))
                if tilt is not None:
                    pos.PanTilt.y = max(self.YMIN, min(self.YMAX, tilt))

            # 设置缩放
            if zoom is not None:
                pos.Zoom.x = max(self.ZMIN, min(self.ZMAX, zoom))
            
            request.Position = pos
            print(f"移动到 - Pan: {pan}, Tilt: {tilt}, Zoom: {zoom}")
            self.ptz.AbsoluteMove(request)
            self.wait_for_stop(pos)
        except Exception as e:
            print(f"绝对位置移动失败: {str(e)}")

    def get_stream_url(self):
        """返回RTSP流地址"""
        return self.rtsp_url

    def step_move(self, direction, step = 1):
        """
        根据方向执行单步移动
        direction: 'top_left', 'top', 'top_right', 'left', 'right', 
                  'bottom_left', 'bottom', 'bottom_right', 'center', None
        """
        pan, tilt = 0, 0
        
        if direction is None or direction == 'center':
            return
            
        # 设置移动参数
        if 'left' in direction:
            pan = -step
        elif 'right' in direction:
            pan = +step
            
        if 'top' in direction:
            tilt = +step
        elif 'bottom' in direction:
            tilt = -step
        print(f"移动方向: {direction}, Pan: {pan}, Tilt: {tilt}")
        try:
            self.request.Velocity['PanTilt']['x'] = pan
            self.request.Velocity['PanTilt']['y'] = tilt
            self.ptz.ContinuousMove(self.request)
            # 移动一小段时间后停止
            time.sleep(0.2)
            self.stop()
        except Exception as e:
            print(f"移动错误: {str(e)}")

    def stop(self):
        """停止移动"""
        try:
            self.ptz.Stop({'ProfileToken': self.media_profile.token})
        except Exception as e:
            print(f"停止错误: {str(e)}")

def main():
    CAMERA_IP = "192.168.2.224"
    CAMERA_PORT = 80
    CAMERA_USER = "admin"
    CAMERA_PASS = "test123456"

    try:
        print("\n初始化系统...")
        ptz = PTZController(CAMERA_IP, CAMERA_PORT, CAMERA_USER, CAMERA_PASS)
        rtsp_url = ptz.get_stream_url()
        
        tracker = FaceTracker()
        stream = VideoStreamReader(rtsp_url).start()
        
        print("\n开始人脸追踪和分析...")
        print("按 'q' 退出")
        
        while True:
            frame = stream.read()
            processed_frame, direction = tracker.process_frame(frame)
            
            if direction and direction != 'center':
                ptz.step_move(direction)
           
            cv2.imshow('Face Tracking', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\n程序错误: {str(e)}")
    finally:
        print("\n清理资源...")
        stream.stop()
        cv2.destroyAllWindows()
        ptz.stop()
        print("程序已退出")

if __name__ == "__main__":
    main()

