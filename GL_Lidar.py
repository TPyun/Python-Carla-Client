"""
RGB Camera, Lidar, Semantic Lidar 시각화 기능이 있다.
RGB Camera를 약간 어둡게 하여 Background로 출력하고 그 위에 Lidar혹은 Semantic Lidar를 출력한다.
Default는 Lidar 2개를 생성하는 것이다.
Sementic Lidar로도 변환 가능하고 코드 상의 최대 Lidar 혹은 Semantic Lidar 동시 설치 개수는 4개이다.
"""

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import carla
import random

display = (960, 690)
camera_ratio = 1
camera_brightness = 0.2


class Camera:
    def __init__(self, world, transform, attached, sensor_options):
        self.world = world
        self.sensor = self.spawn_camera(transform, attached, sensor_options)
        self.image_data = None

    def spawn_camera(self, transform, attached, sensor_options):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(display[0] * camera_ratio))
        camera_bp.set_attribute('image_size_y', str(display[1] * camera_ratio))
        camera_bp.set_attribute('fov', '95')
        for key in sensor_options:
            camera_bp.set_attribute(key, sensor_options[key])
        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
        camera.listen(self.image_callback)
        return camera

    def image_callback(self, buf):
        buf.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(buf.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (buf.height, buf.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = np.flip(array, 0)   # 좌우 대칭으로 이미지 출력
        array = array * camera_brightness   # make it little dark
        self.image_data = array


class Lidar:
    def __init__(self, world, transform, attached, sensor_options, color):
        self.world = world
        self.sensor = self.spawn_semantic_lidar(transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.image_data = None
        self.location = transform.location
        self.color = color

    def spawn_semantic_lidar(self, transform,attached, sensor_options):
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        for key in sensor_options:
            lidar_bp.set_attribute(key, sensor_options[key])
        lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)
        lidar.listen(self.image_callback)
        return lidar

    def image_callback(self, buf):
        data = np.copy(np.frombuffer(buf.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        self.image_data = data[:, :-1]
        self.image_data[:, :1] = -self.image_data[:, :1]
        self.image_data[:, :2] = -self.image_data[:, :2]
        self.image_data += np.array([self.location.y, self.location.x, self.location.z])


class SemanticLidar:
    def __init__(self, world, transform, attached, sensor_options, color):
        self.world = world
        self.sensor = self.spawn_semantic_lidar(transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.image_data = None
        self.location = transform.location
        self.color = color

    def spawn_semantic_lidar(self, transform,attached, sensor_options):
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        for key in sensor_options:
            lidar_bp.set_attribute(key, sensor_options[key])
        lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)
        lidar.listen(self.image_callback)
        return lidar

    def image_callback(self, buf):
        data = np.frombuffer(buf.raw_data, dtype=np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
        self.image_data = np.array([data['x'], -data['y'], data['z']]).T
        self.image_data += np.array([self.location.y, self.location.x, self.location.z])


class CarlaClient:
    def __init__(self):
        # carla 연결
        self.host = '127.0.0.1'
        self.port = 2000
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        # 시뮬레이터 초기화
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.settings = self.world.get_settings()
        self.traffic_manager.set_synchronous_mode(True)
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05
        # self.settings.no_rendering_mode = True
        self.world.apply_settings(self.settings)

        # 차량 블루프린트
        self.bp = self.world.get_blueprint_library().filter('vehicle')[0]
        if self.bp.has_attribute('color'):
            color = random.choice(self.bp.get_attribute('color').recommended_values)
            self.bp.set_attribute('color', color)

        # 차량 생성
        self.vehicle = self.world.spawn_actor(self.bp, self.world.get_map().get_spawn_points()[0])
        self.vehicle.set_autopilot(True)  # 오토 파일럿 모드

        # 센서 생성
        self.camera = None
        self.lidar_sensors = []
        self.spawn_sensors()

        print('CarlaClient initialized')

    def spawn_sensors(self):
        # 센서 옵션
        self.camera = Camera(self.world, carla.Transform(carla.Location(x=-30, z=30), carla.Rotation(pitch=-45, yaw=0)), self.vehicle, {})
        lidar_sensor_options = {'channels': '64',
                                         'range': '100',
                                         'points_per_second': '200000',
                                         'rotation_frequency': '20',
                                         'upper_fov': '10', 'lower_fov': '-30'}
        locations = [carla.Location(x=-1.8, y=0, z=2.0), carla.Location(x=1.8, y=0, z=2.0), carla.Location(x=0, y=-1.5, z=2.0), carla.Location(x=0, y=1.5, z=2.0)]
        colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0), (0.5, 0, 0.5)]
        for i in range(2):
            lidar = Lidar(self.world, carla.Transform(locations[i], carla.Rotation(pitch=0, yaw=90)), self.vehicle, lidar_sensor_options, (0.5, 0, 1))
            self.lidar_sensors.append(lidar)

        print('Sensors spawned')

    def tick(self):
        self.world.tick()

    def __del__(self):
        self.client.apply_batch([carla.command.DestroyActor(self.vehicle)])
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = False
        self.settings.fixed_delta_seconds = None
        self.world.apply_settings(self.settings)


# 카메라 정보
camera_position = np.array([0.0, -20.0, 20.0])  # 카메라 위치 (X, Y, Z 좌표)
pitch = -45  # 위 아래 각도
yaw = 0  # 좌 우 각도
roll = 0  # 회전 각도

# Pygame 초기화
pygame.init()
gl_display = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption('Carla Lidar')
gluPerspective(90, (display[0] / display[1]), 0.001, 10000.0)

glRotatef(pitch, 1, 0, 0)
glRotatef(yaw, 0, 1, 0)
glRotatef(roll, 0, 0, 1)
glTranslatef(-camera_position[0], -camera_position[1], -camera_position[2])

carla_client = CarlaClient()

last_time = pygame.time.get_ticks()
frame_count = 0

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        carla_client.tick()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Camera Image
        if carla_client.camera.image_data is not None:
            glDrawPixels(display[0] * camera_ratio, display[1] * camera_ratio, GL_RGB, GL_UNSIGNED_BYTE,
                         carla_client.camera.image_data)

        # Lidar Points
        glBegin(GL_POINTS)
        for lidar in carla_client.lidar_sensors:
            glColor3fv(lidar.color)
            for point in lidar.image_data:
                glVertex3fv(point)
        glEnd()

        pygame.display.flip()

        # Calculate the time elapsed since the last frame
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - last_time
        last_time = current_time
        # Calculate the FPS
        if elapsed_time > 0:
            fps = 1000 / elapsed_time
        else:
            fps = 0
        # Show FPS
        pygame.display.set_caption('Carla Lidar - FPS: ' + str(fps))
finally:
    del carla_client
    pygame.quit()
