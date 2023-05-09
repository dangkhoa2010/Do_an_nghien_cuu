import numpy as np
import math
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import open3d as o3d


class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width * 2, depth_intrinsics.height * 2

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h) / w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
               (w * view_aspect, h) + (w / 2.0, h / 2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def pointcloud(out, verts, texcoords, color, painter=False):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5 ** state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch - 1, out=u)
    np.clip(v, 0, cw - 1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


out = np.empty((h, w, 3), dtype=np.uint8)


def object_detection(image, show=False):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run YOLOv8 inference on the frame
    results = model.predict(source=image, device='cpu')

    box_list = []
    for result in results:
        # detection
        if len(result):
            boxes = result.boxes
            for box in boxes:
                b = box.xyxy[0] # get box coordinates in (top, left, bottom, right) format
                box_list.append(box.xyxy)
        else:
            print("No results found.")

    if show:
        model = YOLO('yolov8n.pt')
        cap = cv2.VideoCapture(1)
        cap.set(3, 640)
        cap.set(4, 480)

        while True:
            _, frame = cap.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = model.predict(img)

            for r in results:

                annotator = Annotator(frame)

                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    annotator.box_label(b, model.names[int(c)])

            frame = annotator.result()
            cv2.imshow('YOLO V8 Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        cap.release()
        cv2.destroyAllWindows()

    return box_list


def main(show_point=False):
    while True:
        # Get color image from Realsense camera
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        # Grab new intrinsics
        intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        extrinsics = profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
        rotation = np.reshape(extrinsics.rotation, (3, 3))
        translation = np.array(extrinsics.translation)

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rotation
        extrinsic_matrix[:3, 3] = translation

        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

        w, h = depth_intrinsics.width * 2, depth_intrinsics.height * 2

        color_image = np.asanyarray(color_frame.get_data())
        mapped_frame, color_source = color_frame, color_image

        # boxes = object_detection(color_image, show=False)
        #
        # for box in boxes:
        #     # get coordinates of bounding box
        #     box_vals = box[0].squeeze().tolist()
        #     if len(box_vals) == 4:
        #         xmin, ymin, xmax, ymax = box_vals
        #
        #         # box_depth = np.zeros_like(np.asanyarray(depth_frame.get_data()))
        #         # box_depth[int(ymin):int(ymax), int(xmin):int(xmax)] = np.asanyarray(depth_frame.get_data())[int(ymin):int(ymax), int(xmin):int(xmax)]
        #
        #         pcd = o3d.geometry.PointCloud.create_from_depth_image(
        #             depth=o3d.geometry.Image(np.asanyarray(depth_frame.get_data())),
        #             intrinsic=o3d.camera.PinholeCameraIntrinsic(
        #                 width=depth_intrinsics.width,
        #                 height=depth_intrinsics.height,
        #                 fx=depth_intrinsics.fx,
        #                 fy=depth_intrinsics.fy,
        #                 cx=depth_intrinsics.ppx,
        #                 cy=depth_intrinsics.ppy
        #             )
        #         )
        #
        #         # Hiển thị Point Cloud
        #         o3d.visualization.draw_geometries([pcd])

        points = pc.calculate(depth_frame)
        # Lưu point cloud vào file npy
        points_arr = np.asanyarray(points.get_vertices())
        np.save('point_cloud.npy', points_arr)

        # # Load point cloud from .npy file
        # point_cloud = np.load('point_cloud.npy')
        # print(type(point_cloud))

        # Convert points array to float32
        point_cloud = np.load('point_cloud.npy', allow_pickle=True)
        points = point_cloud.view(np.float32).reshape(-1, 3)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)  # Extract x, y, z coordinates


        # Visualize point cloud
        o3d.visualization.draw_geometries([pcd])

        # pc.map_to(mapped_frame)
        #
        # # Point cloud data to arrays
        # v, t = points.get_vertices(), points.get_texture_coordinates()
        # verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        # texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
        #
        # out.fill(0)
        #
        # tmp = np.zeros((h, w, 3), dtype=np.uint8)
        # pointcloud(tmp, verts, texcoords, color_source)
        # tmp = cv2.resize(tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        # np.putmask(out, tmp > 0, tmp)
        #
        # if show_point:
        #     cv2.namedWindow(state.WIN_NAME)
        #     cv2.setWindowTitle(state.WIN_NAME, "RealSense")
        #     cv2.imshow(state.WIN_NAME, out)
        #
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # boxes = object_detection(color_image, show=False)
        #
        # for box in boxes:
        #     # get coordinates of bounding box
        #     box_vals = box[0].squeeze().tolist()
        #     if len(box_vals) == 4:
        #         xmin, ymin, xmax, ymax = box_vals

            # # Get all texcoords points within bounding box
            # in_box = np.where((texcoords[:, 0] >= xmin) & (texcoords[:, 0] <= xmax) & (texcoords[:, 1] >= ymin) & (texcoords[:, 1] <= ymax))[0]
            # print(in_box)
            # box_texcoords = texcoords[in_box]
            #
            # # Mapping texcoords coordinates to 3D . coordinates
            # print(box_texcoords[:, 0], box_texcoords[:, 1])
            # box_depth_values = depth_frame.as_depth_frame().get_distance(box_texcoords[:, 0], box_texcoords[:, 1])
            # box_points = rs.rs2_deproject_pixel_to_point(depth_intrinsics, box_texcoords, box_depth_values)
            #
            # # Calculate the average of the points inside the bounding box
            # avg_point = np.mean(box_points, axis=0)
            #
            # print("Average point: ", avg_point)

    # Release resources
    pipeline.stop()


if __name__ == '__main__':
    main(show_point=False)

# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

# import pyrealsense2 as rs
# import numpy as np
# import open3d as o3d
#
# # Khởi tạo Pipeline và Start Streaming
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# pipeline.start(config)
#
# # Lấy dữ liệu Depth và Intrinsics từ Camera
# frames = pipeline.wait_for_frames()
# depth_frame = frames.get_depth_frame()
# depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
#
# # Convert Depth Frame thành ảnh Depth và tạo Point Cloud từ ảnh Depth
# depth_image = np.asanyarray(depth_frame.get_data())
# pcd = o3d.geometry.PointCloud.create_from_depth_image(
#     depth=o3d.geometry.Image(depth_image),
#     intrinsic=o3d.camera.PinholeCameraIntrinsic(
#         width=depth_intrinsics.width,
#         height=depth_intrinsics.height,
#         fx=depth_intrinsics.fx,
#         fy=depth_intrinsics.fy,
#         cx=depth_intrinsics.ppx,
#         cy=depth_intrinsics.ppy
#     )
# )
#
# # Hiển thị Point Cloud
# o3d.visualization.draw_geometries([pcd])
#
# # Stop Pipeline
# pipeline.stop()