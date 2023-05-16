import numpy as np
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import open3d as o3d


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
decimate.set_option(rs.option.filter_magnitude, 2 ** 1)
colorizer = rs.colorizer()


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

        depth_image = np.asanyarray(depth_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())

        h_depth, w_depth = np.asanyarray(depth_frame.get_data()).shape

        # Resize color image to match depth image size
        color_image = cv2.resize(color_image, (w_depth, h_depth))

        points = pc.calculate(depth_frame)
        # Lưu point cloud vào file npy
        points_arr = np.asanyarray(points.get_vertices())
        np.save('point_cloud.npy', points_arr)

        # Convert points array to float32
        point_cloud = np.load('point_cloud.npy', allow_pickle=True)
        points = point_cloud.view(np.float32).reshape(-1, 3)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)  # Extract x, y, z coordinates

        # Convert units from meters to millimeters
        points_arr = np.asarray(pcd.points)
        points_arr *= 1000

        # Update the point cloud with the converted coordinates
        pcd.points = o3d.utility.Vector3dVector(points_arr)

        # Lấy giới hạn tọa độ của vùng pcd
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()

        # In giới hạn tọa độ
        print("Min bound:", min_bound)
        print("Max bound:", max_bound)

        if show_point:
            # Visualize point cloud
            o3d.visualization.draw_geometries([pcd])

        print(pcd)

        boxes = object_detection(color_image, show=False)

        intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        extrinsics = profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
        rotation = np.reshape(extrinsics.rotation, (3, 3))
        translation = np.array(extrinsics.translation)

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rotation
        extrinsic_matrix[:3, 3] = translation

        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

        for box in boxes:
            # get coordinates of bounding box
            box_vals = box[0].squeeze().tolist()
            if len(box_vals) == 4:
                xmin, ymin, xmax, ymax = box_vals

                depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                depth_sensor = device.first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()

                depth_value_max = depth_image[int(ymax - 0.5), int(xmax - 0.5)]
                point_max = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [int(xmax - 0.5), int(ymax - 0.5)], depth_value_max)
                point_max = np.multiply(point_max, depth_scale * 1000)  # Convert to millimeters

                depth_value_min = depth_image[int(ymin + 0.5), int(xmin + 0.5)]
                point_min = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [int(xmin + 0.5), int(ymin + 0.5)], depth_value_min)
                point_min = np.multiply(point_min, depth_scale * 1000)  # Convert to millimeters

                # Create bounding box in point cloud coordinates
                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=np.array([point_min[0], point_min[1], 0.9*point_min[2]]),
                    max_bound=np.array([point_max[0], point_max[1], 0.95*point_max[2]])
                )

                # Crop point cloud
                cropped_pc = pcd.crop(bbox)

                # # Visualize cropped point cloud
                # o3d.visualization.draw_geometries([cropped_pc])

                # Average of points in cropped_pc
                point_cloud_array = np.asarray(cropped_pc.points)
                mean_point = np.mean(point_cloud_array, axis=0)

                green_color = [0, 255, 0]  # RGB color code for green

                # Tạo điểm đám mây từ mean_point
                mean_point_cloud = o3d.geometry.PointCloud()
                mean_point_cloud.points = o3d.utility.Vector3dVector([mean_point])

                # Gán màu xanh lá cây cho mean_point
                mean_point_cloud.paint_uniform_color(green_color)

                # Thêm điểm trung bình với màu xanh lá cây vào trong cropped_pc
                mean_point_cloud = o3d.geometry.PointCloud()
                mean_point_cloud.points = o3d.utility.Vector3dVector([mean_point])
                mean_point_cloud.colors = o3d.utility.Vector3dVector([green_color])
                cropped_pc += mean_point_cloud

                # Hiển thị point cloud đã được cắt và có điểm trung bình được thêm vào với màu xanh lá cây
                o3d.visualization.draw_geometries([cropped_pc])

    # Release resources
    pipeline.stop()


if __name__ == '__main__':
    main(show_point=True)