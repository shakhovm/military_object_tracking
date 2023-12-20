import cv2
import numpy as np
import os
import pandas as pd


class BboxAugmentation:
    def __init__(self, images, bounding_boxes, out_of_view,
                 new_image_save_folder, new_box_save_folder, new_occlusion_save_folder):
        self.images = images
        self.bounding_boxes = bounding_boxes
        self.out_of_view = out_of_view
        self.new_image_folder = new_image_save_folder
        self.new_box_save_folder = new_box_save_folder
        self.new_occlusion_save_folder = new_occlusion_save_folder

        self.frame_number = 0

        self.rotation_degree = 0.
        self.frame_sets = [5, 10, 20, 30, 40, 50]
        # frame_sets = [5, 10, 20, 25]
        self.rotation_speeds = [0.1, 0.2, 0.3, 0.4]
        self.rotation_speeds = np.linspace(0.2, 0.6, 5)
        # rotation_speeds = [0.]
        self.current_set = 1
        self.direction_speed = np.array([-4, -1])
        self.current_direction = np.array([-30, -1])
        self.current_size = np.array([750, 750])
        self.max_degree = 10

        self.bboxes = []
        self.new_outs = []

        self.size_speed = -1
        self.last_frame = 0
        self.last_index = -1
        self.first_occlusion = 0

        l = np.random.choice([700])
        self.size = (l, l)

    def process(self):

        image_bboxes = pd.read_csv(self.bounding_boxes, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                                   low_memory=False).values

        path = os.path.join(self.new_image_folder, f'augmented_{self.images.split("/")[-1]}')
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(self.out_of_view) as f:
            occlusion = f.read().split(',')

        images = sorted(os.listdir(self.images))
        counter = 0
        for ind, image_file in enumerate(images):
            is_out = int(occlusion[ind])
            if is_out:
                if len(self.bboxes) == 0:
                    self.frame_number += 1
                    self.first_occlusion += 1
                    continue
            else:
                self.last_frame = self.frame_number
            current_bboxes = list(map(int, image_bboxes[ind]))
            image = cv2.imread(os.path.join(self.images, image_file))
            height, width = image.shape[:2]
            is_image = self.augment_image(image=image, bbox=current_bboxes, out_of_view=is_out,
                                          height=height, width=width)
            if is_image is not None:
                number_len = len(str(counter + 1))
                img_name = "00000000"[:-number_len] + str(counter + 1) + ".jpg"
                cv2.imwrite(os.path.join(path, img_name), is_image)
                counter += 1
                # cv2.imshow("Frame", is_image)
                #
                #
                #
                # # #     cv2.imshow("Frame", img)
                # key = cv2.waitKey(1)
                # if key == ord('q'):
                #     break
        with open(os.path.join(self.new_box_save_folder,
                               f'augmented_{self.bounding_boxes.split("/")[-1]}'), 'w') as f:
            f.write("\n".join(list(map(lambda x: ",".join(list(map(str, x))), self.bboxes))))

        with open(os.path.join(self.new_occlusion_save_folder,
                               f'augmented_{self.out_of_view.split("/")[-1]}'), 'w') as f:
            f.write(",".join(list(map(str, self.new_outs))))

    def fix_borders(self, height, width, bbox, coords):
        #         if coords[1] > width:
        if bbox[1] < coords[1]:
            diff = bbox[1] - coords[1]  # - 20
            coords[1] += diff
            coords[3] += diff
        if bbox[0] < coords[0]:
            diff = bbox[0] - coords[0]  # - 20
            coords[2] += diff
            coords[0] += diff
        if bbox[2] > coords[2]:
            diff = bbox[2] - coords[2]  # - 20
            coords[2] += diff
            coords[0] += diff
        if bbox[3] > coords[3]:
            diff = bbox[3] - coords[3]  # - 20
            coords[1] += diff
            coords[3] += diff
        if coords[0] < 0:
            coords[2] -= coords[0]
            coords[0] = 0
        if coords[1] < 0:
            coords[3] -= coords[1]
            coords[1] = 0
        if coords[3] > width:
            coords[1] -= coords[3] - width
            coords[3] = width
        if coords[2] > height:
            coords[0] -= coords[2] - height
            coords[2] = height

    def augment_image(self, image, bbox, out_of_view, height, width):

        if self.current_size[0] < 600 or self.current_size[0] - 100 > self.size[0]:  # or random.random() < 0.05:
            self.size_speed *= -1

        self.current_size += self.size_speed
        #         current_size = size
        #         current_size = np.array([900, 900])
        #     img = aug(image=img)
        #     img[100:1200, 100:1200] = ip.blur(img[100:1200, 100:1200], (5, 5))
        #     img = aug(image=img)

        if bbox[3] > self.current_size[0] or bbox[2] > self.current_size[0]:
            self.frame_number += 1
            return
        coords = [bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]]
        new_size = [self.current_size[0] - bbox[3], self.current_size[1] - bbox[2]]
        #         new_size = size

        #     print(new_size)
        coords = [coords[0] - new_size[0] // 2, coords[1] - new_size[1] // 2, coords[2] + new_size[0] // 2,
                  coords[3] + new_size[1] // 2]
        #     coords = [max(0, coords[0]), min(width, coords[1]),
        #               ((coords[0] - height) > 0) * (coords[0] - height) + ]1
        if np.random.random() < 0.05:
            self.direction_speed[0] *= -1
        if np.random.random() < 0.05:
            self.direction_speed[1] *= -1
            #         current_direction = 0
        self.current_direction += self.direction_speed
        coords = [coords[0] + self.current_direction[1],
                  coords[1] + self.current_direction[1],
                  coords[2] + self.current_direction[0],
                  coords[3] + self.current_direction[0]
                  ]
        #         print(coords, width, height)
        if width - coords[3] < 0:
            coords[1] += width - coords[3]
            coords[3] = width
        if coords[1] < 0:
            coords[1] = 0
            coords[3] += -coords[1]
        if height - coords[2] < 0:
            coords[0] += height - coords[2]
            coords[2] = height
        if coords[0] < 0:
            coords[0] = 0
            coords[2] += -coords[0]
        #         coords[1] = ((width - coords[3]) < 0) * (width - coords[3]) + coords[1]
        #         coords[0] = ((height - coords[2]) < 0) * (height - coords[2]) + coords[0]
        #         coords[0] = ((width - coords[3]) < 0) * (width - coords[3]) + coords[1]
        #         coords[0] = ((width - coords[3]) < 0) * (width - coords[3]) + coords[1]
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), self.rotation_degree, 1.0)

        if self.frame_number % self.current_set == 0:
            self.current_set = np.random.choice(self.frame_sets)
            self.rotation_direction = np.random.choice([-1, 1])
            self.current_speed = np.random.choice(self.rotation_speeds)
        if self.rotation_degree >= self.max_degree:
            self.rotation_direction = -1
        elif self.rotation_degree <= -self.max_degree:
            self.rotation_direction = 1
        self.rotation_degree += self.rotation_direction * self.current_speed
        coords = np.array(coords)

        coords_1 = np.stack([coords[2], coords[0], np.ones_like(coords[0])])
        coords_2 = np.stack([coords[3], coords[1], np.ones_like(coords[0])])

        coords = [*(M @ coords_1), *(M @ coords_2)]
        x_y = [np.array([bbox[0], bbox[1]]),
               np.array([bbox[0], bbox[1] + bbox[3]]),
               np.array([bbox[0] + bbox[2], bbox[1]]),
               np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]])]
        new_coord = []
        for coord in x_y:
            new_coord.append(M @ np.stack([coord[0], coord[1], np.ones_like(coord[0])]))
        new_coord = np.array(new_coord)
        bbox = [new_coord[:, 0].min(), new_coord[:, 1].min(),
                new_coord[:, 0].max(), new_coord[:, 1].max()]
        #     bbox = [min(new_coord[0][0], new_coord[1][0]),
        #             min(new_coord[0][1], new_coord[1][1]),
        #             max(new_coord[2][0], new_coord[3][0]),
        #             max(new_coord[2][1], new_coord[3][1])]

        bbox = np.array([max(0, bbox[0]), max(0, bbox[1]), min(bbox[2], w), min(bbox[3], h)], dtype=np.int32)
        #         bbox = np.array(bbox, dtype=np.int32)
        coords = list(map(int, coords))

        rotated = cv2.warpAffine(image, M, (w, h))
        new_coord = [new_coord[0], new_coord[2], new_coord[3], new_coord[1]]

        coords = [coords[0], coords[1], coords[0] + self.current_size[0], coords[1] + self.current_size[0]]
        self.fix_borders(width, height, bbox, coords)
        #         cv2.imshow("Frame", rotated)
        #         key = cv2.waitKey(1)
        #         if key == ord('q'):
        #             break
        #         if len(np.where(np.array(coords) < 0)) > 0:
        #             print(coords)
        coords = list(map(lambda x: max(x, 0), coords))
        img = rotated[coords[1]: coords[3], coords[0]:coords[2]]
        if img.shape[0] == 0:
            self.last_index = self.frame_number
            return
        new_h, new_w = img.shape[:2]
        #         print(coords, bbox)
        bbox = [coords[0] - bbox[0], coords[1] - bbox[1], coords[0] - bbox[2], coords[1] - bbox[3]]
        bbox = list(map(lambda x: abs(int(x / (new_h / self.size[0]))), bbox))
        if out_of_view:
            self.bboxes.append([0., 0., 0., 0.])
        else:
            #             bboxes.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3]- bbox[1]])
            self.bboxes.append([bbox[0], bbox[1],
                                min(max(bbox[2] - bbox[0], 0), self.size[0]),
                                min(max(0, bbox[3] - bbox[1]), self.size[0])])
        self.new_outs.append(out_of_view)

        #         print(new_h, new_w)
        img = cv2.resize(img, self.size)
        # if not write and not is_out:
        # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
        return img


if __name__ == '__main__':
    example_dir = "../data/train"
    bbox_file = "../data/bounding_boxes/007.txt"
    out_of_view_file = "../data/out_of_view/007.txt"
    image_sample = "../data/train/007"
    bag = BboxAugmentation(images=image_sample, bounding_boxes=bbox_file, out_of_view=out_of_view_file,
                           new_image_save_folder="../data/train", new_box_save_folder="../data/bounding_boxes",
                           new_occlusion_save_folder="../data/out_of_view")
    bag.process()
