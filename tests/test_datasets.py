import unittest
import os
from PIL import Image
import json
import torch
from plate_segmentation.datasets import ObjectDetectionCocoDataset
from ai_utils.transforms import get_transform


class TestObjectDetectionCocoDataset(unittest.TestCase):
    def setUp(self):
        # Generate some example data for testing
        self.test_dir = "tests/"
        self.labels_dict = {
            "images": [
                {
                    "width": 484,
                    "height": 484,
                    "id": 0,
                    "file_name": "images\/00ddb545-buspas_3_1240.jpg",
                },
                {"width": 484, "height": 484, "id": 1, "file_name": "2.jpg"},
                {"width": 484, "height": 484, "id": 2, "file_name": "losowy-img.jpg"},
            ],
            "categories": [{"id": 0, "name": "license plate"}],
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "segmentation": [],
                    "bbox": [
                        66.66544121771197,
                        345.6442376848328,
                        112.25189010831744,
                        38.945710363848114,
                    ],
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": 4371.729599953038,
                },
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "segmentation": [],
                    "bbox": [
                        71.57876778075824,
                        218.28616278774018,
                        105.68230293108094,
                        31.142344680240598,
                    ],
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": 3291.1947044813237,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 0,
                    "segmentation": [],
                    "bbox": [
                        76.78372649727122,
                        113.2190417910702,
                        94.62097230828385,
                        30.218713187711792,
                    ],
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": 2859.3240237264495,
                },
            ],
            "info": {
                "year": 2023,
                "version": "1.0",
                "description": "",
                "contributor": "Label Studio",
                "url": "",
                "date_created": "2023-07-28 07:44:32.552498",
            },
        }
        self.labels_path = os.path.join(self.test_dir, "test_labels_coco.json")
        with open(self.labels_path, "w") as f:
            json.dump(self.labels_dict, f)

        self.dataset = ObjectDetectionCocoDataset(
            root="tests/",
            images_path="test_images/",
            coco_labels_path="test_labels_coco.json",
        )

    def tearDown(self):
        # Clean up the temporary json
        os.remove(self.labels_path)

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 3)
        print(self.dataset.imgs)

    def test_valid_extracted_fields(self):
        img, target = self.dataset[0]
        self.assertTrue("boxes" in target)
        self.assertTrue("labels" in target)
        self.assertTrue("image_id" in target)
        self.assertTrue("area" in target)
        self.assertTrue("iscrowd" in target)
    
    def test_valid_extract_types(self):
        img, target = self.dataset[0]
        self.assertIsInstance(img, Image.Image)
        self.assertIsInstance(target, dict)
        self.assertIsInstance(target["boxes"], torch.tensor(dtype=torch.float32))

    def test_valid_extract_shapes(self):
        pass

    def test_merge_datasets(self):
        dataset2 = ObjectDetectionCocoDataset(
            root="tests/",
            images_path="test_images/",
            coco_labels_path="test_labels_coco.json",
            )

        dataset2.merge_dataset(self.dataset)
        self.assertEqual(len(dataset2), 6)

    def test_getitem_types(self):

        img, target = self.dataset[0]
        self.assertEqual(
            target,
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.int64),
                "image_id": torch.tensor([0], dtype=torch.int64),
                "area": torch.tensor([1600], dtype=torch.float32),
                "iscrowd": torch.tensor([0], dtype=torch.int64),
            },
        )


if __name__ == "__main__":
    unittest.main()
