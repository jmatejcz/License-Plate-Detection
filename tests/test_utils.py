import unittest
import numpy as np
import torch
import torchvision.transforms as T
from utils.image_utils import draw_bboxes, get_box_img, enlarge_box

class TestDrawBboxes(unittest.TestCase):

    def test_draw_boxes_numpy_image(self):
        image = np.random.randint(0, 256, size=(3, 400, 600))
        boxes = np.array([[100, 200, 150, 250]]) 
        result = draw_bboxes(image, boxes)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(image.shape, torch.Size([3, 400, 600]))

    def test_draw_boxes_tensor_image(self):
        image = torch.randint(0, 256, size=(3, 400, 600))
        boxes = torch.Tensor([[100, 200, 150, 250]]) 

        result = draw_bboxes(image, boxes)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(image.shape, torch.Size([3, 400, 600]))

    def test_draw_boxes_wrong_format_channels_image(self):
        image = torch.randint(0, 256, size=(1, 400, 600))
        boxes = torch.Tensor([[100, 200, 150, 250]]) 

        with self.assertRaises(ValueError):
            draw_bboxes(image, boxes)

    def test_draw_boxes_wrong_format_shape_image(self):
        image = torch.randint(0, 256, size=(1, 3, 400, 600))
        boxes = torch.Tensor([[100, 200, 150, 250]]) 
        with self.assertRaises(ValueError):
            draw_bboxes(image, boxes)
    
    def test_draw_boxes_wrong_format_shape_boxes(self):
        image = torch.randint(0, 256, size=(3, 400, 600))
        boxes = torch.Tensor([100, 200, 150, 250]) 
        with self.assertRaises(ValueError):
            draw_bboxes(image, boxes)

        boxes = torch.Tensor([[[100, 200, 150, 250]]])
        with self.assertRaises(ValueError):
            draw_bboxes(image, boxes)

    def test_draw_boxes_no_boxes(self):
        image = torch.randint(0, 256, size=(3, 400, 600), dtype=torch.uint8)
        boxes = torch.empty((0, 4)) 
        
        result = draw_bboxes(image, boxes)
        self.assertTrue(torch.equal(result, image))
        
    def test_draw_boxes_multiple_boxes(self):
        image = torch.randint(0, 256, size=(3, 400, 600))
        boxes = torch.Tensor([[100, 200, 150, 250], [170, 120, 230, 200]]) 

        result = draw_bboxes(image, boxes)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(image.shape, torch.Size([3, 400, 600]))


class TestGetBoxImg(unittest.TestCase):

    def test_get_box_img_tensor_type(self):
        image = torch.randint(0, 256, size=(3, 400, 600))
        box = torch.Tensor([100, 200, 150, 250]) 
        result = get_box_img(image, box)
        self.assertEqual(result.shape, torch.Size([3, 50, 50]))

    def test_get_box_img_numpy_type(self):
        image = np.random.randint(0, 256, size=(3, 400, 600))
        box = np.array([100, 200, 150, 250]) 
        result = get_box_img(image, box)
        self.assertEqual(result.shape, (3, 50, 50))

    def test_get_box_img_box_bigger_than_img(self):
        image = torch.randint(0, 256, size=(3, 400, 600))
        box = torch.Tensor([100, 150, 600, 750]) 
        result = get_box_img(image, box)
        self.assertEqual(result.shape, torch.Size([3, 250, 500]))
    
    def test_get_box_img_no_box(self):
        image = torch.randint(0, 256, size=(3, 400, 600))
        box = torch.zeros(4)
        result = get_box_img(image, box)
        self.assertEqual(result.shape, torch.Size([3, 0, 0]))

    def test_get_box_img_box_wrong_format(self):
        image = torch.randint(0, 256, size=(3, 400, 600))
        box = torch.Tensor([[100, 200, 150, 250], [170, 120, 230, 200]]) 
        with self.assertRaises(ValueError):
            get_box_img(image, box)


class TestEnlargeBox(unittest.TestCase):

    def test_enlarge_box_within_limits(self):
        img_shape = [3, 100, 100]
        box = [30, 30, 50, 50]
        magnify = 2.0 
        result = enlarge_box(img_shape, box, magnify)
        expected_box = [20, 20, 60, 60]

        self.assertListEqual(result, expected_box)

    def test_enlarge_box_outside_bounds(self):
        img_shape = [3, 100, 100]
        box = [30, 30, 90, 90]
        magnify = 2.0
        result = enlarge_box(img_shape, box, magnify)
        expected_box = [0, 0, 100, 100]

        self.assertListEqual(result, expected_box)

if __name__ == '__main__':
    unittest.main()