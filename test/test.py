import sys
sys.path.insert(0, '../')
import unittest
import yaml

from preprocess import *

class test(unittest.TestCase):

    def test_make_continuous(self):
        self.assertEqual(make_continuous([1,5]), [1,2,3,4,5], 
                "Should be [1,2,3,4,5]")

    def test_get_wanted_frame(self):
        wanted_frames = get_wanted_frames_for_condition('800us_55W',
                                                       'test.yaml')
        ans = { 
                1: [1,2,3,4,5,6,7,8,9,10],
                2: [2,3,4,5]
                }
        self.assertEqual(wanted_frames, ans)





if __name__ == '__main__':
    unittest.main()
