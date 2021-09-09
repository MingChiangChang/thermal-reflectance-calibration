import sys
sys.path.insert(0, '..')
import unittest

import preprocess as pp

class TEST_preprocess(unittest.TestCase):

    png_ls = ['/Users/mingchiang/Desktop/Run-0002_LED-On_Power-On_Frame-0030.png',
              '/Users/mingchiang/Desktop/Run-0010_LED-On_Power-Off_Frame-0020.png',
              '/Users/mingchiang/Desktop/Run-0100_LED-Off_Power-On_Frame-0010.png',
              '/Users/mingchiang/Desktop/Run-0002_LED-Off_Power-Off_Frame-0030.png']

    def test_parse_names(self):
        l = pp.parse_names(self.png_ls) 
        self.assertEqual(l, [[2, True, True, 30],
                             [10, True, False, 20],
                             [100, False, True, 10],
                             [2, False, False, 30]])
    
    def test_parse_name(self):
        self.assertEqual(pp.parse_name('Run-0002_LED-On_Power-On_Frame-0030.png'),
                         [2, True, True, 30]) 

    def test_average_images(self):
        imgs = [[1, 2, 3]]


if __name__ == '__main__':
    unittest.main()
