# Copyright 2021-2023 Visual Meaning Ltd
# This is free software licensed as GPL-3.0-or-later - see COPYING for terms.

"""Unittests for the labels.py"""

import unittest

import labels


class ScalerTestCase(unittest.TestCase):

    def make_scaler(self, artboard=dict(left=10, right=266, top=10, bottom=266)):
        return labels.Scaler.from_artboard(artboard)

    def test_path(self):
        area = {
            "areaType": "pathPoints",
            "points": [
                {
                    "anchor": [
                        40,
                        540
                    ]
                },
                {
                    "anchor": [
                        20,
                        540
                    ]
                },
                {
                    "anchor": [
                        40,
                        1620
                    ]
                },
                {
                    "anchor": [
                        20,
                        1620
                    ]
                }
            ]
        }
        scaler = self.make_scaler()
        result = [[530.0, 30.0], [530.0, 10.0], [1610.0, 30.0], [1610.0, 10.0]]
        self.assertEqual(scaler.path(area), result)

    def test_svg_path_right_CP_on_1st_anchor(self):
        area = {
            "areaType": "pathPoints",
            "points": [
                {
                    "anchor": [
                        797.549775054185,
                        -665.80063709559
                    ],
                    "rightDirection": [
                        1091.4785945457,
                        -516.718490518526
                    ]
                },
                {
                    "anchor": [
                        1032.05752553983,
                        -665.80063709559
                    ]
                }
            ]
        }
        scaler = self.make_scaler()
        result = 'M6.8e+02,7.9e+02,c-1.5e+02,2.9e+02,0,2.3e+02,0,2.3e+02'
        self.assertEqual(scaler.svg_path(area), result)

    def test_svg_path_left_CP_on_2nd_anchor(self):
        area = {
            "areaType": "pathPoints",
            "points": [
                {
                    "anchor": [
                        752.596448481565,
                        -917.31293237433
                    ]
                },
                {
                    "anchor": [
                        1060.92523895681,
                        -917.31293237433
                    ],
                    "leftDirection": [
                        713.29706960577,
                        -775.899868469694
                    ]
                }
            ]
        }
        scaler = self.make_scaler()
        result = 'M9.3e+02,7.4e+02,c0,0,-1.4e+02,-39,0,3.1e+02'
        self.assertEqual(scaler.svg_path(area), result)

    def test_svg_path_left_right_CPs_on_middle_anchor(self):
        area = {
            "areaType": "pathPoints",
            "points": [
                {
                    "anchor": [
                        750.148859132802,
                        -1168.3982764433
                    ]
                },
                {
                    "anchor": [
                        1094.90110562573,
                        -1153.15124834288
                    ],
                    "leftDirection": [
                        950.901395788392,
                        -1048.96322298998
                    ],
                    "rightDirection": [
                        1238.90081546307,
                        -1257.33927369578
                    ]
                },
                {
                    "anchor": [
                        1382.90052530042,
                        -1129.43364907555
                    ]
                }
            ]
        }
        scaler = self.make_scaler()
        result = 'M1.2e+03,7.4e+02,c0,0,-1.2e+02,2e+02,-15,3.4e+02,1e+02,1.4e+02,-24,2.9e+02,-24,2.9e+02'
        self.assertEqual(scaler.svg_path(area), result)

    def test_svg_path_right_CP_on_first_anchor_and_left_CP_on_3rd_anchor(self):
        area = {
            "areaType": "pathPoints",
            "points": [
                {
                    "anchor": [
                        767.521675161353,
                        -1519.80332874631
                    ],
                    "rightDirection": [
                        854.999201826087,
                        -1284.32141483837
                    ]
                },
                {
                    "anchor": [
                        1082.62689977073,
                        -1436.79173206662
                    ]
                },
                {
                    "anchor": [
                        1439.23798375511,
                        -1412.227034801
                    ],
                    "leftDirection": [
                        1309.63825088957,
                        -1589.26197684686
                    ]
                }
            ]
        }
        scaler = self.make_scaler()
        result = 'M1.5e+03,7.6e+02,c-2.4e+02,87,-83,3.2e+02,-83,3.2e+02,0,0,1.5e+02,2.3e+02,-25,3.6e+02'
        self.assertEqual(scaler.svg_path(area), result)

    def test_svg_path_CPs_on_all_anchors(self):
        area = {
            "areaType": "pathPoints",
            "points": [
                {
                    "anchor": [
                        881.386056161845,
                        -1889.72327431123
                    ],
                    "rightDirection": [
                        739.080460557878,
                        -1737.26686041619
                    ]
                },
                {
                    "anchor": [
                        1174.18541965766,
                        -1875.59177212833
                    ],
                    "leftDirection": [
                        1040.40947971625,
                        -1758.56601529239
                    ],
                    "rightDirection": [
                        1318.99657688422,
                        -2002.27097134708
                    ]
                },
                {
                    "anchor": [
                        1462.18493137641,
                        -1851.87424283145
                    ],
                    "leftDirection": [
                        1638.09041373846,
                        -1967.16007335906
                    ]
                }
            ]
        }
        scaler = self.make_scaler()
        result = 'M1.9e+03,8.7e+02,c-1.5e+02,-1.4e+02,-1.3e+02,1.6e+02,-14,2.9e+02,1.3e+02,1.4e+02,92,4.6e+02,-24,2.9e+02'
        self.assertEqual(scaler.svg_path(area), result)
