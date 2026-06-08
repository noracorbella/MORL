import sys
sys.path.insert(0, 'utils')
from CH_operations import get_hull
import numpy as np

points = np.array([[-1.,        25.,  -1.       ],
                   [-1.7,       17.5,  -0.7      ],
                   [-2.19,      12.25, -0.49     ],
                   [-2.533,      8.575, -0.343   ],
                   [-2.7731,     6.0025, -0.2401  ],
                   [-2.94117,    4.20175, -0.16807],
                   [-3.058819,   2.941225, -0.117649]])

result = get_hull(points)
print('Hull vertices:')
for v in result:
    print(v)
print(f'Count: {len(result)} (expected 2 extreme points)')