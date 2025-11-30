import numpy as np
import matplotlib.pyplot as plt

# Load CSV data produced by ./bandwidth > bandwidth.csv
data = np.loadtxt('bandwidth.csv', delimiter=',', skiprows=1)

size_mb = data[:, 0]
h2d_pageable = data[:, 1]
d2h_pageable = data[:, 2]
h2d_pinned = data[:, 3]
d2h_pinned = data[:, 4]

plt.figure(figsize=(8, 6))

# H2D curves
plt.plot(size_mb, h2d_pageable, 'o-', label='H2D - Pageable')
plt.plot(size_mb, h2d_pinned, 'o-', label='H2D - Pinned')

# D2H curves
plt.plot(size_mb, d2h_pageable, 's--', label='D2H - Pageable')
plt.plot(size_mb, d2h_pinned, 's--', label='D2H - Pinned')

plt.xscale('log', base=2)
plt.xlabel('Array size (MB)')
plt.ylabel('Bandwidth (GB/s)')
plt.title('CPU-GPU Transfer Bandwidth vs Array Size')
plt.grid(True, which='both', linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('bandwidth_plot.png', dpi=300)
plt.show()
