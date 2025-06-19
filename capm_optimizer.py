
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

def generate_efficient_frontier_plot():
    # Simulated efficient frontier data
    x = np.linspace(0.05, 0.25, 100)
    y = 0.1 + 0.4 * (x - 0.05)**0.5

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label='Efficient Frontier')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.grid(True)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf
