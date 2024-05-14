import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
x1 = [58.99, 55.64, 54.81, 53.13, 52.16]
y1 = [97.07, 96.87, 96.23, 95.81, 95.39]
x2 = [58.72, 56.01, 54.78, 54.01, 51.98]
# y2 = [94.97, 92.88, 92.46, 91.05, 90.79]
y2 = [95.97, 95.88, 94.46, 93.79, 93.05]

fig = plt.figure(figsize=(6, 4), dpi=200, facecolor = "white")
ax = plt.subplot(facecolor = "#EFE9E6")


# --- Remove spines and add gridlines

# ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.spines["bottom"].set_visible(False)

ax.grid(ls="--", lw=0.5, color="#4E616C")

# --- The data

ax.plot(x1, y1, marker="o",color='#fdcf41')
ax.plot(x2, y2, marker="x",color='#153aab')
ax.legend(labels=['MOVRL', 'AF'],loc='best')
plt.xlabel('Success Rate (%)')
plt.ylabel('Accuracy (%)')