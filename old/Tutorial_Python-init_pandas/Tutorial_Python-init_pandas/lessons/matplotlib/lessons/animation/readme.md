# **Animation in Matplotlib**  

Matplotlib provides the `matplotlib.animation` module to create **animated visualizations**. This is useful for dynamic data visualization, real-time updates, and interactive plots.

---

## **1. Key Animation Classes**  

Matplotlib provides two main classes for animations:  

| Class | Description |
|-------|-------------|
| `FuncAnimation` | Repeatedly calls a function to update the plot frame by frame. |
| `ArtistAnimation` | Uses a sequence of pre-drawn frames and updates them in an animation. |

---

## **2. Creating an Animation with `FuncAnimation`**  

### **2.1 Basic Line Animation**
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
xdata, ydata = [], []
line, = ax.plot([], [], 'r-')  # Empty line plot

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return line,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))  # Example function
    line.set_data(xdata, ydata)
    return line,

ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), init_func=init, blit=True)
plt.show()
```

| Function | Description |
|----------|-------------|
| `FuncAnimation(fig, update, frames, init_func, blit)` | Creates an animation by updating frames. |
| `init_func()` | Initializes the plot before animation starts. |
| `update(frame)` | Function that updates the plot for each frame. |
| `blit=True` | Improves performance by redrawing only changed parts. |

---

## **3. Creating an Animation with `ArtistAnimation`**  

`ArtistAnimation` is useful when pre-drawn frames are stored and updated in a sequence.

### **3.1 Example: Multiple Frames Animation**
```python
fig, ax = plt.subplots()
frames = []

for i in range(10):
    frame, = ax.plot(np.random.rand(10), np.random.rand(10), 'ro')
    frames.append([frame])

ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True)
plt.show()
```

| Parameter | Description |
|-----------|-------------|
| `frames` | A list of lists containing `Artist` objects (plot elements). |
| `interval` | Time (ms) between frames. |

---

## **4. Customizing Animation**  

### **4.1 Controlling Speed and Frames**  
| Parameter | Description |
|-----------|-------------|
| `interval=50` | Delay between frames in milliseconds. |
| `frames=range(100)` | Number of frames to render. |

Example:
```python
ani = animation.FuncAnimation(fig, update, frames=range(50), interval=100)
```

---

## **5. Saving Animations**  

Matplotlib supports saving animations in different formats.

### **5.1 Save as GIF**
```python
ani.save("animation.gif", writer="pillow")
```

### **5.2 Save as MP4**
```python
ani.save("animation.mp4", writer="ffmpeg", fps=30)
```

| Format | Writer |
|--------|--------|
| GIF | `pillow` |
| MP4 | `ffmpeg` |

---

## **6. Interactive Animations**  

You can pause and resume animations using `pause()` and `resume()`.  

```python
def onClick(event):
    if ani.event_source.running:
        ani.event_source.stop()
    else:
        ani.event_source.start()

fig.canvas.mpl_connect('button_press_event', onClick)
```

| Function | Description |
|----------|-------------|
| `mpl_connect('button_press_event', onClick)` | Detects mouse click for interaction. |
| `ani.event_source.stop()` | Stops animation. |
| `ani.event_source.start()` | Resumes animation. |

---

## **Conclusion**  
Matplotlib’s `animation` module allows the creation of dynamic visualizations using `FuncAnimation` and `ArtistAnimation`. Animations can be customized, saved in multiple formats, and made interactive for better user experience.