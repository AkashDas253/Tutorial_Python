# **Media Elements** in Streamlit

Streamlit supports embedding various **media types** — including images, audio, and video — into your app. These elements enhance interactivity and presentation, especially for dashboards, reports, and demos.

---

## 🔹 Supported Media Functions

| Function           | Media Type        | Purpose                               |
|--------------------|-------------------|----------------------------------------|
| `st.image()`       | Image             | Display image files or arrays          |
| `st.audio()`       | Audio             | Play audio from file or bytes          |
| `st.video()`       | Video             | Embed video content                    |

---

## 🔸 `st.image()`

- **Purpose**: Display static or animated images (e.g., PNG, JPEG, GIF).

### ✅ Syntax:
```python
st.image(image, caption=None, width=None, use_column_width=False, clamp=False, channels="RGB", output_format="auto")
```

| Parameter            | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `image`              | File path, URL, PIL.Image, numpy array, or bytes             |
| `caption`            | Optional image caption                                       |
| `width`              | Width in pixels                                              |
| `use_column_width`   | Scale to column width (`False` by default)                   |
| `clamp`              | Clamp pixel values between 0–255 if using array (`False`)    |
| `channels`           | `"RGB"` or `"BGR"` for arrays                                |
| `output_format`      | `"JPEG"` or `"PNG"` (default: `"auto"`)                      |

### 🔹 Example:
```python
from PIL import Image
img = Image.open("photo.jpg")
st.image(img, caption="Sample Photo", use_column_width=True)
```

---

## 🔸 `st.audio()`

- **Purpose**: Embed an audio player to play `.mp3`, `.wav`, `.ogg`, etc.

### ✅ Syntax:
```python
st.audio(data, format="audio/wav", start_time=0)
```

| Parameter     | Description                                          |
|---------------|------------------------------------------------------|
| `data`        | File path, URL, or bytes-like object                 |
| `format`      | MIME type of the audio (e.g., `"audio/mp3"`)         |
| `start_time`  | Time (in seconds) where audio starts                 |

### 🔹 Example:
```python
audio_file = open("sound.mp3", "rb")
st.audio(audio_file.read(), format="audio/mp3")
```

---

## 🔸 `st.video()`

- **Purpose**: Embed videos from local files, byte streams, or URLs (YouTube, Vimeo).

### ✅ Syntax:
```python
st.video(data, format="video/mp4", start_time=0)
```

| Parameter     | Description                                          |
|---------------|------------------------------------------------------|
| `data`        | File path, URL, or bytes                             |
| `format`      | MIME type of video (e.g., `"video/mp4"`)             |
| `start_time`  | Start time in seconds (if supported)                 |

### 🔹 Example:
```python
st.video("https://www.youtube.com/watch?v=abc123")
```

---

## 🧩 When to Use Each

| Requirement                     | Use Function    |
|----------------------------------|------------------|
| Display static image or array    | `st.image()`     |
| Add narration or background music| `st.audio()`     |
| Embed tutorials or walkthroughs  | `st.video()`     |

---

## 📝 Combined Example

```python
import streamlit as st
from PIL import Image

# Image
st.image("logo.png", caption="Company Logo", use_column_width=True)

# Audio
with open("audio_sample.wav", "rb") as f:
    st.audio(f.read(), format="audio/wav")

# Video
st.video("https://www.youtube.com/watch?v=jNQXAC9IVRw")
```

---
