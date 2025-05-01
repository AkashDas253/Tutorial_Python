## Cross-Platform Notes in PySimpleGUI

PySimpleGUI is designed to be **cross-platform**, allowing GUI applications to run on Windows, macOS, Linux, and even Raspberry Pi with minimal or no modification. However, platform differences can affect appearance, behavior, and dependencies.

---

### Core Cross-Platform Capabilities

- Runs on all major platforms: **Windows, macOS, Linux**
- Same Python script works across platforms
- Built on top of cross-platform GUI frameworks (default: Tkinter)

---

### Backend Compatibility

| PySimpleGUI Version | Underlying GUI Framework | Cross-Platform? | Notes                                      |
|----------------------|--------------------------|------------------|---------------------------------------------|
| PySimpleGUI (default) | Tkinter                  | Yes              | Most stable and widely supported            |
| PySimpleGUIQt         | PyQt5                    | Yes              | Supports advanced features, larger installer|
| PySimpleGUIWx         | wxPython                 | Yes              | Less popular, fewer users                   |
| PySimpleGUIWeb        | Remi (Web GUI)           | Runs in browser  | Good for web-based GUIs                     |

---

### Platform-Specific Considerations

#### Appearance

- **Fonts, button sizes, spacing, and colors** may differ across OSes.
- Use `sg.set_options()` to normalize fonts and element sizes.
- Avoid hardcoding sizes; let elements auto-size when possible.

#### File Paths

- Use `os.path` or `pathlib.Path` to handle file paths.
- Avoid hardcoded separators (`/` or `\`).
```python
from pathlib import Path
config_path = Path.home() / 'app_config' / 'settings.ini'
```

#### File Browsing Elements

- `sg.FileBrowse`, `sg.FolderBrowse`, and related elements work across platforms but:
  - **macOS** may restrict access to certain directories.
  - **Linux** may require `tk` or desktop environment support.

#### Right-Click Menus

- Appear slightly different on each platform.
- Some older Linux distros may not show right-click menu unless widget is in focus.

#### Keyboard Shortcuts

- Use OS-aware shortcuts:
  - `Ctrl` on Windows/Linux
  - `Cmd` on macOS (may require handling manually)

---

### Deployment Notes

#### Windows

- Use **PyInstaller** or **cx_Freeze** to bundle as `.exe`
- Tkinter usually preinstalled with Python

#### macOS

- Use PyInstaller with `--windowed` to suppress terminal
- Gatekeeper may block unsigned apps; may need to notarize or allow manually

#### Linux

- Tkinter must be installed separately in some distros (`sudo apt install python3-tk`)
- Use `chmod +x` and desktop shortcut files for launching apps

---

### Tips for Writing Cross-Platform Code

- Use relative paths and configuration files
- Use layout functions for consistency
- Test on all target platforms if possible
- Avoid platform-specific APIs unless wrapped with checks:
```python
import platform
if platform.system() == "Windows":
    # Windows-specific code
```

---

### Summary

| Feature            | Cross-Platform Support |
|--------------------|------------------------|
| Layout and Elements| Yes                    |
| File Browsing      | Yes (minor differences)|
| Fonts and Sizes    | May differ             |
| Keyboard Shortcuts | OS-dependent           |
| Deployment         | Requires platform-specific bundling |

---
