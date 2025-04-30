## Theming and Configurations in Streamlit

Streamlit allows developers to customize the look and behavior of their apps using configuration settings and theming options. These can be set globally or per project.

---

## Configuration File (`.streamlit/config.toml`)

A TOML-formatted file used to customize app behavior, layout, theming, and more.

### File Location

Place the file at the root of your project in a folder named `.streamlit`.

**Path:**
```
.project_root/
└── .streamlit/
    └── config.toml
```

---

## Available Configuration Sections

### `[server]`
Controls the app server behavior.

| Option               | Description                                | Default           |
|----------------------|--------------------------------------------|-------------------|
| `headless`           | Run app without browser launching          | `false`           |
| `port`               | Custom server port                         | `8501`            |
| `enableCORS`         | Allow cross-origin requests                | `true`            |
| `enableXsrfProtection` | Prevent cross-site request forgery     | `true`            |
| `maxUploadSize`      | Max file upload size (MB)                  | `200`             |
| `maxMessageSize`     | Max websocket message size (MB)            | `200`             |

### `[theme]`
Customize app’s look and feel.

| Option           | Description                          | Default        |
|------------------|--------------------------------------|----------------|
| `primaryColor`   | Color for buttons, sliders, etc.     | `#F63366`      |
| `backgroundColor`| Main background color                | `#FFFFFF`      |
| `secondaryBackgroundColor`| Sidebar/background blocks | `#F0F2F6`      |
| `textColor`      | Main text color                      | `#262730`      |
| `font`           | One of `sans serif`, `serif`, `monospace` | `sans serif` |

### `[client]`
Controls frontend behavior.

| Option               | Description                       |
|----------------------|-----------------------------------|
| `toolbarMode`        | Show/hide toolbar (`auto`, `developer`, `viewer`) |
| `showSidebarNavigation` | Display navigation in sidebar  |

---

## Sample `config.toml`

```toml
[server]
headless = true
port = 8502
enableCORS = false

[theme]
primaryColor = "#FF5733"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#000000"
font = "serif"

[client]
toolbarMode = "viewer"
```

---

## Programmatic Configuration (limited)

Some settings like page title or icon can be set inside the script:

```python
st.set_page_config(
    page_title="My Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

| Parameter                | Values                            | Description                                  |
|--------------------------|------------------------------------|----------------------------------------------|
| `page_title`             | String                            | Browser tab title                            |
| `page_icon`              | Emoji or path to `.ico`/`.png`    | App icon                                     |
| `layout`                 | `"centered"` or `"wide"`          | App layout width                             |
| `initial_sidebar_state`  | `"auto"`, `"expanded"`, `"collapsed"` | Sidebar initial state                     |

---

## Environment Variables

Set in shell or `.env` file to override configs temporarily.

```bash
STREAMLIT_SERVER_PORT=8505
STREAMLIT_THEME_PRIMARYCOLOR="#00FF00"
```

---

## Reset to Defaults

Use:
```bash
streamlit config show > ~/.streamlit/config.toml
```

Then edit or remove file to reset.

---

## Tips

- Use the `[theme]` section to enforce consistent branding.
- Use `st.set_page_config()` early in the script.
- Avoid hardcoding visuals; prefer using the config file.
- Configs take effect on app restart.

