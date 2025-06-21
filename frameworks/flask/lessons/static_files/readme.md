## Static Files in Flask  

### Overview  
Flask serves **static files** such as CSS, JavaScript, and images from the **`static/`** directory. These files enhance web applications by adding styling and interactivity.

---

### Directory Structure  
```
/my_flask_app
    /static
        /css
            styles.css
        /js
            script.js
        /images
            logo.png
    /templates
        index.html
    app.py
```
- **`static/css/styles.css`** → CSS files  
- **`static/js/script.js`** → JavaScript files  
- **`static/images/logo.png`** → Images  

---

### Serving Static Files  

#### Linking a CSS File  
**templates/index.html**  
```html
<!DOCTYPE html>
<html>
<head>
    <title>Flask Static Files</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
</head>
<body>
    <h1>Welcome to Flask!</h1>
</body>
</html>
```
- **`url_for('static', filename='css/styles.css')`** generates the correct static file path.

#### Example CSS File  
**static/css/styles.css**  
```css
body {
    background-color: lightblue;
    text-align: center;
}
```

---

### Adding JavaScript  
**templates/index.html**  
```html
<script src="{{ url_for('static', filename='js/script.js') }}"></script>
```

**static/js/script.js**  
```js
document.addEventListener("DOMContentLoaded", function() {
    alert("JavaScript Loaded!");
});
```

---

### Serving Images  
**templates/index.html**  
```html
<img src="{{ url_for('static', filename='images/logo.png') }}" alt="Logo">
```

---

### Custom Static Routes  
You can serve static files differently if needed.

#### Custom Route for Serving Files  
```python
from flask import send_from_directory

@app.route('/files/<path:filename>')
def custom_static(filename):
    return send_from_directory('static', filename)
```
- Access files via `/files/css/styles.css`.

---

### Summary  

| Feature | Description |
|---------|------------|
| **Static Folder** | Store static files in `/static` |
| **CSS & JS** | Use `url_for('static', filename='path')` to link files |
| **Images** | `<img>` tag with `url_for()` for dynamic paths |
| **Custom Routes** | `send_from_directory()` for custom static serving |
