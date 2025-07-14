# Flask Cheatsheet

## 1. Importing Flask
- from flask import Flask, render_template, request, redirect, url_for  # Import Flask modules

## 2. Creating a Flask App
- app = Flask(__name__)  # Create a Flask application instance

## 3. Running the App
- if __name__ == '__main__':
  - app.run(debug=True)  # Run the application

## 4. Defining Routes
- @app.route('/')  # Define a route
  - def home():
    - return 'Hello, World!'  # Return response

## 5. Rendering HTML Templates
- @app.route('/template')
  - def template():
    - return render_template('template.html')  # Render HTML template

## 6. Handling GET and POST Requests
- @app.route('/submit', methods=['GET', 'POST'])
  - def submit():
    - if request.method == 'POST':
      - data = request.form['field']  # Access form data
    - return render_template('submit.html')  # Render template

## 7. Redirecting
- return redirect(url_for('home'))  # Redirect to another route

## 8. URL Parameters
- @app.route('/user/<username>')
  - def user(username):
    - return f'Hello, {username}!'  # Access URL parameter

## 9. Static Files
- app = Flask(__name__, static_url_path='/static')  # Serve static files
- url_for('static', filename='style.css')  # Generate URL for static file

## 10. Sending JSON Responses
- from flask import jsonify  # Import jsonify
- @app.route('/api/data')
  - def api_data():
    - return jsonify({'key': 'value'})  # Return JSON response

## 11. Using Flask Extensions
- from flask_sqlalchemy import SQLAlchemy  # Import SQLAlchemy
- db = SQLAlchemy(app)  # Initialize SQLAlchemy with Flask app

## 12. Handling Errors
- @app.errorhandler(404)
  - def not_found(error):
    - return 'Page not found', 404  # Custom error response

## 13. Session Management
- from flask import session  # Import session
- session['key'] = 'value'  # Set session variable
- value = session.get('key')  # Get session variable

## 14. Configuring Flask App
- app.config['DEBUG'] = True  # Set configuration
- app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'  # Database URI

## 15. Creating Forms
- from flask_wtf import FlaskForm  # Import FlaskForm
- class MyForm(FlaskForm):
  - field = StringField('Field', validators=[DataRequired()])  # Define form field
