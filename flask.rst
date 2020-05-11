Flask
======

Flask is a micro web framework written in Python. 
It is easy and fast to implement with the knowledge of basic web development and REST APIs.
How is it relevant to model building? Sometimes, it might be necessary to 
run models in the cloud, and 


Basics
------
This gives a basic overall of how to run flask, with the debugger on,
and displaying a static ``index.html`` file.
A browser can then be nagivated to ``http://127.0.0.1:5000/`` to view the index page.


.. code:: Python

    from flask import Flask, render_template

    app = Flask(__name__)


    @app.route('/')
    def index():
        return render_template('index.html')

    if __name__ == '__main__':
        app.run(debug = True)


Folder Structure
-----------------

There are some default directory structure to adhere to. 
The first is that HTML files are placed under /templates, 
second is for Javascript, CSS or other static files like images, will be placed under /static

.. code:: bash

    ├── app.py
    ├── static
    │   └── img
    │       └── image.png
    └── templates
        └── index.html


App Configs
-----------

Flask by default comes with a configuration dictionary which can be called as below.

.. code:: Python

    print(app.config)

    {'APPLICATION_ROOT': '/',
    'DEBUG': True,
    'ENV': 'development',
    'EXPLAIN_TEMPLATE_LOADING': False,
    'JSONIFY_MIMETYPE': 'application/json',
    'JSONIFY_PRETTYPRINT_REGULAR': False,
    'JSON_AS_ASCII': True,
    'JSON_SORT_KEYS': True,
    'MAX_CONTENT_LENGTH': None,
    'MAX_COOKIE_SIZE': 4093,
    'PERMANENT_SESSION_LIFETIME': datetime.timedelta(days=31),
    'PREFERRED_URL_SCHEME': 'http',
    'PRESERVE_CONTEXT_ON_EXCEPTION': None,
    'PROPAGATE_EXCEPTIONS': None,
    'SECRET_KEY': None,
    'SEND_FILE_MAX_AGE_DEFAULT': datetime.timedelta(seconds=43200),
    'SERVER_NAME': None,
    'SESSION_COOKIE_DOMAIN': None,
    'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_NAME': 'session',
    'SESSION_COOKIE_PATH': None,
    'SESSION_COOKIE_SAMESITE': None,
    'SESSION_COOKIE_SECURE': False,
    'SESSION_REFRESH_EACH_REQUEST': True,
    'TEMPLATES_AUTO_RELOAD': None,
    'TESTING': False,
    'TRAP_BAD_REQUEST_ERRORS': None,
    'TRAP_HTTP_EXCEPTIONS': False,
    'USE_X_SENDFILE': False}


We can add new key-values or change values as any dictionary in python.

..code:: python

    # add a directory for image upload
    app.config['UPLOAD_IMG_FOLDER'] = 'static/img'


However, for a large project,
if there are multiple environments, each with different set of config values, 
we can create a configuration file. Refer to the link below for more. 

https://pythonise.com/series/learning-flask/flask-configuration-files



Manipulating HTML
-----------------

There are various ways to pass variables into or manipulate html using flask.

Passing Variables
******************

We can use the double curly brackets ``{{ variable_name }}`` in html, and within flask
define a route. Within the render_template, we pass in the variable.

*In Python*

.. code:: python

    @app.route('/upload', methods=["POST"])
    def upload_file():
        img_path = 'static/img'
        img_name = 'img_{}.png'
        img = os.path.join(img_path, img_name)
        file = request.files['image_upload']
        file.save(img)

        return render_template('index.html', img_show=img)


*In HTML*

.. code:: html 

    <div class="row">
        <img class="img-thumbnail" src={{img_show}} alt="">
    </div>


If Conditions, Loops, etc.
***************************

We can implement python code in the html using the syntax, i.e., ``{% if something %}``.
However, note that we need to close it with the same synatx also, i.e. ``{% endif %}``.

*In Python*

.. code:: python

    @app.route('/upload', methods=["POST"])
    def upload_file():
        img_path = 'static/img'
        img_name = 'img_{}.png'
        img = os.path.join(img_path, img_name)
        file = request.files['image_upload']
        file.save(img)

        return render_template('index.html', img_show=img)

*In HTML*

.. code:: html 

    {% if img_show %}
    <div class="row">
        <img class="img-thumbnail" src={{img_show}} alt="">
    </div>
    {% endif %}


Requests
--------

There are a number of HTTP request methods. Below are the two commonly used ones.

+-----------+------------------------------------------------------------------------------------+
| ``GET``   | Sends data in unencrypted form to the server. E.g.  the ? values in URL            |
+-------------------+----------------------------------------------------------------------------+
| ``POST``  | Used to send HTML form data to server. Data received not cached by server.         |
+-----------+------------------------------------------------------------------------------------+



File Upload
-----------

Below shows up to upload a file, e.g., an image to a directory in the server.


*In HTML*

.. code:: html 

    <div class="row">
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="image_upload" accept=".jpg,.jpeg,.gif,.png" />
            <button type="submit" class="btn btn-primary">Submit</button>
        </form>
    </div>


.. code:: python

    import os
    from time import time

    @app.route('/upload', methods=["POST"])
    def upload_file():
        img_path = 'static/img'

        # delete original image
        if len(os.listdir(path)) != 0:
            img = os.listdir(path)[0]
            os.remove(os.path.join(path,img))

        # retrieve and save image with unique name
        img_name = 'img_{}.png'.format(int(time()))
        img = os.path.join(path, img_name)
        file = request.files['image_upload']
        file.save(img)

        return render_template('index.html')
        



Resources
---------

 * https://www.tutorialspoint.com/flask/index.htm