Flask
======

Flask is a micro web framework written in Python. 
It is easy and fast to implement with the knowledge of basic web development and REST APIs.


Basics
------

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
second is for css or other static files like images, will be placed under /static

.. code:: bash

    ├── app.py
    ├── static
    │   └── img
    │       └── image.png
    └── templates
        └── index.html
