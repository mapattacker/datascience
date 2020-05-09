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
second is for css or other static files like images, will be placed under /static

.. code:: bash

    ├── app.py
    ├── static
    │   └── img
    │       └── image.png
    └── templates
        └── index.html
