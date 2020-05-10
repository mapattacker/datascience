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
--------------------------

We can implement python code in the html using the syntax, i.e., ``{% if something %}``.
However, note that we need to close it with the same synatx also, i.e. ``{% endif %}``.

*In Python*
.. code:: python

    # in python
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


Resources
---------

 * https://www.tutorialspoint.com/flask/index.htm