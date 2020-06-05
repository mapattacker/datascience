FastAPI
=======

FastAPI is one of the next generation python web framework that uses 
ASGI (asynchronous server gateway interface) instead of the old WSGI.
It also includes a number of useful functions to make API creations easier.


Request & Response Schema
-------------------------

FastAPI uses pydantic to define the schema of the request & response APIs.
This allows the auto generation in the OpenAPI documentations, and for the 
former, for validating the schema when a request is received.

For example given the json:

.. code:: json

    {
        "ANIMAL_DETECTION": {
            "boundingPoly": {
            "normalizedVertices": [
                {
                "x": 0.406767,
                "y": 0.874573,
                "width": 0.357321,
                "height": 0.452179,
                "score": 0.972167
                },
                {
                "x": 0.56781,
                "y": 0.874173,
                "width": 0.457373,
                "height": 0.452121,
                "score": 0.982109
                }
            ]
        },
        "name": "Cat"
        }
    }

We can define in pydantic as below, using multiple basemodels for each level.

.. code:: python

    class lvl3_list(BaseModel):
        x: float
        y: float
        width: float
        score: float

    class lvl2_item(BaseModel):
        normalizedVertices: List[lvl3_list]

    class lvl1_item(BaseModel):
        boundingPoly: lvl2_item
        name: str = "Human"

    class response_item(BaseModel):
        HUMAN_DETECTION: lvl1_item

    RESPONSE_SCHEMA = response_item


We do the same for the request schema and place them in the post definition.

.. code:: python

    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    from typing import List

    import json
    import base64
    import numpy as np

    @app.post('/api', response_model= RESPONSE_SCHEMA)
    def human_detection(request: REQUEST_SCHEMA):

        JScontent = json.loads(request.json())
        encodedImage = JScontent['requests'][0]['image']['content']
        npArr = np.fromstring(base64.b64decode(encodedImage), np.uint8)
        imgArr = cv2.imdecode(npArr, cv2.IMREAD_ANYCOLOR)
        pred_output = model(imgArr)

        return pred_output


Render Template
---------------

We can render templates like html, and pass variables into html using the below.
Like flask, in html, the variables are called with double curly brackets ``{{variablemame}}``.

.. code:: python

    from fastapi import FastAPI
    from fastapi.templating import Jinja2Templates

    app = FastAPI()
    templates = Jinja2Templates(directory="templates")


    @app.get('/')
    def index():
        UPLOAD_URL = '/upload/url'
        MODULE = 'name of module'
        return templates.TemplateResponse('index.html', \
                                {"upload_url": UPLOAD_URL, "module":MODULE})


OpenAPI
-------

OpenAPI documentations of Swagger UI or Redoc are automatically generated.
You can access it at the endpoints of ``/docs`` and ``/redoc``.

.. code:: python

    app = FastAPI(title="Human Detection API",
                    description="Submit Image to Return Detected Humans in Bounding Boxes",
                    version="1.0.0")