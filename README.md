# Text-Visualization

## The structure of this project

Here is the whole structure
```
/application
    setup.py
    /application
        __init__.py
        cluster.py
        views.py
        /data
            some text files
        /templates
            post.html
            ...
    /flask
    /lib
```
The files in /flask and /lib folders are Flask framework and extensions that will be used for the application. All the executable files are in the /application folder.

* Data processing is done in *cluster.py* file
* The Flask application object creation is in the *\__init\__.py* file
* All the view functions are in the views.py file and imported in the *\__init\__.py* file.
* D3 code could be found in *templates/post.html*


## How to run this code

First, go to the /application folder, which is the basic folder for our application.

In order to run the application, export an environment variable that tells Flask where to find the application instance, install and run the application you need to issue the following commands.

```
export FLASK_APP=application
export FLASK_DEBUG=true
pip install -e .
flask run
```
After the server initializes it will listen on port 5000 waiting for connections. Now open up the web browser and enter the following URL in the address field:

http://localhost:5000

Now you can play on the text visualization tool!
