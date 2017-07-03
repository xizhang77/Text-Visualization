# Text-Visualization

1. The structure of this project
The files in /flask and /lib folders are Flask framework and extensions that will be used for the application. All the executable files are in the /app folder.

Data processing is done in app/cluster.py
app/app.py is used to start the server and send data to javascript.
D3 code could be found in app/templates/

2. How to run this code

First, go to the app folder, which is the basic folder for our application.

To start the application, just run this script app.py as an argument to the Python interpreter from the virtual environment:

python app.py

After the server initializes it will listen on port 5000 waiting for connections. Now open up the web browser and enter the following URL in the address field:

http://localhost:5000

Now you can play on the text visualization tool!
