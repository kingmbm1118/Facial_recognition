# face-rec


## What's new
Face Recognition app for detecting faces 

Reads video stream from camera and live track faces stored in the DB (encoded faces)
>    server.py


----------- server setup instruction -------------
https://www.digitalocean.com/community/tutorials/how-to-deploy-a-flask-application-on-an-ubuntu-vps
sudo apt-get update
sudo apt-get install apache2 libapache2-mod-wsgi python-dev
sudo vi /etc/apache2/sites-available/FlaskApp.conf

for python 3:

sudo apt-get install libapache2-mod-wsgi-py3


apache log path:
/var/log/apache2/error.log

clear log:
----------
sudo bash -c 'echo > /var/log/apache2/error.log'

install python3 pip3
Install libsm6 libxext6:
    sudo apt update
    sudo apt install -y libsm6 libxext6
    sudo apt-get install libxrender1

Flask Install:
 pip install flask  --trusted-host=github.com   --trusted-host=codeload.github.com  --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --trusted-host=pypi.python.org -i https://pypi.python.org/simple/

pip install -r requirements.txt
pip freeze > requirements.txt

pip install opencv-python --cert D:\SandipDas\InstalledSoftware\ZscalerRootCerts\ZscalerRootCerts\ZscalerRootCertificate-2048-SHA256.crt  -i https://pypi.python.org/simple/



$env:FLASK_APP = "server.py"
python -m flask run --host=0.0.0.0 --port 5000




pip3 install Pillow==2.6.0
pip3 install scipy==1.1.0
pip3 install opencv-python tensorflow Flask-Cors Flask-SocketIO sklearn request eventlet

#Object arrays cannot be loaded when allow_pickle=False
pip3 install numpy==1.16.1

#from google.protobuf.pyext import _message tensorflow
pip install protobuf==3.6.0

---------running the server---------
nohup python3 serverSocket.py > server-socket-error.log 2>&1 &
echo $! > server-socket-pid.txt

-----checking----
ps -ef | grep python3

---stop the server----
kill `cat server-socket2-pid.txt`
rm server-socket2-pid.txt


nohup node index.js > socket-client.log 2>&1 &
echo $! > socket-client-pid.txt

watch tail -n 15 server-socket-error.log

/etc/apache2/sites-available/FlaskApp.conf


Socket Client:
--------------

Install nodejs and npm 
    https://nodejs.org/en/download/


cd /path/to/project/ && npm i

to run the socket client:
------------------------
node index.js


Socket Server:
--------------

install extra modules
---------------------
pip install -r requirements.txt

to run the socket server:
------------------------
python serverSocket.py