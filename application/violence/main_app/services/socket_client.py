import json
from PyQt5 import QtCore, QtWebSockets

from ..controller.config import Config
from ..model.camera import Camera


class ConnectToWSS(QtCore.QObject):
    def __init__(self, camera:Camera):
        super().__init__()
        self.camera = camera
        self.config = Config()
        self.init_socket()
        
    def init_socket(self):
        self.client =  QtWebSockets.QWebSocket("",QtWebSockets.QWebSocketProtocol.Version13, None)
        self.client.open(QtCore.QUrl(self.config.SOCKET_SERVER))
        self.client.error.connect(self.onError)
        self.client.textMessageReceived.connect(self.onMessageReceived)
        self.client.disconnected.connect(self.onDisconnected)
        self.client.connected.connect(self.onConnected)

    def onConnected(self):
        print("-------- Connected to server ^-^ ")
        self.send_Identification_message()

    # handle command from server
    def onMessageReceived(self, response):
        print("-------- Received message from server: ", response)
        try:
            response = json.loads(response)
            for k, v in response.items():
                if response["CAM_CODE"] == self.camera.code:
                    if response["EVENT_ID"] == 2:
                        self.camera.is_streaming = True

                    elif response["EVENT_ID"] == 3:
                        self.camera.is_streaming = False

        except Exception as e:
            pass  

    def send_Identification_message(self):
        data = {
                "EVENT_ID": 1,
                "SYS_CODE": "DRHP",
                "SEVER_NAME": "DR20231"
                }
        print("-------- Send message to server: ", data)
        doc = json.dumps(data)
        self.client.sendTextMessage(doc)

        print("-------- Send Identification message to server !!!")

    def onError(self, error_code):
        print("Error code: {}".format(error_code))
        print("-------- ", self.client.errorString())

    def onDisconnected(self):
        print("Disconnect to server !!!")
        self.reconnect()
        print("Reconnected to server !!!")

    def reconnect(self):
        print("-------- Reconnecting to server ...")
        self.client.open(QtCore.QUrl(self.config.SOCKET_SERVER))


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    import sys
    app = QtCore.QCoreApplication(sys.argv)
    wss = ConnectToWSS()
    sys.exit(app.exec_())