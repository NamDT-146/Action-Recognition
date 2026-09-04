import os
import cv2
import json
import yaml
import base64
import logging
import numpy as np

from datetime import datetime, timedelta


def read_yaml_file(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def write_yaml_file(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)
        

def create_folder(path:str):
    if os.path.exists(path):
        return
    os.mkdir(path)


def read_json_data(fp):
    assert os.path.isfile(fp), "{} is not a file".format(fp)
    with open(fp, mode='r', encoding='utf-8') as f:
        dt = json.load(f)
        return dt


def write_json_data(fp, json_data:str):
    with open(fp, mode='w', encoding='utf8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def image_to_base64(image):
    img_to_byte = cv2.imencode('.jpg', image)[1].tobytes()
    byte_to_base64 = base64.b64encode(img_to_byte)
    return byte_to_base64.decode('ascii')  # chuyển về string


def base64_to_image(base64_string):
    base64_to_byte = base64.b64decode(base64_string)
    byte_to_image = np.frombuffer(base64_to_byte, np.uint8)
    image = cv2.imdecode(byte_to_image, cv2.IMREAD_COLOR)
    return image


def check_save_dir(root_dir):
    date_dir = datetime.now().strftime("%Y-%m-%d")
    dir_path = os.path.join(root_dir, date_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


# Function to convert string time to timedelta
def convert_to_timedelta(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
    return timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second)


def get_logger(file_name, mode="w", name_logger='urbanGUI') -> logging.Logger:
    create_folder(os.path.dirname(file_name))
    if not os.path.exists(file_name):
        open(file_name, 'x').close()
    logging.basicConfig(filename=file_name,
                        filemode=mode,
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    fh = logging.FileHandler(file_name)
    fh.setLevel(logging.DEBUG)
    logger = logging.getLogger(name_logger)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(fh)
    return logger


# def dial_numbers(numbers_list = ["+84349403820"]):
#     """Dials one or more phone numbers from a Twilio phone number."""
    
#     # Twilio phone number goes here. Grab one at https://twilio.com/try-twilio
#     # and use the E.164 format, for example: "+12025551234"
#     TWILIO_PHONE_NUMBER = "+12532317439"

#     # URL location of TwiML instructions for how to handle the phone call
#     TWIML_INSTRUCTIONS_URL = \
#     "http://static.fullstackpython.com/phone-calls-python.xml"

#     # replace the placeholder values with your Account SID and Auth Token
#     # found on the Twilio Console: https://www.twilio.com/console
#     client = Client("ACee0489cf3bc9070f4d72ff3441cd07b1", "efd3d9809a09a0566516bc3c14c9328e")
#     for number in numbers_list:
#         print("Dialing " + number)
#         # set the method to "GET" from default POST because Amazon S3 only
#         # serves GET requests on files. Typically POST would be used for apps
#         client.calls.create(to=number, from_=TWILIO_PHONE_NUMBER,
#                             url=TWIML_INSTRUCTIONS_URL, method="GET")
