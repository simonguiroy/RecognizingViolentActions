import requests
import os

ID = "1sdnZwKl91SO2clkffERbiE7R6-SGakWu"
DESTINATION = "datasets/ViolentHumanActions_v2.zip"


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


os.system("mkdir -p datasets")
download_file_from_google_drive(ID, DESTINATION)
os.system("cd datasets; unzip ViolentHumanActions_v2.zip; rm ViolentHumanActions_v2.zip")
