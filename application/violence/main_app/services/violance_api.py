import requests
import json
from .base_api import BaseAPI
import copy

class ViolenceWarningAPI(BaseAPI):
    def __init__(self):
        super().__init__()
        self._URI_VIOLANCE_EVENT = f"{self.config.WEB_HOST}/Service/api/event"
        self._MOBIFONE_TOKEN = None 

    def post_violance_event(self, json_data):
        self.get_access_token()
        try:
            print("📦 Dữ liệu gửi lên ATIN:")
            print(json.dumps(json_data, indent=2, ensure_ascii=False))
            
            response = requests.post(
                self._URI_VIOLANCE_EVENT,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._token}"
                },
                json=json_data,
                timeout=10
            )

            print("🔁 Status:", response.status_code)

            if response.status_code == 200:
                print("✅ Push thành công!")
                return response.json()
            elif response.status_code in [400, 500]:
                print(f"⚠️ Lỗi {response.status_code}")
                return {}
            else:
                print(f"🔄 Thử lại vì mã lỗi {response.status_code}")
                self.get_access_token()
                response = requests.post(
                    self._URI_VIOLANCE_EVENT,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._token}"
                    },
                    json=json_data,
                    timeout=20
                )
                return response.json()

        except requests.exceptions.ConnectTimeout:
            raise requests.exceptions.ConnectTimeout("Timeout When Posting violance Event")
        except Exception as e:
            print(f"❌ Exception: {e}")
            return {}

   