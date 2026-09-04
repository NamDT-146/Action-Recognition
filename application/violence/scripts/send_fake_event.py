#!/usr/bin/env python3
import sys
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


# ==============================
# Config - edit these if needed
# ==============================
WEB_HOST = "http://192.168.1.144:42048"  # e.g., http://your-web-host
USERNAME = "mbf"
PASSWORD = "Mbf@123456"
CLIENT_ID = "IAC_Cloud"
CLIENT_SECRET = "1a82f1d60ba6353bb64a8fb4b05e4bc4"


def get_access_token(web_host: str, username: str, password: str,
                     client_id: str, client_secret: str,
                     timeout_seconds: int = 10) -> str:
    token_url = f"{web_host}/Service/api/token/auth"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    response = requests.post(token_url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or "access_token" not in data:
        raise RuntimeError("Token response missing 'access_token'")
    return data["access_token"]


def fetch_devices(web_host: str,
                  token: str,
                  comp_id: str,
                  server_name: Optional[str] = None,
                  event_type_id: str = "202",
                  timeout_seconds: int = 15) -> List[Dict[str, Any]]:
    """Fetch list of devices; try multiple endpoint/param variants for compatibility."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    base_params = f"page=1&itemsPerPage=999&sortBy=id&sortDesc=true&compId={comp_id}&eventTypeId={event_type_id}"
    variants = []
    # With serverName filter if provided
    if server_name:
        variants.append((f"{web_host}/Service/api/Device?{base_params}&serverName={server_name}", "Device+serverName"))
        variants.append((f"{web_host}/Service/api/device?{base_params}&serverName={server_name}", "device+serverName"))
    # Without serverName (broader)
    variants.append((f"{web_host}/Service/api/Device?{base_params}", "Device"))
    variants.append((f"{web_host}/Service/api/device?{base_params}", "device"))
    # Some deployments require status=1
    if server_name:
        variants.append((f"{web_host}/Service/api/device?{base_params}&status=1&serverName={server_name}", "device+status+serverName"))
    variants.append((f"{web_host}/Service/api/device?{base_params}&status=1", "device+status"))

    last_error_text = None
    for url, label in variants:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_seconds)
            # Accept 200 only
            if resp.status_code != 200:
                last_error_text = f"{label} -> HTTP {resp.status_code}: {resp.text[:300]}"
                continue
            data = resp.json()
            # Expected structure: {"items": [...]} or a list
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                items = data["items"]
            elif isinstance(data, list):
                items = data
            else:
                # Try common wrappers
                if isinstance(data, dict):
                    for key in ("data", "result", "rows"):
                        if isinstance(data.get(key), list):
                            items = data[key]
                            break
                    else:
                        last_error_text = f"{label} -> Unexpected JSON shape: {str(list(data.keys()))[:120]}"
                        continue
                else:
                    last_error_text = f"{label} -> Unexpected JSON type: {type(data)}"
                    continue

            if items:
                return items
        except Exception as e:
            last_error_text = f"{label} -> Exception: {e}"
            continue

    if last_error_text:
        print(f"Device fetch attempts failed. Last error: {last_error_text}")
    return []


def select_device(devices: List[Dict[str, Any]], device_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Pick a device by code if provided, else the first available."""
    if not devices:
        return None
    if device_code:
        for d in devices:
            try:
                if str(d.get("code", "")) == str(device_code):
                    return d
            except Exception:
                continue
    return devices[0]


def build_event_from_device(device: Dict[str, Any],
                            list_media: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build payload using real IDs from a device record.

    Fields expected on device: id, compId, areaId, eventTypeId
    """
    event_id = str(uuid.uuid4())
    access_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if list_media is None:
        list_media = []

    violance_event = {
        "violanceType": "1",
    }

    payload = {
        "eventId": event_id,
        "accessTime": access_time,
        "eventTypeId": str(device.get("eventTypeId", "")),
        "areaId": str(device.get("areaId", "")),
        "compId": str(device.get("compId", "")),
        "deviceId": str(device.get("id", "")),
        "listMedia": list_media,
        "violanceEvent": violance_event,
    }
    return payload


def build_fake_event(
    event_type_id: str = "203",
    area_id: str = "101",
    comp_id: str = "39",
    device_id: str = "1001",
    list_media: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    # Match the schema used in the application
    event_id = str(uuid.uuid4())
    access_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if list_media is None:
        list_media = []

    violance_event = {
        # Minimal example payload for the nested object
        "violanceType": "1",
    }

    payload = {
        "eventId": event_id,
        "accessTime": access_time,
        "eventTypeId": str(event_type_id),
        "areaId": str(area_id),
        "compId": str(comp_id),
        "deviceId": str(device_id),
        "listMedia": list_media,
        "violanceEvent": violance_event,
    }
    return payload


def post_event(web_host: str, token: str, payload: Dict[str, Any], timeout_seconds: int = 15) -> Dict[str, Any]:
    url = f"{web_host}/Service/api/event"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    # Some servers return 200 with JSON, others might return 204 No Content
    if response.status_code == 204 or not response.content:
        return {"status": response.status_code, "message": "No Content"}
    try:
        return response.json()
    except Exception:
        return {"status": response.status_code, "text": response.text}


def main(argv: List[str]) -> int:
    # Allow simple CLI overrides:
    # argv: web_host username password [comp_id] [server_name] [event_type_id] [device_code]
    web_host = WEB_HOST
    username = USERNAME
    password = PASSWORD
    comp_id = "39"
    server_name = "Server 01"
    event_type_id = "202"
    device_code = None
    if len(argv) >= 2:
        web_host = argv[1]
    if len(argv) >= 3:
        username = argv[2]
    if len(argv) >= 4:
        password = argv[3]
    if len(argv) >= 5:
        comp_id = argv[4]
    if len(argv) >= 6:
        server_name = argv[5]
    if len(argv) >= 7:
        event_type_id = argv[6]
    if len(argv) >= 8:
        device_code = argv[7]

    print(f"Getting token from: {web_host}/Service/api/token/auth")
    try:
        token = get_access_token(web_host, username, password, CLIENT_ID, CLIENT_SECRET)
    except Exception as e:
        print(f"Failed to get access token: {e}")
        return 1

    # Fetch devices and build payload from a real device
    print("Token acquired. Fetching devices...")
    try:
        devices = fetch_devices(web_host, token, comp_id, server_name, event_type_id)
    except Exception as e:
        print(f"Failed to fetch devices: {e}")
        return 2

    if not devices:
        print("No devices found with the given filters.")
        return 3

    device = select_device(devices, device_code=device_code)
    if device is None:
        print("Failed to select a device.")
        return 4

    print("Using device:")
    try:
        print(json.dumps(device, ensure_ascii=False, indent=2))
    except Exception:
        print(str(device))

    print("Building event payload from selected device...")
    payload = build_event_from_device(device)
    print("Payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"Posting event to: {web_host}/Service/api/event")
    try:
        result = post_event(web_host, token, payload)
        print("Response:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except requests.exceptions.ConnectTimeout:
        print("Timeout when posting the event")
        return 5
    except Exception as e:
        print(f"Error posting event: {e}")
        return 6

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))


