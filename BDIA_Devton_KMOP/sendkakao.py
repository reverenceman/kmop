import requests
import json

#1.
with open(r"kakao_code.json","r") as fp:
    tokens = json.load(fp)


url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

# kapi.kakao.com/v2/api/talk/memo/default/send 

headers={
    "Authorization" : "Bearer " + tokens["access_token"]
}

data1={
    "template_object": json.dumps({
        "object_type":"text",
        "text":"넘어짐이 감지되었습니다!",
        "link":{
            "web_url":"https://tidy-whole-titmouse.ngrok-free.app/",
            "mobile_web_url": "https://tidy-whole-titmouse.ngrok-free.app/"
        },
        "button_title": "웹캠 확인"
    })
}

data2={
    "template_object": json.dumps({
        "object_type":"text",
        "text":"4시간동안 움직임이 없습니다. 욕창 주의가 요구됩니다.",
        "link":{
            "web_url":"https://www.naver.com/",
            "mobile_web_url": "https://www.naver.com/"
        },
        "button_title": "웹캠 확인"
    })
}

data3={
    "template_object": json.dumps({
        "object_type":"text",
        "text":"방에 들어간지 12시간이 넘었습니다.",
        "link":{
            "web_url":"https://tidy-whole-titmouse.ngrok-free.app/",
            "mobile_web_url": "https://tidy-whole-titmouse.ngrok-free.app/"
        },
        "button_title": "웹캠 확인"
    })
}

def send_messege1():
    response = requests.post(url, headers=headers, data=data1)
    response.status_code
    print(response.status_code)
    if response.json().get('result_code') == 0:
        print('낙상메시지를 성공적으로 보냈습니다.')
    else:
        print('낙상메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))
        
def send_messege2():
    response = requests.post(url, headers=headers, data=data2)
    response.status_code
    print(response.status_code)
    if response.json().get('result_code') == 0:
        print('욕창메시지를 성공적으로 보냈습니다.')
    else:
        print('욕창메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))
        
def send_messege3():
    response = requests.post(url, headers=headers, data=data3)
    response.status_code
    print(response.status_code)
    if response.json().get('result_code') == 0:
        print('갇힘메시지를 성공적으로 보냈습니다.')
    else:
        print('갇힘메시지를 성공적으로 보내지 못했습니다. 오류메시지 : ' + str(response.json()))