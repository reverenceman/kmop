import requests
import json

url = 'https://kauth.kakao.com/oauth/token'
rest_api_key = '14196365806c4ca58b5113fed17206e2'
redirect_uri = 'https://example.com/oauth'
authorize_code = 'Jhr6bOYTKMWBAdd9pAHKdJeUDKaZ0a2K1n2kOv-ieRj_UHHuQXpCEQAAAAQKKwzUAAABkt4BHhz6Fwx8Dt1GgQ'

# 토큰을 요청하고 저장하는 초기 단계
data = {
    'grant_type': 'authorization_code',
    'client_id': rest_api_key,
    'redirect_uri': redirect_uri,
    'code': authorize_code,
}

response = requests.post(url, data=data)
tokens = response.json()

# 토큰을 파일로 저장하기
if "access_token" in tokens:
    with open("kakao_code.json", "w") as fp:
        json.dump(tokens, fp)
        print("Tokens saved successfully")
else:
    print("Failed to get tokens:", tokens)

# Kakao 클래스 정의
class Kakao:
    def __init__(self):
        self.app_key = rest_api_key  # REST API 키 설정

        # 저장된 JSON 파일을 읽어와서 토큰 정보를 로드
        try:
            with open("kakao_code.json", "r") as fp:
                self.tokens = json.load(fp)
        except FileNotFoundError:
            self.tokens = None
            print("Token file not found")

        # 토큰이 있으면 갱신
        if self.tokens:
            self.refresh_token()

    # 카카오 토큰 갱신하기
    def refresh_token(self):
        url = "https://kauth.kakao.com/oauth/token"
        data = {
            "grant_type": "refresh_token",
            "client_id": self.app_key,
            "refresh_token": self.tokens['refresh_token']
        }

        response = requests.post(url, data=data)

        # 갱신된 토큰 내용 확인
        result = response.json()

        # 갱신된 내용으로 파일 업데이트
        if 'access_token' in result:
            self.tokens['access_token'] = result['access_token']

        if 'refresh_token' in result:
            self.tokens['refresh_token'] = result['refresh_token']
        
        # 토큰을 파일로 저장
        with open("kakao_code.json", "w") as fp:
            json.dump(self.tokens, fp)

        print("Tokens refreshed and saved successfully")

# Kakao 클래스 인스턴스 생성 및 토큰 갱신 테스트
kakao = Kakao()
