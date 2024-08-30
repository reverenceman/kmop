# pip install -r requirements.txt
import cv2
import numpy as np
import time
import ultralytics
import threading
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
from deep_sort_realtime.deepsort_tracker import DeepSort

# 모델 초기화 함수
def initialize_variables():
    global model, cap
    model = YOLO("yolov8s-pose.pt")
    model.fuse()

    ip_camera_url = "http://58.78.240.146:12060//videostream.cgi?user=admin&pwd=sk201507&resolution=32&rate=0"
    cap = cv2.VideoCapture(ip_camera_url)       # 홈캠 이용시
    #cap = cv2.VideoCapture(0)                  # 웹캠 이용시
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    return model

# 바운딩 박스 그리기 함수
def draw_rectangle(frame,event, x, y, flags, param):
    global start_x, start_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True  # 마우스 왼쪽 버튼을 눌렀을 때 drawing을 True로 설정
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Webcam", frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False  # 마우스 왼쪽 버튼을 놓았을 때 drawing을 False로 설정
        width, height = x - start_x, y - start_y
        if width > 0 and height > 0:
            box_name = input("방 이름을 입력하세요: ")  # 사용자에게 이름을 묻기
            boxes.append(((start_x, start_y, width, height), box_name))
                  
# 한글 출력 함수
def draw_korean_text(frame, text, position, font_size, color):
    font = ImageFont.truetype('NanumGothic.ttf', font_size)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position,  text, font=font, fill=color)
    img = np.array(img_pil)
    return img

# 객체 탐지 함수
def perform_detection(frame, model, CONFIDENCE_THRESHOLD, person_class_id=0):
    result = []
    detection = model(frame,verbose=False)[0]
    for data in detection.boxes.data.tolist():
        confidence = float(data[4])
        class_id = int(data[5])
        if class_id == person_class_id and confidence >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            result.append([[xmin, ymin, xmax-xmin, ymax-ymin], confidence])
    return result, detection

# 바운딩 박스 및 글자 출력 함수
def draw_output(frame, intersected_box_name,insided_room,detected_person_count):
    global no_move_time_hour, no_move_time_minute,no_move_time_sec,no_move_time,room_in_time_hour,room_in_time_minute,room_in_time_sec,web_box
    # 초를 분과 시간으로 변환
    if no_movement_elapsed_time is not None:
        no_move_time = int(no_movement_elapsed_time)
        no_move_time_hour= int(no_move_time//3600)
        no_move_time_minute=int((no_move_time-no_move_time_hour*3600)//60)
        no_move_time_sec= int(no_move_time-(no_move_time_hour*3600)-(no_move_time_minute*60))
    else:
        no_move_time = 0 
    if insided_room_time is not None:
        room_in_time = int(insided_room_time)
        room_in_time_hour= int(room_in_time//3600)
        room_in_time_minute=int((room_in_time-room_in_time_hour*3600)//60)
        room_in_time_sec= int(room_in_time-(room_in_time_hour*3600)-(room_in_time_minute*60))
    else:
        room_in_time = 0    
    # 저장된 바운딩 박스들과 이름을 화면에 그림
    for (x, y, w, h), box_name in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame = draw_korean_text(frame, box_name, (x, y-25), 20, (0, 255, 0))
        if insided_room:
            if box_name == intersected_box_name:
                web_box = box_name
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                text_combined1 = f"{box_name}에 입실 중"
                frame = draw_korean_text(frame,text_combined1, (x, y-25), 20, (0, 165, 255))
                text_combined2 = f"{box_name}에 들어간지 {str(room_in_time_hour)} 시간 {str(room_in_time_minute)} 분 {str(room_in_time_sec)} 초가 경과하였습니다."
                frame = draw_korean_text(frame, text_combined2, (150, 340), 16, (255, 255, 255))
                if room_in_time_minute > 1:
                    frame = draw_korean_text(frame, text_combined1, (x, y-25), 20, (0, 0, 255))
                    frame = draw_korean_text(frame, text_combined2, (150, 340), 16, (0, 0, 255))
    if not insided_room:
        if detected_person_count == 0:
            frame = draw_korean_text(frame, "탐지된 사람이 없어 움직임 감지를 중지합니다.", (150, 340), 16, (255, 255, 255))
        if detected_person_count > 1:
            frame = draw_korean_text(frame, "탐지된 사람이 2명 이상이여서 움직임 감지를 중지합니다.", (130, 340), 16, (255, 255, 255))
        else:
            if no_move_time > 0:
                text_combined3 = f"움직임이 {str(no_move_time_hour)} 시간 {str(no_move_time_minute)} 분 {str(no_move_time_sec)} 초 동안 감지되지 않았습니다."
                frame = draw_korean_text(frame, text_combined3, (130, 340), 16, (255, 255, 255))
            if no_move_time > 60:
                frame = draw_korean_text(frame, text_combined3, (130, 340), 16, (0, 165, 255))
            if no_move_time > 90:
                frame = draw_korean_text(frame, text_combined3, (130, 340), 16, (0, 0, 255))
    return frame

# 방 출입 여부 인식 함수
def check_room_inout(detected_person_count):
    global intersected_box_name,insided_room,person_inside_box_previous,box_state,elapsed_time,elapsed_time
    person_inside_box = False
    
    # 확인하려는 바운딩 박스와 사람 객체 바운딩 박스의 교차 여부를 확인
    if detected_person_count == 1:
        for (x, y, w, h), box_name in boxes:
            if xmin >= x and xmax <= x + w and ymin >= y and ymax <= y + h:
                person_inside_box = True
                intersected_box_name = box_name    
                     
    # 방 출입여부 확인
    if person_inside_box and not person_inside_box_previous:
        box_state = True
    elif not person_inside_box and person_inside_box_previous and detected_person_count == 1:
        box_state = False
    if detected_person_count == 0 and box_state:
        if not insided_room:
            insided_room = True
            elapsed_time = time.time()
            print(f"사람이 {intersected_box_name}에 들어갔습니다.")
        else:
            insided_room_time = time.time() - elapsed_time
            print(f"{intersected_box_name}에 들어간지 {insided_room_time:.0f}초가 경과하였습니다.")
    else:
        if insided_room:
            insided_room = False
            insided_room_time = 0.0 
            
    # 이전 박스의 상태를 업데이트
    person_inside_box_previous = person_inside_box

# Deepsort 알고리즘 함수
def Deepsort_traking(frame, tracker, results):
    global detected_person_count,xmin, ymin, xmax, ymax
    detected_person_count = 0
    if no_movement_elapsed_time is not None:
        no_move_time = int(no_movement_elapsed_time)
    else:
        no_move_time = 0
    if detected_person_count < 2:
        tracks = tracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            if no_move_time > 60:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 165, 255), 2)
            if no_move_time > 90 or fall_state:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            detected_person_count += 1
    
# 움직임 감지 함수        
def motion_detection(frame,insided_room,detected_person_count):
    global previous_frame,no_movement_start_time,last_movement_time,no_movement_elapsed_time,xmin, ymin, xmax, ymax
    movement_threshold = 300000 # 움직임 감지 민감도
    frame_height, frame_width, _ = frame.shape
    if insided_room == False:
        if detected_person_count == 0:
            no_movement_start_time = None
            previous_frame = None
        elif detected_person_count > 1:
            no_movement_start_time = None
            previous_frame = None
        elif detected_person_count == 1:
            if previous_frame is not None and frame is not None:
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(xmax, frame_width)
                ymax = min(ymax, frame_height)
                frame_diff = cv2.absdiff(cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY), cv2.cvtColor(previous_frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY))
                movement = np.sum(frame_diff)
                if movement > movement_threshold:
                    no_movement_elapsed_time = 0
                    if last_movement_time is None:
                        last_movement_time = time.time()
                    no_movement_start_time = None
                elif last_movement_time is not None and no_movement_start_time is None:
                    no_movement_start_time = time.time()
            previous_frame = frame
        if no_movement_start_time is not None:
            no_movement_elapsed_time = time.time() - no_movement_start_time
    return frame
    
    
def main_loop():
    global previous_frame, last_movement_time, no_movement_start_time, insided_room_time, no_movement_elapsed_time, annotated_frame, CONFIDENCE_THRESHOLD, xmin, ymin, xmax, ymax, drawing, start_x, start_y, boxes, box_name, intersected_box_name, box_state, insided_room, person_inside_box_previous, fall_state, sit_count, detected_person_count,no_move_time_hour, no_move_time_minute,no_move_time_sec,no_move_time,web_frame,room_in_time_hour,room_in_time_minute,room_in_time_sec,web_box
    
    previous_frame = last_movement_time = no_movement_start_time = insided_room_time = no_movement_elapsed_time = annotated_frame = None
    CONFIDENCE_THRESHOLD = 0.65 
    xmin = ymin = xmax = ymax = room_in_time_hour=room_in_time_minute=room_in_time_sec =0 
    drawing = False
    start_x, start_y = -1, -1
    boxes = []  
    box_name = intersected_box_name = ""
    box_state = insided_room= person_inside_box_previous = fall_state = False     
    sit_count = 0
    detected_person_count = 0
    
    initialize_variables()
    tracker = DeepSort(max_age=3)

    cv2.namedWindow("Webcam")
    cv2.setMouseCallback("Webcam", lambda event, x, y, flags, param: draw_rectangle(frame, event, x, y, flags, param))
    
    while True:
        ret, frame = cap.read()
        # 바운딩 박스 및 글자 출력
        frame = draw_output(frame,intersected_box_name,insided_room,detected_person_count)
        
        # 객체 탐지 수행
        results, detection = perform_detection(frame, model, CONFIDENCE_THRESHOLD, person_class_id = 0)

        # Deepsort traking 알고리즘 실행
        Deepsort_traking(frame, tracker, results)
        
        # 방 출입 감지 실행
        check_room_inout(detected_person_count)
        
        # 움직임 감지 실행
        frame = motion_detection(frame,insided_room,detected_person_count)
                
        # 영상 출력
        annotated_frame = detection.plot(boxes=False)
        web_frame = frame
        cv2.imshow("Webcam", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # 'Esc' 키로 종료
            break
    cap.release()
    cv2.destroyAllWindows()
    
main_loop()