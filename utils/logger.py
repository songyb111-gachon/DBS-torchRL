import sys
import logging
from datetime import datetime
import os


def setup_logger(log_dir="log"):
    # 로그를 저장할 디렉토리 설정
    os.makedirs(log_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    # 현재 파일 이름과 실행 시간 가져오기
    if '__file__' in globals():
        current_file = os.path.splitext(os.path.basename(__file__))[0]  # 현재 파일 이름(확장자 제거)
    else:
        current_file = "interactive"  # 인터프리터나 노트북 환경에서 기본 파일 이름 사용

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 현재 시간
    log_filename = os.path.join(log_dir, f"{current_file}_{current_datetime}.log")  # log 폴더에 파일 저장

    # 로그 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),  # 동적으로 생성된 파일 이름 사용
            logging.StreamHandler()  # 콘솔 출력
        ]
    )

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for file in self.files:
                file.write(data)
                file.flush()  # 실시간 저장

        def flush(self):
            for file in self.files:
                file.flush()

    # stdout을 파일과 콘솔로 동시에 출력
    log_file = open(log_filename, "a")
    sys.stdout = Tee(sys.stdout, log_file)

    return log_filename  # 로그 파일 이름 반환
