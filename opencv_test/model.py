import cv2

class LenaModel:
    def __init__(self):
        self.fname = './data/lena.jpg'

    def execute(self):
        """
        https://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html
        """
        print('cv2 버전: %s' % cv2.__version__)
        original = cv2.imread(self.fname, cv2.IMREAD_COLOR)
        gray = cv2.imread(self.fname, cv2.IMREAD_GRAYSCALE)
        unchanged = cv2.imread(self.fname, cv2.IMREAD_UNCHANGED)
        """
        이미지 읽기에는 위 3가지 속성값이 있습니다.
        대신에 1, 0, -1을 사용해도 됨
        """
        cv2.imshow('Original', original)
        cv2.imshow('Gray', gray)
        cv2.imshow('Unchanged', unchanged)
        cv2.waitKey(0)
        """
        키보드 입력대기 시간 정하는 함수로 0이면 무한대기
        """
        cv2.destroyAllWindows()
        """
        윈도우 종료
        """