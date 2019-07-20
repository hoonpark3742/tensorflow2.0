from simple_classification_algorithm.model import IrisModel


class IrisController:
    def __init__(self):
        self.model = IrisModel()

    @staticmethod
    def print_menu():
        print('0. 종료')
        print('1. 아이리스 데이터 출력')
        print('2. 아이리스 산점도 그리기')
        print('3. 에포크 대비 잘못 분류된 오차 그래프')
        print('4. 2차원 데이터셋의 결정 경계 시각화')
        print('5. Adaline 그래프 그리기')
        return int(input('메뉴 선택 \n'))

    @staticmethod
    def show(param):
        print('RESULT : %s' % param)

    def run(self):
        model = self.model
        while 1:
            menu = self.print_menu()
            if menu == 0:
               break
            elif menu == 1:
               self.show(model.get_iris())
            elif menu == 2:
               model.draw_scatter()
            elif menu == 3:
               model.draw_errors()
            elif menu == 4:
               model.plot_decision_regions()
            elif menu == 5:
               model.draw_adaline_graph()

