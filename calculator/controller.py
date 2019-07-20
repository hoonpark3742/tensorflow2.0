from calculator.model import CalculatorModel

class CalculatorController:
    def __init__(self):
        self.model = CalculatorModel()
    @staticmethod
    def print_menu():
        print('0.  종료')
        print('1.  +')
        print('2.  -')
        print('3.  *')
        print('4.  /')
        menu = input('메뉴 선택\n')
        return int(menu)

    def run(self):
        model = self.model
        while 1:
            menu = self.print_menu()

            if menu == 0:
                break
            else :
                self.show(model.hook(menu))

    @staticmethod
    def show(param):
        print('RESULT : %s' % param)