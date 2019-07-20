from mail_checker.mail_checker_model import MailChecker

class MailCheckerController:
    def __init__(self):
        self.model = MailChecker()
    @staticmethod
    def print_menu():
        print('0.  종료')
        print('1.  영어사전 다운로드')
        print('2.  스팸메일 판별')
        print('3.  ')
        print('4.  ')
        menu = input('메뉴 선택\n')
        return int(menu)

    def run(self):
        model = self.model
        while 1:
            menu = self.print_menu()
            if menu == 0:
                break
            elif menu == 1 :
                model.down_eng_dictionary()
            elif menu == 2:
                emails_test = [
                    '''Subject: flat screens
                    hello ,
                    please call or contact regarding the other flat screens requested .
                    trisha tlapek - eb 3132 b
                    michael sergeev - eb 3132 a
                    also the sun blocker that was taken away from eb 3131 a .
                    trisha should two monitors also michael .
                    thanks
                    kevin moore''',
                    '''Subject: having problems in bed ? we can help !
                    cialis allows men to enjoy a fully normal sex life without having to plan the sexual act .
                    if we let things terrify us , life will not be worth living .
                    brevity is the soul of lingerie .
                    suspicion always haunts the guilty mind .''',
                ]
                result = model.email_test(emails_test)
                self.show(result)

    @staticmethod
    def show(param):
        print('RESULT : %s' % param)