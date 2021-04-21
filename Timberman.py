import Main_Header

if __name__ == '__main__':
    '''pyqt5 실행'''
    app = Main_Header.QtWidgets.QApplication(Main_Header.sys.argv)
    MainWindow = Main_Header.QtWidgets.QMainWindow()
    ui = Main_Header.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    Main_Header.sys.exit(app.exec_())