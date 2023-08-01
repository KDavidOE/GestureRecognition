import gesture_recognition
import view
import view_model
import presentation


class Application:
    def __init__(self):
        self.pres_obj = presentation.Presentation()
        self.gesture_recognition = gesture_recognition.GestureRecognition()
        self.main_vm = view_model.MainViewModel(self.gesture_recognition, self.pres_obj)
        self.main_view = view.MainView(self.main_vm)

    def start_application(self):
        self.main_view.init_view()


if __name__ == '__main__':
    Application().start_application()

