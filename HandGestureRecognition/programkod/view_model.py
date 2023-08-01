class MainViewModel:
    def __init__(self, recognition_module, presentation):
        self.gesture_recognition = recognition_module
        self.presentation = presentation
        self.frame = self.gesture_recognition.frame_to_pass
        self.command = self.gesture_recognition.command
        self.sequence = self.gesture_recognition.gesture_seq
        self.mirrored = False
        self.de_shadow = False
        self.error_text = "We are sorry, en error occurred. Please open the presentation again!"
        self.missing_error = "The model files are missing!"
        self.started = False
        self.loaded = False

    def recognize_gesture(self):
        self.gesture_recognition.evaluate_gesture(self.mirrored, self.de_shadow)

    def execute_command(self):
        if self.sequence == 5:
            self.presentation.execute_command(self.command)
            self.sequence = 0

    def init_models(self):
        self.loaded = self.gesture_recognition.init_models("test_15.h5", "model_743100.h5")
        return self.loaded

    def start_presentation(self, path):
        if self.loaded and not self.started:
            self.gesture_recognition.subscribe_for_output_change(self.update_outputs)
            self.presentation.open_presentation(path)
            self.started = True

    def close_presentation(self):
        self.gesture_recognition.unsubscribe_from_output_change()
        self.presentation.close_presentation()
        self.started = False

    def close_processes(self):
        self.gesture_recognition.unsubscribe_from_output_change()
        self.presentation.close_presentation()
        self.presentation.close_application()

    def update_outputs(self):
        self.frame = self.gesture_recognition.frame_to_pass
        self.command = self.gesture_recognition.command
        self.sequence = self.gesture_recognition.gesture_seq
