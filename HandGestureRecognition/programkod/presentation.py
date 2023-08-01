import win32api
import win32com.client
import win32con


class Presentation:
    def __init__(self):
        self.app = win32com.client.Dispatch("PowerPoint.Application")
        self.pres = None
        self.total_slides = None
        self.presentation_opened = False
        self.presentation_started = False

    def open_presentation(self, path):
        self.pres = self.app.Presentations.Open(FileName=path, WithWindow=1)
        self.presentation_opened = True
        self.total_slides = len(self.pres.slides)

    def close_presentation(self):
        if self.presentation_opened:
            self.pres.close()
            self.presentation_opened = False
            self.presentation_started = False

    def close_application(self):
        self.app.Quit()
        self.app = None
        self.pres = None

    def execute_command(self, command_code):
        if command_code == 'Stop':
            if self.presentation_started:
                self.presentation_started = False
                self.pres.SlideShowWindow.View.Exit()
        elif command_code == 'Start':
            if not self.presentation_started:
                self.pres.SlideShowSettings.Run()
                self.presentation_started = True
        elif command_code == 'Forward':
            if self.presentation_started and (self.pres.SlideShowWindow.View.Slide.SlideIndex != self.total_slides):
                self.pres.SlideShowWindow.View.Next()
        elif command_code == 'Back':
            if self.presentation_started:
                self.pres.SlideShowWindow.View.Previous()
        elif command_code == "VolumeUp":
            win32api.keybd_event(win32con.VK_VOLUME_UP, 0)
        elif command_code == "VolumeDown":
            win32api.keybd_event(win32con.VK_VOLUME_DOWN, 0)
