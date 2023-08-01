import PySimpleGUI as sg


def show_popup_window(error_msg):
    sg.popup(error_msg, title="Hiba", modal=True)


class MainView:
    def __init__(self, vm):
        self.vm = vm

    image_viewer_column = [
        [sg.Column(justification="center", layout=[[sg.Image(key="-IMAGE-", size=(200,200))]])],
        [sg.HSeparator()],
        [sg.Text(size=(40, 1), text="Gesture: ", key="-COMMAND-")],
        [sg.Text(size=(40, 1), text="Sequence: ", key="-SEQUENCE-")],
    ]

    options_bar = [
        [
            sg.Text("Presentation file: "),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-", disabled=True),
            sg.FileBrowse(enable_events=True, key="-SELECTOR-", file_types=(("Pptx Files", "*.pptx"),)),
            sg.Button(enable_events=True,button_text="Start", key="-START-"),
            sg.Button(enable_events=True, button_text="Stop", key="-STOP-"),
            sg.Checkbox("Mirror", key='-MIRROR-'),
            sg.Checkbox("Shadow-removal", key='-DE_SHADOW-'),
        ],
    ]

    layout = [
        [
            options_bar,
        ],
        [
            sg.Column(justification="center", layout=image_viewer_column),
        ]
    ]

    def init_view(self):
        window = sg.Window("PPT Navigator", self.layout)
        while True:
            event, values = window.read(timeout=33)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            if event == "-SELECTOR-":
                window["-FOLDER-"].update(window["-SELECTOR-"].get_text())

            if event == "-START-" and (window["-FOLDER-"].get() != ""):
                if not self.vm.loaded:
                    if not self.vm.init_models():
                        show_popup_window(self.vm.missing_error)

                if not self.vm.started:
                    self.vm.start_presentation(window["-FOLDER-"].get())

            if event == "-STOP-":
                self.vm.close_presentation()

            if self.vm.frame is not None:
                window["-IMAGE-"].update(self.vm.frame)

            if self.vm.command is not None:
                window["-COMMAND-"].update("Gesture: " + str(self.vm.command))
                window["-SEQUENCE-"].update("Sequence: " + str(self.vm.sequence))
            else:
                window["-COMMAND-"].update("Gesture: ")
            if self.vm.started & self.vm.loaded:
                if window["-MIRROR-"].get():
                    self.vm.mirrored = True
                else:
                    self.vm.mirrored = False

                if window["-DE_SHADOW-"].get():
                    self.vm.de_shadow = True
                else:
                    self.vm.de_shadow = False

                try:
                    self.vm.recognize_gesture()
                    self.vm.execute_command()
                except Exception:
                    self.vm.close_presentation()
                    show_popup_window(self.vm.error_text)
                    self.vm.started = False
        try:
            self.vm.close_processes()
        except Exception as e:
            print(e)
        window.close()
