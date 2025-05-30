import wx
import wx.grid as gridlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial.distance import euclidean
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import threading
import time

DEFAULT_BINARY_THRESHOLD = 127
DEFAULT_MIN_AREA = 50
MAXIMUM_AREA = 500

# --- Common Functions Area ---
# OpenCV findContours return format processing function
# Handle different versions of findContours return value structure
def grab_contours(cnts):
    if len(cnts) == 2:
        return cnts[0]
    elif len(cnts) == 3:
        return cnts[1]
    raise Exception("Contours tuple must have length 2 or 3.")


class ContourInfoPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)

        self.grid = gridlib.Grid(self)
        self.grid.CreateGrid(0, 5)

        self.labels = ["Width (mm)", "Height (mm)", "Area (pixels)",
                       "Bounding Rect Area", "Rect (W x H)"]
        for idx, label in enumerate(self.labels):
            self.grid.SetColLabelValue(idx, label)

        self.grid.SetRowLabelSize(0)  # Hide row label
        #self.grid.SetMargins(0, 0)  # Remove top and bottom margins

        # Wrap Grid with sizer and enable expansion
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.grid, 1, flag=wx.EXPAND)
        self.SetSizer(sizer)

        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.grid.Bind(wx.EVT_SIZE, self.on_resize)

    def on_resize(self, event):
        size = self.grid.GetClientSize()
        num_cols = self.grid.GetNumberCols()
        if num_cols > 0 and size.width > 0:
            padding = 4  # Conservative estimate for column spacing
            col_width = max((size.width - padding * num_cols) // num_cols, 5)
            for i in range(num_cols):
                self.grid.SetColSize(i, col_width)
        self.grid.ForceRefresh()
        if event:
            event.Skip()

    def update_contours(self, contours_info):
        self.grid.ClearGrid()
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())

        self.grid.AppendRows(len(contours_info))
        for row, contour in enumerate(contours_info):
            self.grid.SetCellValue(row, 0, f"{contour['width_mm']:.1f}")
            self.grid.SetCellValue(row, 1, f"{contour['height_mm']:.1f}")
            self.grid.SetCellValue(row, 2, str(contour['area']))
            self.grid.SetCellValue(row, 3, str(contour['bounding_rect_area']))
            self.grid.SetCellValue(row, 4, str(contour['bounding_rect_size']))

        self.Layout()
        self.on_resize(None)  # Manually trigger column width adjustment

# --- Add Preprocessing Step Dialog ---
class AddStepDialog(wx.Dialog):
    def __init__(self, parent, title="Add Preprocessing Step"):
        super().__init__(parent, title=title, size=(400, 300))
        self.selected_step = None  # User selected step
        self.parameters = {}       # Parameters for the selected step

        # Default parameters for common steps
        self.default_parameters = {
            'Gaussian Blur': {'Kernel Size': '5'},
            'Binary Threshold': {'Threshold': '127'},
            'Morphological Operations': {'Kernel Size': '5'},
            'Canny Edge Detection': {'Threshold': '50'}
        }

        vbox = wx.BoxSizer(wx.VERTICAL)
        steps = ['Gray Conversion', 'Gaussian Blur', 'Binary Threshold', 'Morphological Operations', 'Canny Edge Detection']

        # Create dropdown menu for step selection
        self.step_choice = wx.Choice(self, choices=steps)
        self.step_choice.Bind(wx.EVT_CHOICE, self.on_step_selected)
        vbox.Add(self.step_choice, flag=wx.EXPAND | wx.ALL, border=10)

        # Panel for parameter input fields
        self.param_panel = wx.Panel(self)
        self.param_sizer = wx.BoxSizer(wx.VERTICAL)
        self.param_panel.SetSizer(self.param_sizer)
        vbox.Add(self.param_panel, flag=wx.EXPAND | wx.ALL, border=10)

        # Bottom buttons: Add and Cancel
        hbox_buttons = wx.BoxSizer(wx.HORIZONTAL)
        add_button = wx.Button(self, label="Add")
        add_button.Bind(wx.EVT_BUTTON, self.on_add)
        hbox_buttons.Add(add_button, flag=wx.RIGHT, border=10)

        cancel_button = wx.Button(self, label="Cancel")
        cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)
        hbox_buttons.Add(cancel_button)

        vbox.Add(hbox_buttons, flag=wx.ALIGN_CENTER | wx.TOP, border=10)
        self.SetSizer(vbox)

    # Dynamically generate parameter fields when step is selected
    def on_step_selected(self, event):
        step = self.step_choice.GetString(self.step_choice.GetSelection())
        self.param_sizer.Clear(True)
        self.parameters = {}

        if step in self.default_parameters:
            for param_name, default_value in self.default_parameters[step].items():
                self.add_param_textbox(param_name, default_value)

        self.param_panel.Layout()
        self.Layout()

    # Add a parameter text input box
    def add_param_textbox(self, param_name, default_value):
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        label = wx.StaticText(self.param_panel, label=param_name)
        textbox = wx.TextCtrl(self.param_panel, value=default_value)
        hbox.Add(label, flag=wx.RIGHT, border=8)
        hbox.Add(textbox, proportion=1)
        self.param_sizer.Add(hbox, flag=wx.EXPAND | wx.ALL, border=5)
        self.parameters[param_name] = textbox

    # Save selection when user clicks "Add"
    def on_add(self, event):
        self.selected_step = self.step_choice.GetString(self.step_choice.GetSelection())
        if self.selected_step:
            self.EndModal(wx.ID_OK)
        else:
            wx.MessageBox('Please select a step.', 'Error', wx.OK | wx.ICON_ERROR)

    # Close window when user clicks "Cancel"
    def on_cancel(self, event):
        self.EndModal(wx.ID_CANCEL)

    # Get current parameter settings (dictionary format)
    def get_parameters(self):
        return {name: textbox.GetValue() for name, textbox in self.parameters.items()}

# --- Main Window Class ---
class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MyFrame, self).__init__(parent, title=title, size=(1920, 1080))

        # Initialize variables
        self.pixel_to_mm_ratio = None  # Pixel to millimeter conversion ratio
        self.selected_contour = None   # User selected contour
        self.image = None              # Original image
        self.capture = None           # Camera object
        self.streaming = False        # Whether streaming is active
        self.stream_thread = None     # Streaming thread
        self.binary_threshold = DEFAULT_BINARY_THRESHOLD
        self.min_area = DEFAULT_MIN_AREA

        # Create GUI main panel and layout containers
        self.panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.HORIZONTAL)  # Outer horizontal layout container

        control_panel = wx.BoxSizer(wx.VERTICAL)  # Left side control components vertical layout

        # --- Create button row ---
        hbox_buttons = wx.BoxSizer(wx.HORIZONTAL)

        # Start camera button
        self.start_button = wx.Button(self.panel, label='Start Webcam')
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start_webcam)
        hbox_buttons.Add(self.start_button, flag=wx.EXPAND | wx.ALL, border=10)

        # Freeze frame button
        self.show_frame_button = wx.Button(self.panel, label='Freeze Frame for Selection')
        self.show_frame_button.Bind(wx.EVT_BUTTON, self.on_freeze_frame)
        hbox_buttons.Add(self.show_frame_button, flag=wx.EXPAND | wx.ALL, border=10)

        # Set reference width button
        self.set_ref_button = wx.Button(self.panel, label='Set Reference Width (mm)')
        self.set_ref_button.Bind(wx.EVT_BUTTON, self.on_set_reference_width)
        hbox_buttons.Add(self.set_ref_button, flag=wx.EXPAND | wx.ALL, border=10)

        # Apply preprocessing and size marking button
        self.process_mm_button = wx.Button(self.panel, label='Apply Processing with Size Info')
        self.process_mm_button.Bind(wx.EVT_BUTTON, self.on_apply_processing_mm)
        hbox_buttons.Add(self.process_mm_button, flag=wx.EXPAND | wx.ALL, border=10)

        # Live measurement button
        self.live_measure_button = wx.Button(self.panel, label='Live Measurement')
        self.live_measure_button.Bind(wx.EVT_BUTTON, self.on_live_measurement)
        hbox_buttons.Add(self.live_measure_button, flag=wx.EXPAND | wx.ALL, border=10)

        control_panel.Add(hbox_buttons, flag=wx.EXPAND | wx.ALL, border=10)

        # --- Create parameter sliders row ---
        hbox_sliders = wx.BoxSizer(wx.HORIZONTAL)

        # Binary threshold slider
        self.binary_label = wx.StaticText(self.panel, label='Binary Threshold: 127')
        hbox_sliders.Add(self.binary_label, flag=wx.LEFT | wx.RIGHT, border=8)
        self.binary_slider = wx.Slider(self.panel, value=127, minValue=0, maxValue=255, style=wx.SL_HORIZONTAL)
        self.binary_slider.Bind(wx.EVT_SLIDER, self.on_slider_update)
        hbox_sliders.Add(self.binary_slider, proportion=1)

        # Minimum contour area slider
        self.area_label = wx.StaticText(self.panel, label=f'Minimum Area: {self.min_area}')
        hbox_sliders.Add(self.area_label, flag=wx.LEFT | wx.RIGHT, border=8)
        self.area_slider = wx.Slider(self.panel, value=self.min_area, minValue=0, maxValue=MAXIMUM_AREA, style=wx.SL_HORIZONTAL)
        self.area_slider.Bind(wx.EVT_SLIDER, self.on_slider_update)
        hbox_sliders.Add(self.area_slider, proportion=1)

        control_panel.Add(hbox_sliders, flag=wx.EXPAND | wx.ALL, border=10)

        # --- Create preprocessing steps and operations ---
        self.preprocess_steps = ['Gray Conversion', 'Gaussian Blur', 'Binary Threshold', 'Morphological Operations', 'Canny Edge Detection']
        self.preprocess_parameters = {
            'Gaussian Blur': {'Kernel Size': '5'},
            'Binary Threshold': {'Threshold': str(DEFAULT_BINARY_THRESHOLD)},
            'Morphological Operations': {'Kernel Size': '5'},
            'Canny Edge Detection': {'Threshold': '50'}
        }

        # Steps list display component
        self.preprocess_listbox = wx.ListBox(self.panel, choices=self.get_preprocess_display(), style=wx.LB_SINGLE)
        control_panel.Add(self.preprocess_listbox, flag=wx.EXPAND | wx.ALL, border=10)

        # Preprocessing step adjustment area: add, remove, move up, move down
        hbox_add_remove = wx.BoxSizer(wx.HORIZONTAL)
        self.add_step_button = wx.Button(self.panel, label='Add Step')
        self.add_step_button.Bind(wx.EVT_BUTTON, self.on_add_step)
        hbox_add_remove.Add(self.add_step_button, flag=wx.EXPAND | wx.ALL, border=10)

        self.remove_step_button = wx.Button(self.panel, label='Remove Step')
        self.remove_step_button.Bind(wx.EVT_BUTTON, self.on_remove_step)
        hbox_add_remove.Add(self.remove_step_button, flag=wx.EXPAND | wx.ALL, border=10)

        self.move_up_button = wx.Button(self.panel, label='Move Up')
        self.move_up_button.Bind(wx.EVT_BUTTON, self.on_move_up_step)
        hbox_add_remove.Add(self.move_up_button, flag=wx.EXPAND | wx.ALL, border=10)

        self.move_down_button = wx.Button(self.panel, label='Move Down')
        self.move_down_button.Bind(wx.EVT_BUTTON, self.on_move_down_step)
        hbox_add_remove.Add(self.move_down_button, flag=wx.EXPAND | wx.ALL, border=10)

        control_panel.Add(hbox_add_remove, flag=wx.EXPAND | wx.ALL, border=10)

        # Initialize contour info panel
        self.contour_info_panel = ContourInfoPanel(self.panel)
        control_panel.Add(self.contour_info_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        # Add control panel to main horizontal layout (left side)
        vbox.Add(control_panel, flag=wx.EXPAND | wx.ALL, border=10)

        # --- Create image display canvas (right side) ---
        self.figure = Figure()
        self.canvas = FigureCanvas(self.panel, -1, self.figure)
        vbox.Add(self.canvas, 1, flag=wx.EXPAND | wx.ALL, border=10)

        # Allow user to select contours by clicking on canvas
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        self.panel.SetSizer(vbox)

        # --- Record frozen frame and contour list ---
        self.freeze_contours = []
        self.frozen_frame = None

    def update_contour_info(self, contours):
        # Get information for each contour
        contours_info = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            rect_area = w * h
            width_mm = w * self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else w
            height_mm = h * self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else h
            contours_info.append({
                'width_mm': width_mm,
                'height_mm': height_mm,
                'area': int(area),
                'bounding_rect_area': int(rect_area),
                'bounding_rect_size': f'{w} x {h}'
            })

        # Update panel content
        self.contour_info_panel.update_contours(contours_info)

    def get_preprocess_display(self):
        return [f'{step} ({", ".join([f"{k}: {v}" for k, v in self.preprocess_parameters.get(step, {}).items()])})'
                for step in self.preprocess_steps]

    # --- Start camera stream ---
    def on_start_webcam(self, event):
        if self.streaming:
            wx.MessageBox('Please stop other streaming functions (e.g., Live Measurement) first', 'Error', wx.OK | wx.ICON_ERROR)
            return

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            wx.MessageBox('Cannot open camera', 'Error', wx.OK | wx.ICON_ERROR)
            return

        self.streaming = True

        def webcam_loop():
            while self.streaming:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                self.image = frame
                processed = self.preprocess_image(frame)[-1]  # Pass original BGR frame
                contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                display = frame.copy()
                for contour in contours:
                    if cv2.contourArea(contour) < self.min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display, f"{w} x {h} px", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                ax.axis('off')
                self.canvas.draw()
                time.sleep(0.03)

        self.stream_thread = threading.Thread(target=webcam_loop, daemon=True)
        self.stream_thread.start()

    # --- Stop camera stream ---
    def stop_webcam(self, event=None):
        if self.streaming:
            self.streaming = False
            if self.capture:
                self.capture.release()

    # --- Freeze camera frame and capture contours ---
    def on_freeze_frame(self, event):
        if self.capture is None or not self.capture.isOpened():
            wx.MessageBox('Camera not started', 'Error', wx.OK | wx.ICON_ERROR)
            return

        self.streaming = False  # Stop webcam thread
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)

        ret, frame = self.capture.read()
        if not ret:
            wx.MessageBox('Cannot get image from camera', 'Error', wx.OK | wx.ICON_ERROR)
            return

        self.image = frame.copy()
        processed = self.preprocess_image(frame)[-1]  # Pass original BGR frame
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{w} x {h} px", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        self.canvas.draw()

        # Save frozen frame and contours
        self.freeze_contours = contours
        self.frozen_frame = frame.copy()

        self.update_contour_info(contours)

    def on_canvas_click(self, event):
        if event.xdata is None or event.ydata is None or not hasattr(self, 'freeze_contours'):
            return

        click_x, click_y = int(event.xdata), int(event.ydata)
        self.selected_contour = None

        # Find contour containing click coordinates (using bounding box)
        for contour in self.freeze_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x <= click_x <= x + w and y <= click_y <= y + h:
                self.selected_contour = contour
                break

        # If selected, draw red box
        if self.selected_contour is not None:
            selected_image = self.frozen_frame.copy()
            x, y, w, h = cv2.boundingRect(self.selected_contour)
            cv2.rectangle(selected_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            self.show_full_image(selected_image)

    # --- Display single image on canvas ---
    def show_full_image(self, image):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        self.canvas.draw()

    # --- Set pixel to millimeter conversion ratio ---
    def on_set_reference_width(self, event):
        if self.selected_contour is None:
            wx.MessageBox('Please select a contour as reference first', 'Prompt', wx.OK | wx.ICON_INFORMATION)
            return

        x, y, w, h = cv2.boundingRect(self.selected_contour)
        dlg = wx.TextEntryDialog(self, f'Contour width is {w} pixels\nPlease enter actual width (mm):', 'Set Reference Width')
        if dlg.ShowModal() == wx.ID_OK:
            try:
                real_width_mm = float(dlg.GetValue())
                self.pixel_to_mm_ratio = real_width_mm / w
                wx.MessageBox(f'Setting successful: 1 px = {self.pixel_to_mm_ratio:.3f} mm', 'Success',
                              wx.OK | wx.ICON_INFORMATION)
            except:
                wx.MessageBox('Input format error, please enter a number', 'Error', wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    # --- Apply processing and display size in mm ---
    def on_apply_processing_mm(self, event):
        if self.image is None:
            wx.MessageBox('Please start camera first', 'Error', wx.OK | wx.ICON_ERROR)
            return

        if self.pixel_to_mm_ratio is None:
            wx.MessageBox('Please set pixel to mm conversion ratio first', 'Error', wx.OK | wx.ICON_ERROR)
            return

        processed_images = self.preprocess_image(self.image)
        final_processed_image = processed_images[-1]
        contours, _ = cv2.findContours(final_processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = self.image.copy()

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            width_mm = w * self.pixel_to_mm_ratio
            height_mm = h * self.pixel_to_mm_ratio
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"{width_mm:.1f}mm x {height_mm:.1f}mm", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        self.show_full_image(result)
        self.update_contour_info(contours)

    # --- Start real-time size measurement mode ---
    def on_live_measurement(self, event):
        if self.pixel_to_mm_ratio is None:
            wx.MessageBox('Please set pixel to mm conversion ratio first', 'Error', wx.OK | wx.ICON_ERROR)
            return

        if self.streaming:
            wx.MessageBox('Live Measurement is already active', 'Error', wx.OK | wx.ICON_ERROR)
            return

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            wx.MessageBox('Cannot open camera', 'Error', wx.OK | wx.ICON_ERROR)
            return

        self.streaming = True

        def live_loop():
            while self.streaming:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                self.image = frame
                processed = self.preprocess_image(frame)[-1]  # Pass original BGR frame
                contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                display = frame.copy()
                for contour in contours:
                    if cv2.contourArea(contour) < self.min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    width_mm = w * self.pixel_to_mm_ratio
                    height_mm = h * self.pixel_to_mm_ratio
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display, f"{width_mm:.1f}mm x {height_mm:.1f}mm", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                ax.axis('off')
                self.canvas.draw()
                time.sleep(0.03)

        self.stream_thread = threading.Thread(target=live_loop, daemon=True)
        self.stream_thread.start()

    # --- Execute preprocessing pipeline ---
    def preprocess_image(self, image):
        processed_images = []
        processed_images.append(image)
        processed_image = image
        for step in self.preprocess_steps:
            params = self.preprocess_parameters.get(step, {})
            if step == 'Gray Conversion':
                # Only convert if image has 3 channels
                if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            elif step == 'Gaussian Blur':
                ksize = int(params.get('Kernel Size', '5'))
                processed_image = cv2.GaussianBlur(processed_image, (ksize, ksize), 0)
            elif step == 'Binary Threshold':
                threshold = self.binary_threshold
                _, processed_image = cv2.threshold(processed_image, threshold, 255, cv2.THRESH_BINARY)
            elif step == 'Morphological Operations':
                ksize = int(params.get('Kernel Size', '5'))
                kernel = np.ones((ksize, ksize), np.uint8)
                processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
            elif step == 'Canny Edge Detection':
                threshold = int(params.get('Threshold', '50'))
                processed_image = cv2.Canny(processed_image, threshold, threshold * 3)
            processed_images.append(processed_image)
        return processed_images

    # --- Display multiple images after preprocessing ---
    def show_multiple_images(self, images):
        self.figure.clear()
        n = len(images)
        nrows = (n + 1) // 2
        for i, img in enumerate(images):
            ax = self.figure.add_subplot(nrows, 2, i + 1)
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis('off')
        self.canvas.draw()

    # --- Update parameters with slider and display effects in real-time ---
    def on_slider_update(self, event):
        self.binary_threshold = self.binary_slider.GetValue()
        self.binary_label.SetLabel(f'Binary Threshold: {self.binary_threshold}')
        self.min_area = self.area_slider.GetValue()
        self.area_label.SetLabel(f'Minimum Area: {self.min_area}')

    # --- Add preprocessing step ---
    def on_add_step(self, event):
        dialog = AddStepDialog(self)
        if dialog.ShowModal() == wx.ID_OK:
            new_step = dialog.selected_step
            if new_step and new_step not in self.preprocess_steps:
                self.preprocess_steps.append(new_step)
                self.preprocess_parameters[new_step] = dialog.get_parameters()
                self.update_preprocess_listbox()
        dialog.Destroy()

    # --- Remove preprocessing step ---
    def on_remove_step(self, event):
        selection = self.preprocess_listbox.GetSelection()
        if selection != wx.NOT_FOUND:
            step = self.preprocess_steps.pop(selection)
            if step in self.preprocess_parameters:
                del self.preprocess_parameters[step]
            self.update_preprocess_listbox()

    # --- Move preprocessing step up ---
    def on_move_up_step(self, event):
        selection = self.preprocess_listbox.GetSelection()
        if selection != wx.NOT_FOUND and selection > 0:
            self.preprocess_steps[selection], self.preprocess_steps[selection - 1] = (
                self.preprocess_steps[selection - 1], self.preprocess_steps[selection])
            self.update_preprocess_listbox()
            self.preprocess_listbox.SetSelection(selection - 1)

    # --- Move preprocessing step down ---
    def on_move_down_step(self, event):
        selection = self.preprocess_listbox.GetSelection()
        if selection != wx.NOT_FOUND and selection < len(self.preprocess_steps) - 1:
            self.preprocess_steps[selection], self.preprocess_steps[selection + 1] = (
                self.preprocess_steps[selection + 1], self.preprocess_steps[selection])
            self.update_preprocess_listbox()
            self.preprocess_listbox.SetSelection(selection + 1)

    # --- Update list display and synchronize parameters ---
    def update_preprocess_listbox(self):
        self.preprocess_listbox.Clear()
        self.preprocess_listbox.AppendItems([
            f'{step} ({", ".join([f"{k}: {v}" for k, v in self.preprocess_parameters.get(step, {}).items()])})'
            for step in self.preprocess_steps
        ])

# --- Main Program Entry Point ---
class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame(None, title='Object Detection and Size Measurement System')
        self.SetTopWindow(frame)
        frame.Show()
        return True

    def OnExit(self):
        if hasattr(self, 'frame') and self.frame.streaming:
            self.frame.stop_webcam()
        return 0

if __name__ == '__main__':
    app = MyApp()
    app.MainLoop()