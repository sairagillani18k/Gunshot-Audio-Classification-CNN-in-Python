import sys
from PyQt5.QtGui import QIcon, QMovie
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QFont
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import librosa
import numpy as np
from PyQt5.QtCore import Qt

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set the window title and icon
        self.setWindowTitle("Audio Classification")
        self.setWindowIcon(QIcon('icon.ico'))

        # Create a welcome label and add it to the layout
        self.label = QLabel(self)
        self.movie = QMovie('background-gif.gif')
        self.label.setMovie(self.movie)
        self.movie.start()

        # Add the Recognition button
        self.recognition_button = QPushButton("Choose Audio File", self)
        font = QFont()
        font.setPointSize(16)
        self.recognition_button.setFont(font)

        # Add the Storing button
        self.storing_button = QPushButton("Predict", self)
        self.storing_button.setFont(font)

        # Add a QLabel for displaying the prediction result
        self.result_label = QLabel(self)
        font.setPointSize(18)  # Increase font size for better visibility
        self.result_label.setFont(font)

        # Set up the layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.recognition_button)
        button_layout.addWidget(self.storing_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label, 1)  # Make the gif cover the entire window
        main_layout.addStretch()  # Push buttons to the bottom
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

        self.model = load_model("best_model.h5")  # Replace with the path to your best model file
        self.le = LabelEncoder()
        self.le.classes_ = np.load("label_encoder_classes.npy")  # Replace with the path to your label encoder classes file

        # Connect button clicks to functions
        self.recognition_button.clicked.connect(self.choose_file)
        self.storing_button.clicked.connect(self.show_prediction_result)

    def choose_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Audio File", "", "Audio Files (*.wav *.mp3);;All Files (*)", options=options)
        if file_path:
            # Do something with the selected file path, e.g., print it
            print("Selected File:", file_path)
            self.selected_file = file_path

    def show_prediction_result(self):
        if hasattr(self, 'selected_file'):
            result = self.make_predictions(self.selected_file)
            self.result_label.setText(f"Prediction Result: {result}")
            self.result_label.setAlignment(Qt.AlignCenter)
        else:
            self.result_label.setText("Please choose an audio file first.")
            self.result_label.setAlignment(Qt.AlignCenter)

    def make_predictions(self, file_path):
        audio, sample_rate = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        features = mfccs_scaled.reshape(1, mfccs_scaled.shape[0], 1)
        predicted_vector = self.model.predict(features)
        predicted_class_index = np.argmax(predicted_vector, axis=-1)
        return self.le.inverse_transform(predicted_class_index)[0]

if __name__ == '__main__':
    # Create the application and main window
    app = QApplication(sys.argv)
    window = MainWindow()

    # Set the application style
    app.setStyle('Fusion')

    # Set the application icon and logo
    app.setWindowIcon(QIcon('logo.png'))

    # Show the main window and start the application loop
    window.show()
    sys.exit(app.exec_())
