:

📡 Modulation Recognition using Deep Learning

This project implements a Convolutional Neural Network (CNN) for automatic modulation classification using both:

🧪 Synthetic dataset (generated I/Q signals)

📻 RadioML2016.10a dataset (real-world benchmark)

It also includes an interactive Streamlit app for visualization, predictions, and model explainability with Grad-CAM.

🚀 Features

✅ Train a CNN on Synthetic or RadioML datasets
✅ Interactive Streamlit app for uploading signals and predicting modulation type
✅ Visualization of:

Raw input signal

Spectrogram representation

Grad-CAM heatmap highlighting important regions
✅ Support for multiple modulation classes
✅ Clean UI with insights on predicted class

🛠️ Installation

Clone the repository:

git clone https://github.com/NupurrSharma/modulation-recognition.git
cd modulation-recognition


Create a virtual environment:

python3 -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows


Install dependencies:

pip install -r requirements.txt

📂 Project Structure
modulation-recognition/
│── app.py                 # Streamlit app
│── train_cnn.py           # Training script
│── grad_cam.py            # Grad-CAM implementation
│── radioml_loader.py      # RadioML dataset loader
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
│── dataset/               # (excluded from repo, place datasets here)

📊 Training

To train on synthetic dataset:

python train_cnn.py --dataset synthetic --epochs 50


To train on RadioML dataset:

python train_cnn.py --dataset radioml --epochs 50


Models will be saved as:

cnn_classifier_synthetic.pth

cnn_classifier_radioml.pth

🎮 Run the App

Launch Streamlit:

streamlit run app.py


Features available in the app:

Upload your own signal file (.npy)

Visualize raw signal & spectrogram

Predict modulation class with CNN

View Grad-CAM heatmaps for interpretability

📈 Example Outputs

Signal waveform

Spectrogram

Predicted modulation class

Grad-CAM heatmap overlay

⚠️ Notes

Datasets (.npy, .pkl) are excluded from GitHub due to size limits.

Place signals.npy, labels.npy, or RML2016.10a_dict.pkl in the dataset/ folder.

🤝 Contributing

Pull requests are welcome! Feel free to open an issue for discussions or improvements.