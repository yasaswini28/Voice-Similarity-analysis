import os
import tkinter as tk
from tkinter import filedialog
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def compute_and_resize_spectrogram(audio_file, target_shape):
    y, sr = librosa.load(audio_file)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    resized_spectrogram = np.zeros(target_shape)
    resized_spectrogram[:D.shape[0], :D.shape[1]] = D
    return resized_spectrogram
def compare_spectrograms(spectrogram1, spectrogram2):
    flat_spectrogram1 = spectrogram1.flatten()
    flat_spectrogram2 = spectrogram2.flatten()
    similarity = cosine_similarity([flat_spectrogram1], [flat_spectrogram2])[0][0]
    return similarity
def plot_spectrogram(audio_file, ax):
    y, sr = librosa.load(audio_file)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title('Spectrogram')
def handle_upload_button_click(entries, analyze_button):
    audio_files = [entry.get() for entry in entries]
    for audio_file in audio_files:
        file_path = filedialog.askopenfilename(title=f"Select {audio_file}", filetypes=[("Audio files", "*.wav;*.mp3;*.ogg")])
        if file_path:
            entry = next(entry for entry in entries if entry.get() == audio_file)
            entry.delete(0, tk.END)
            entry.insert(tk.END, file_path)
    analyze_button.config(state=tk.NORMAL)
def handle_analyze_button_click(entries, similarity_threshold):
    spectrograms = []
    for entry in entries:
        audio_file = entry.get()
        if os.path.exists(audio_file):
            spectrogram = compute_and_resize_spectrogram(audio_file, (1025, 1292))
            spectrograms.append((audio_file, spectrogram))
        else:
            print(f"File not found: {audio_file}")
    for i, (file, spec) in enumerate(spectrograms):
        root.after(i * 1000, lambda f=file, s=spec: perform_spectrogram_analysis(f, s))
    root.after(len(spectrograms) * 1000, lambda: calculate_and_display_similarities(spectrograms, similarity_threshold))
def perform_spectrogram_analysis(audio_file, spectrogram):
    fig, ax = plt.subplots()
    plot_spectrogram(audio_file, ax)
    ax.set_title('Spectrogram for ' + os.path.basename(audio_file))
    plt.show()
def calculate_and_display_similarities(spectrograms, similarity_threshold):
    print("\nSimilarities:")
    for i, (file1, spec1) in enumerate(spectrograms):
        for j, (file2, spec2) in enumerate(spectrograms):
            if i < j:
                similarity = compare_spectrograms(spec1, spec2)
                print(f"{os.path.basename(file1)} vs {os.path.basename(file2)}: {similarity:.4f}")

                if similarity >= similarity_threshold:
                    print("Voices are considered the same.")
                else:
                    print("Voices are considered different.")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Audio Spectrogram Analyzer")
    num_files = int(input("Enter the number of audio files: "))
    similarity_threshold = float(input("Enter the similarity threshold (e.g., 0.9): "))
    entries = []
    for i in range(num_files):
        entry = tk.Entry(root, width=50)
        entry.pack(pady=10)
        entries.append(entry)
    upload_button = tk.Button(root, text="Upload Audio Files",
                              command=lambda: handle_upload_button_click(entries, analyze_button))
    upload_button.pack(pady=10)
    analyze_button = tk.Button(root, text="Analyze", state=tk.DISABLED,
                               command=lambda: handle_analyze_button_click(entries, similarity_threshold))
    analyze_button.pack(pady=20)
    root.mainloop()