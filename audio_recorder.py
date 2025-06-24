#!/usr/bin/env python3
"""
Programme modulaire d'enregistrement audio avec détection automatique de silence
Auteur: Assistant Claude
Version: 1.0
"""

import pyaudio
import wave
import numpy as np
import threading
import time
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import deque


class MicrophoneSelector:
    """Classe pour détecter et gérer les microphones disponibles"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.microphones = []
        self._detect_microphones()
    
    def _detect_microphones(self):
        """Détecte tous les microphones disponibles"""
        self.microphones = []
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                self.microphones.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxInputChannels'),
                    'sample_rate': device_info.get('defaultSampleRate')
                })
    
    def get_microphones(self):
        """Retourne la liste des microphones disponibles"""
        return self.microphones
    
    def get_default_microphone(self):
        """Retourne le microphone par défaut"""
        if self.microphones:
            return self.microphones[0]
        return None
    
    def cleanup(self):
        """Nettoie les ressources PyAudio"""
        self.audio.terminate()


class SilenceDetector:
    """Classe pour détecter le silence dans l'audio"""
    
    def __init__(self, silence_threshold=1000, silence_duration=2.0, sample_rate=44100):
        self.silence_threshold = silence_threshold  # Seuil d'amplitude pour détecter le silence
        self.silence_duration = silence_duration    # Durée de silence avant arrêt (secondes)
        self.sample_rate = sample_rate
        self.silence_frames = int(silence_duration * sample_rate / 1024)  # Nombre de frames de silence
        self.recent_volumes = deque(maxlen=self.silence_frames)
        self.is_recording_started = False
        self.speech_detected = False
    
    def _calculate_volume(self, audio_data):
        """Calcule le volume RMS de manière sécurisée"""
        try:
            # Convertir les données audio en numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Vérifier que le array n'est pas vide
            if len(audio_array) == 0:
                return 0.0
            
            # Calculer le volume (RMS) avec protection contre les valeurs invalides
            mean_square = np.mean(audio_array.astype(np.float64) ** 2)
            
            # Vérifier que la valeur est valide
            if np.isnan(mean_square) or np.isinf(mean_square) or mean_square < 0:
                return 0.0
            
            volume = np.sqrt(mean_square)
            
            # Vérifier le résultat final
            if np.isnan(volume) or np.isinf(volume):
                return 0.0
                
            return float(volume)
            
        except Exception as e:
            print(f"Erreur dans le calcul du volume: {e}")
            return 0.0
    
    def process_audio_chunk(self, audio_data):
        """
        Analyse un chunk audio et détermine s'il y a du silence
        Retourne True si l'enregistrement doit continuer, False pour arrêter
        """
        # Calculer le volume de manière sécurisée
        volume = self._calculate_volume(audio_data)
        self.recent_volumes.append(volume)
        
        # Détecter si on commence à parler
        if volume > self.silence_threshold:
            self.speech_detected = True
            self.is_recording_started = True
        
        # Si on n'a pas encore commencé à parler, continuer l'enregistrement
        if not self.speech_detected:
            return True
        
        # Vérifier si toutes les frames récentes sont en dessous du seuil
        if len(self.recent_volumes) >= self.silence_frames:
            if all(vol < self.silence_threshold for vol in self.recent_volumes):
                return False  # Arrêter l'enregistrement
        
        return True  # Continuer l'enregistrement
    
    def reset(self):
        """Remet à zéro le détecteur pour un nouvel enregistrement"""
        self.recent_volumes.clear()
        self.is_recording_started = False
        self.speech_detected = False


class AudioRecorder:
    """Classe principale pour l'enregistrement audio"""
    
    def __init__(self):
        self.chunk_size = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 44100
        self.microphone_index = None
        
        self.is_recording = False
        self.audio_data = []
        self.recording_thread = None
        
        # Composants
        self.mic_selector = MicrophoneSelector()
        self.silence_detector = SilenceDetector(
            silence_threshold=1000, 
            silence_duration=2.0,
            sample_rate=self.sample_rate
        )
        
        # Callbacks
        self.on_recording_start = None
        self.on_recording_stop = None
        self.on_volume_update = None
    
    def set_microphone(self, mic_index):
        """Définit le microphone à utiliser"""
        self.microphone_index = mic_index
    
    def set_silence_settings(self, threshold, duration):
        """Configure les paramètres de détection de silence"""
        self.silence_detector.silence_threshold = threshold
        self.silence_detector.silence_duration = duration
        self.silence_detector.silence_frames = int(duration * self.sample_rate / self.chunk_size)
        self.silence_detector.recent_volumes = deque(maxlen=self.silence_detector.silence_frames)
    
    def start_recording(self):
        """Démarre l'enregistrement audio"""
        if self.is_recording:
            return False
        
        if self.microphone_index is None:
            return False
        
        self.is_recording = True
        self.audio_data = []
        self.silence_detector.reset()
        
        # Lancer l'enregistrement dans un thread séparé
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
        if self.on_recording_start:
            self.on_recording_start()
        
        return True
    
    def stop_recording(self):
        """Arrête l'enregistrement audio"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
    
    def _record_audio(self):
        """Fonction d'enregistrement exécutée dans un thread"""
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=self.sample_format,
                channels=self.channels,
                rate=self.sample_rate,
                frames_per_buffer=self.chunk_size,
                input=True,
                input_device_index=self.microphone_index
            )
            
            while self.is_recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_data.append(data)
                
                # Détecter le silence
                should_continue = self.silence_detector.process_audio_chunk(data)
                
                # Notifier le volume pour l'interface (calcul sécurisé)
                if self.on_volume_update:
                    volume = self.silence_detector._calculate_volume(data)
                    self.on_volume_update(volume)
                
                if not should_continue:
                    self.is_recording = False
                    break
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Erreur d'enregistrement: {e}")
        finally:
            audio.terminate()
            if self.on_recording_stop:
                self.on_recording_stop()
    
    def save_recording(self, filename):
        """Sauvegarde l'enregistrement dans un fichier WAV"""
        if not self.audio_data:
            return False
        
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(pyaudio.get_sample_size(self.sample_format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(self.audio_data))
            return True
        except Exception as e:
            print(f"Erreur de sauvegarde: {e}")
            return False
    
    def get_recording_duration(self):
        """Retourne la durée de l'enregistrement en secondes"""
        if not self.audio_data:
            return 0
        total_frames = len(self.audio_data) * self.chunk_size
        return total_frames / self.sample_rate
    
    def cleanup(self):
        """Nettoie les ressources"""
        self.stop_recording()
        self.mic_selector.cleanup()


class AudioRecorderGUI:
    """Interface graphique pour l'enregistreur audio"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enregistreur Audio - Détection Automatique")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        
        # Thème moderne
        self.root.configure(bg='#f0f0f0')
        
        # Enregistreur
        self.recorder = AudioRecorder()
        self.recorder.on_recording_start = self.on_recording_start
        self.recorder.on_recording_stop = self.on_recording_stop
        self.recorder.on_volume_update = self.on_volume_update
        
        # Variables d'état
        self.is_recording = False
        self.last_filename = None
        self.start_time = None
        
        self.setup_ui()
        self.setup_microphones()
        
        # Fermeture propre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Titre
        title_label = tk.Label(
            self.root, 
            text="🎤 Enregistreur Audio Intelligent",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        title_label.pack(pady=20)
        
        # Sélection du microphone
        mic_frame = tk.Frame(self.root, bg='#f0f0f0')
        mic_frame.pack(pady=10)
        
        tk.Label(mic_frame, text="Microphone:", font=("Arial", 10), bg='#f0f0f0').pack(side=tk.LEFT)
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(mic_frame, textvariable=self.mic_var, width=40, state='readonly')
        self.mic_combo.pack(side=tk.LEFT, padx=(10, 0))
        self.mic_combo.bind('<<ComboboxSelected>>', self.on_mic_selected)
        
        # Paramètres de silence
        settings_frame = tk.LabelFrame(self.root, text="Paramètres de détection", bg='#f0f0f0', font=("Arial", 10))
        settings_frame.pack(pady=15, padx=20, fill=tk.X)
        
        # Seuil de silence
        threshold_frame = tk.Frame(settings_frame, bg='#f0f0f0')
        threshold_frame.pack(fill=tk.X, pady=5)
        tk.Label(threshold_frame, text="Seuil de silence:", bg='#f0f0f0').pack(side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=1000)
        threshold_scale = tk.Scale(
            threshold_frame, 
            from_=100, to=5000, 
            orient=tk.HORIZONTAL, 
            variable=self.threshold_var,
            bg='#f0f0f0'
        )
        threshold_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        threshold_scale.bind("<Motion>", self.on_settings_change)
        
        # Durée de silence
        duration_frame = tk.Frame(settings_frame, bg='#f0f0f0')
        duration_frame.pack(fill=tk.X, pady=5)
        tk.Label(duration_frame, text="Durée de silence (s):", bg='#f0f0f0').pack(side=tk.LEFT)
        self.duration_var = tk.DoubleVar(value=2.0)
        duration_scale = tk.Scale(
            duration_frame, 
            from_=0.5, to=5.0, 
            resolution=0.1,
            orient=tk.HORIZONTAL, 
            variable=self.duration_var,
            bg='#f0f0f0'
        )
        duration_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        duration_scale.bind("<Motion>", self.on_settings_change)
        
        # Indicateur de volume
        volume_frame = tk.Frame(self.root, bg='#f0f0f0')
        volume_frame.pack(pady=10)
        
        tk.Label(volume_frame, text="Volume:", font=("Arial", 10), bg='#f0f0f0').pack()
        self.volume_progress = ttk.Progressbar(
            volume_frame, 
            mode='determinate', 
            length=300,
            style='Volume.Horizontal.TProgressbar'
        )
        self.volume_progress.pack(pady=5)
        
        # État de l'enregistrement
        self.status_label = tk.Label(
            self.root,
            text="Prêt à enregistrer",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#666'
        )
        self.status_label.pack(pady=10)
        
        # Bouton principal
        self.record_button = tk.Button(
            self.root,
            text="🔴 Commencer l'enregistrement",
            font=("Arial", 14, "bold"),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10,
            command=self.toggle_recording,
            relief=tk.RAISED,
            borderwidth=2
        )
        self.record_button.pack(pady=20)
        
        # Bouton de sauvegarde
        self.save_button = tk.Button(
            self.root,
            text="💾 Sauvegarder",
            font=("Arial", 10),
            bg='#2196F3',
            fg='white',
            padx=15,
            pady=5,
            command=self.save_recording,
            state=tk.DISABLED
        )
        self.save_button.pack(pady=5)
        
        # Timer
        self.timer_label = tk.Label(
            self.root,
            text="00:00",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#333'
        )
        self.timer_label.pack(pady=5)
        
        # Configurer le style de la progress bar
        style = ttk.Style()
        style.configure('Volume.Horizontal.TProgressbar', background='#4CAF50')
    
    def setup_microphones(self):
        """Configure la liste des microphones"""
        microphones = self.recorder.mic_selector.get_microphones()
        mic_names = [f"{mic['name']} (Index: {mic['index']})" for mic in microphones]
        
        self.mic_combo['values'] = mic_names
        if microphones:
            self.mic_combo.current(0)
            self.recorder.set_microphone(microphones[0]['index'])
    
    def on_mic_selected(self, event):
        """Callback pour la sélection du microphone"""
        selection = self.mic_combo.current()
        if selection >= 0:
            microphones = self.recorder.mic_selector.get_microphones()
            self.recorder.set_microphone(microphones[selection]['index'])
    
    def on_settings_change(self, event):
        """Callback pour le changement des paramètres"""
        self.recorder.set_silence_settings(
            self.threshold_var.get(),
            self.duration_var.get()
        )
    
    def toggle_recording(self):
        """Bascule entre démarrer et arrêter l'enregistrement"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Démarre l'enregistrement"""
        if self.recorder.start_recording():
            self.is_recording = True
            self.start_time = time.time()
            # Réinitialiser le timer
            self.timer_label.config(text="00:00")
            self.update_timer()
    
    def stop_recording(self):
        """Arrête l'enregistrement"""
        self.recorder.stop_recording()
        self.is_recording = False
    
    def on_recording_start(self):
        """Callback appelé quand l'enregistrement commence"""
        self.record_button.config(
            text="⏹️ Arrêter l'enregistrement",
            bg='#f44336'
        )
        self.status_label.config(
            text="🔴 Enregistrement en cours... Parlez maintenant!",
            fg='#f44336'
        )
        self.save_button.config(state=tk.DISABLED)
        # Réinitialiser l'indicateur de volume
        self.volume_progress['value'] = 0
    
    def on_recording_stop(self):
        """Callback appelé quand l'enregistrement s'arrête"""
        self.is_recording = False  # S'assurer que l'état est mis à jour
        
        self.record_button.config(
            text="🔴 Commencer l'enregistrement",
            bg='#4CAF50'
        )
        
        duration = self.recorder.get_recording_duration()
        self.status_label.config(
            text=f"✅ Enregistrement terminé ({duration:.1f}s)",
            fg='#4CAF50'
        )
        self.save_button.config(state=tk.NORMAL)
        self.volume_progress['value'] = 0
        
        # Mettre à jour le timer avec la durée finale
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
    
    def on_volume_update(self, volume):
        """Callback pour mettre à jour l'indicateur de volume"""
        # Normaliser le volume pour la progress bar (0-100) avec protection
        try:
            if volume is not None and not np.isnan(volume) and not np.isinf(volume):
                normalized_volume = min(100, max(0, (volume / 3000) * 100))
                self.volume_progress['value'] = normalized_volume
            else:
                self.volume_progress['value'] = 0
        except Exception:
            self.volume_progress['value'] = 0
        
        self.root.update_idletasks()
    
    def update_timer(self):
        """Met à jour le timer d'enregistrement"""
        if self.is_recording and self.start_time:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_timer)
        elif not self.is_recording and self.recorder.audio_data:
            # Afficher la durée finale de l'enregistrement
            duration = self.recorder.get_recording_duration()
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
        else:
            # Reset du timer si pas d'enregistrement
            self.timer_label.config(text="00:00")
    
    def save_recording(self):
        """Sauvegarde l'enregistrement"""
        if not self.recorder.audio_data:
            messagebox.showwarning("Avertissement", "Aucun enregistrement à sauvegarder!")
            return
        
        # Générer un nom de fichier par défaut
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"enregistrement_{timestamp}.wav"
        
        # Boîte de dialogue de sauvegarde
        filename = filedialog.asksaveasfilename(
            title="Sauvegarder l'enregistrement",
            defaultextension=".wav",
            filetypes=[("Fichiers WAV", "*.wav"), ("Tous les fichiers", "*.*")],
            initialfile=default_filename
        )
        
        if filename:
            if self.recorder.save_recording(filename):
                self.last_filename = filename
                messagebox.showinfo("Succès", f"Enregistrement sauvegardé:\n{filename}")
                self.status_label.config(
                    text=f"💾 Sauvegardé: {os.path.basename(filename)}",
                    fg='#4CAF50'
                )
            else:
                messagebox.showerror("Erreur", "Impossible de sauvegarder l'enregistrement!")
    
    def on_closing(self):
        """Callback pour la fermeture de l'application"""
        if self.is_recording:
            if messagebox.askokcancel("Fermeture", "Un enregistrement est en cours. Voulez-vous vraiment quitter?"):
                self.recorder.cleanup()
                self.root.destroy()
        else:
            self.recorder.cleanup()
            self.root.destroy()
    
    def run(self):
        """Lance l'application"""
        self.root.mainloop()


def main():
    """Fonction principale"""
    try:
        app = AudioRecorderGUI()
        app.run()
    except Exception as e:
        print(f"Erreur lors du lancement de l'application: {e}")
        messagebox.showerror("Erreur", f"Impossible de lancer l'application:\n{e}")


if __name__ == "__main__":
    # Vérifier les dépendances
    try:
        import pyaudio
        import numpy as np
        print("✅ Toutes les dépendances sont installées")
        main()
    except ImportError as e:
        print("❌ Dépendances manquantes:")
        print("Installez les avec: pip install pyaudio numpy")
        print(f"Erreur: {e}")