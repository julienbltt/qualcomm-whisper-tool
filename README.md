# qualcomm-whisper-tool
This tool can be used for audio transcription (speech-to-text) for implement into the Commpanion project.

---

## Installation

Follow this guide, [here](docs/qualcomm_ai_hub_onnx_installation_guide.md).

If you have a issue in part 6 in state 2, follow the [troubleshootinh guide](docs/qnn_troubleshooting_guide.md).
In part 8, installe whisper dependancies : `pip install "qai_hub_models[whisper_base_en]"`

## Description

Ce projet utilise les modèles OpenAI Whisper ASR (Automatic Speech Recognition) optimisés pour les appareils Qualcomm. Ces modèles excellent dans la transcription de clips audio jusqu'à 30 secondes et offrent des performances robustes dans des environnements bruyants réalistes.

### Modèles disponibles

- **whisper-tiny-en** : Modèle le plus compact pour l'anglais uniquement
- **whisper-base-en** : Modèle de base pour l'anglais (recommandé pour ce projet)
- **whisper-small-en** : Modèle small pour l'anglais 
- **whisper-small-v2** : Modèle multilingue optimisé

## Installation

### Prérequis système

- Python 3.8-3.11
- Compte Qualcomm AI Hub (requis pour la compilation et le profilage)
- Dispositifs compatibles : Snapdragon 8 Elite, 8 Gen 3, 8 Gen 2, 8 Gen 1

**Note importante pour les utilisateurs Snapdragon X Elite** : Seul Python AMDx64 (64-bit) est supporté sur Windows. L'installation échouera avec Python Windows ARM64.

### 1. Installation du package principal

```bash
# Installation de base
pip install qai_hub_models

# Installation avec les dépendances Whisper spécifiques
pip install "qai_hub_models[whisper_base_en]"
```

### 2. Configuration de Qualcomm AI Hub

Pour accéder aux fonctionnalités de compilation et de profilage, vous devez configurer votre accès à Qualcomm AI Hub :

1. **Créer un compte Qualcomm AI Hub** :
   - Rendez-vous sur [Qualcomm AI Hub](https://aihub.qualcomm.com)
   - Créez un Qualcomm ID et connectez-vous

2. **Obtenir votre token API** :
   - Naviguez vers Account → Settings → API Token
   - Copiez votre token API

3. **Configurer le client** :
```bash
qai-hub configure --api_token VOTRE_TOKEN_API
```

### 3. Guides d'installation détaillés

Pour une installation complète incluant ONNX et QNN :
- Suivez le [guide d'installation Qualcomm AI Hub ONNX](docs/qualcomm_ai_hub_onnx_installation_guide.md)
- En cas de problème lors de l'étape 6 état 2, consultez le [guide de dépannage QNN](docs/qnn_troubleshooting_guide.md)

## Utilisation

### Module whisper.py

Le module `whisper.py` fournit une interface simplifiée pour la transcription audio en utilisant les modèles Whisper optimisés Qualcomm.

#### Fonctionnement

Le module encapsule les fonctionnalités suivantes :
- **Prétraitement audio** : Conversion et normalisation des fichiers audio
- **Inférence modèle** : Utilisation des modèles Whisper optimisés Qualcomm
- **Post-traitement** : Extraction et formatage du texte transcrit
- **Gestion des dispositifs** : Optimisation automatique pour les puces Qualcomm

#### Importation et utilisation de base

```python
from qai_hub_models.models.whisper_base_en import Model
from qai_hub_models.models.whisper_base_en import App

# Initialisation du modèle
model = Model.from_pretrained()

# Utilisation avec l'application intégrée
app = App(model)

# Transcription d'un fichier audio
result = app.predict("chemin/vers/audio.wav")
print(result)
```

#### Démo interactive

```bash
# Démo basique avec fichier d'exemple
python -m qai_hub_models.models.whisper_base_en.demo

# Démo avec un fichier audio spécifique
python -m qai_hub_models.models.whisper_base_en.demo --audio-file votre_audio.wav

# Pour Jupyter Notebook ou Google Colab
%run -m qai_hub_models.models.whisper_base_en.demo
```

### Exemples d'utilisation avancée

#### 1. Transcription en local

```python
import torch
from qai_hub_models.models.whisper_base_en import Model, App

# Chargement du modèle
model = Model.from_pretrained()
app = App(model)

# Transcription d'un fichier audio
audio_path = "exemple.wav"
transcription = app.predict(audio_path)

print(f"Transcription : {transcription}")
```

#### 2. Transcription sur dispositif cloud Qualcomm

```python
import qai_hub as hub
from qai_hub_models.models.whisper_base_en import Model

# Configuration pour l'inférence cloud
model = Model.from_pretrained()

# Export vers un dispositif cloud
job = hub.submit_compile_job(
    model=model,
    device="Samsung Galaxy S24 (Snapdragon 8 Gen 3)"
)

# Attendre la compilation
job.wait()

# Utilisation du modèle compilé
compiled_model = job.get_target_model()
```

#### 3. Export pour déploiement mobile

```python
# Export du modèle pour Android
python -m qai_hub_models.models.whisper_base_en.export \
    --target-runtime tflite \
    --device "Samsung Galaxy S24 (Snapdragon 8 Gen 3)"

# Export pour QNN (format .so)
python -m qai_hub_models.models.whisper_base_en.export \
    --target-runtime qnn \
    --device "Samsung Galaxy S24 (Snapdragon 8 Gen 3)"
```

#### 4. Utilisation avec options avancées

```python
from qai_hub_models.models.whisper_base_en import Model, App
import librosa

class CustomWhisperApp:
    def __init__(self):
        self.model = Model.from_pretrained()
        self.app = App(self.model)
    
    def transcribe_with_preprocessing(self, audio_path, sample_rate=16000):
        # Prétraitement audio personnalisé
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Normalisation
        audio = audio / max(abs(audio))
        
        # Transcription
        result = self.app.predict(audio)
        
        return {
            'transcription': result,
            'sample_rate': sr,
            'duration': len(audio) / sr
        }
    
    def batch_transcribe(self, audio_files):
        results = []
        for file_path in audio_files:
            try:
                result = self.transcribe_with_preprocessing(file_path)
                results.append({
                    'file': file_path,
                    'success': True,
                    'data': result
                })
            except Exception as e:
                results.append({
                    'file': file_path,
                    'success': False,
                    'error': str(e)
                })
        return results

# Utilisation
whisper_app = CustomWhisperApp()

# Transcription simple
result = whisper_app.transcribe_with_preprocessing("audio.wav")
print(result['transcription'])

# Transcription en lot
files = ["audio1.wav", "audio2.wav", "audio3.wav"]
batch_results = whisper_app.batch_transcribe(files)
```

### Interface en ligne de commande

Le module propose également une interface CLI pour une utilisation rapide :

```bash
# Aide sur les options disponibles
python -m qai_hub_models.models.whisper_base_en.demo --help

# Transcription avec options personnalisées
python -m qai_hub_models.models.whisper_base_en.demo \
    --audio-file audio.wav \
    --on-device \
    --output-file transcription.txt
```

## Optimisation et performance

### Modèles recommandés par cas d'usage

- **Applications temps réel** : `whisper-tiny-en` (latence minimale)
- **Usage général** : `whisper-base-en` (bon équilibre performance/qualité)
- **Haute précision** : `whisper-small-en` (meilleure qualité)
- **Support multilingue** : `whisper-small-v2` (langues multiples)

### Optimisations Qualcomm

Les modèles sont optimisés pour tirer parti des unités de calcul spécialisées des puces Qualcomm (NPU, GPU, CPU) :

```
Performances exemple (Snapdragon 8 Gen 3) :
- Encoder : ~402ms, Mémoire : 19-41MB
- Decoder : ~6.9ms par token, Mémoire : 3-70MB
- Unités utilisées : NPU + GPU + CPU
```

## Dépendances

### Dépendances principales

```txt
qai_hub_models>=2.17.0
torch>=1.13.0
torchaudio>=0.13.0
numpy>=1.21.0
librosa>=0.9.0
```

### Dépendances optionnelles

```bash
# Pour le support d'autres formats audio
pip install "qai_hub_models[whisper_base_en,audio]"

# Pour le développement
pip install "qai_hub_models[whisper_base_en,dev]"
```

## Formats audio supportés

- **WAV** (recommandé)
- **MP3**
- **FLAC**
- **M4A**
- **OGG**

**Note** : Les fichiers sont automatiquement convertis au format requis (16kHz, mono) lors du prétraitement.

## Limitations

- **Durée audio** : Optimisé pour des clips de 30 secondes maximum
- **Langues** : Performance variable selon la langue (meilleures performances en anglais)
- **Qualité audio** : Performance dégradée sur audio de très faible qualité
- **Répétitions** : Possible génération de texte répétitif (mitigé par beam search)

## Déploiement

### Android

1. Exportez le modèle au format TensorFlow Lite :
```bash
python -m qai_hub_models.models.whisper_base_en.export --target-runtime tflite
```

2. Intégrez le fichier `.tflite` dans votre application Android

3. Consultez le [tutoriel de déploiement Android](https://aihub.qualcomm.com/mobile-development)

### Applications natives

Les applications natives sont disponibles dans le [dépôt AI Hub Apps](https://github.com/quic/ai-hub-apps).

## Dépannage

### Problèmes courants

1. **Erreur de token API** :
   ```bash
   qai-hub configure --api_token NOUVEAU_TOKEN
   ```

2. **Problèmes de compilation QNN** :
   - Consultez [docs/qnn_troubleshooting_guide.md](docs/qnn_troubleshooting_guide.md)

3. **Erreurs de mémoire** :
   - Utilisez un modèle plus petit (`whisper-tiny-en`)
   - Réduisez la durée des clips audio

4. **Performance dégradée** :
   - Vérifiez la qualité audio (16kHz, mono recommandé)
   - Assurez-vous que le NPU est utilisé

### Support

- [Documentation Qualcomm AI Hub](https://docs.qualcomm.com/aihub)
- [Communauté Slack AI Hub](https://qualcomm-ai-hub.slack.com)
- [Issues GitHub du projet](https://github.com/julienbltt/qualcomm-whisper-tool/issues)

## Licence

Ce projet utilise :
- **MIT License** pour l'implémentation Whisper originale
- **BSD-3 License** pour les assets compilés Qualcomm AI Hub
- Consultez le fichier [LICENSE](LICENSE) pour plus de détails

## Contribution au projet Commpanion

Ce module fait partie de l'écosystème Commpanion. Pour contribuer :

1. Fork du dépôt
2. Créez une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Committez vos changements (`git commit -am 'Ajout d'une nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créez une Pull Request

## Auteurs

- **julienbltt** - Développement initial
- Équipe **Commpanion** - Integration et optimisations

---

*Pour plus d'informations sur les modèles Whisper Qualcomm, visitez [Qualcomm AI Hub](https://aihub.qualcomm.com/models/whisper_base_en).*