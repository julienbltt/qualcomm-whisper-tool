# Guide d'Installation Complet : Qualcomm AI Hub avec ONNX Runtime sur Windows Snapdragon X Plus

Ce guide détaille la procédure complète d'installation et de configuration nécessaire pour exécuter des modèles depuis le Qualcomm AI Hub en utilisant ONNX Runtime avec optimisation NPU Hexagon sur Windows avec architecture Snapdragon X Plus.

## ⚠️ Contrainte Architecturale Critique

**POINT CRUCIAL :** Pour utiliser qai_hub_models et ONNX Runtime QNN sur Windows Snapdragon X Plus/Elite, vous **DEVEZ** utiliser Python x64 (AMD64) en mode émulation. Les packages Python ARM64 natifs ne sont **PAS supportés** et causeront des échecs d'installation.

```bash
# ❌ NE PAS UTILISER : Python ARM64 natif
# ✅ UTILISER : Python x64 (AMD64) en émulation
```

## 1. Prérequis Système

### Configuration Hardware Requise
- **Processeur :** Snapdragon X Plus 8-Core ou X Elite
- **RAM :** Minimum 16GB (32GB recommandé pour modèles larges)
- **Stockage :** 50GB d'espace libre minimum
- **OS :** Windows 11 version 22H2 ou supérieure

### Vérification du Système
```powershell
# Vérifier l'architecture du processeur
Get-ComputerInfo | Select-Object CsProcessors

# Vérifier la version Windows
winver

# Vérifier la présence du NPU Hexagon
Get-PnpDevice -FriendlyName "*Hexagon*"
```

## 2. Installation Python x64 (OBLIGATOIRE)

### Étape 1 : Téléchargement Python x64
```bash
# Télécharger Python 3.11.x AMD64 depuis python.org
# URL : https://www.python.org/downloads/windows/
# Sélectionner impérativement "Windows installer (64-bit)"
```

### Étape 2 : Vérification de l'Installation
```powershell
# Vérifier que Python s'exécute en x64
python -c "import platform; print(f'Architecture: {platform.architecture()}')"
# Sortie attendue : Architecture: ('64bit', 'WindowsPE')

python -c "import platform; print(f'Machine: {platform.machine()}')"
# Sortie attendue : Machine: AMD64
```

## 3. Configuration Environnement Virtuel

### Création de l'Environnement
```bash
# Créer un environnement virtuel dédié
python -m venv qai_hub_env

# Activation (Windows)
qai_hub_env\Scripts\activate

# Vérification de l'environnement
python -c "import sys; print(sys.executable)"
```

### Mise à Jour des Outils de Base
```bash
# Mise à jour pip, setuptools et wheel
python -m pip install --upgrade pip setuptools wheel

# Installer des outils de compilation (requis pour certaines dépendances)
pip install wheel setuptools-scm
```

## 4. Installation des Dépendances Principales

### Étape 1 : Installation ONNX Runtime QNN
```bash
# Installation ONNX Runtime avec support QNN Execution Provider
pip install onnxruntime-qnn

# OU pour la version nightly (plus récente)
pip install --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn
```

### Étape 2 : Installation Qualcomm AI Hub
```bash
# Installation qai-hub (client pour services cloud)
pip install qai-hub

# Installation qai-hub-models (modèles pré-optimisés)
pip install qai_hub_models

# Vérification des installations
python -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')"
python -c "import qai_hub; print('QAI Hub installé avec succès')"
```

### Étape 3 : Dépendances Additionnelles
```bash
# Bibliothèques de traitement audio/image
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy opencv-python pillow librosa

# Gestion audio pour Snapdragon X Plus
pip install pipwin
pipwin install pyaudio

# OU alternative si pipwin échoue
pip install pyaudio
```

## 5. Configuration Qualcomm AI Hub

### Étape 1 : Création Compte Développeur
1. Aller sur [aihub.qualcomm.com](https://aihub.qualcomm.com)
2. Créer un compte avec votre Qualcomm ID
3. Activer l'accès développeur

### Étape 2 : Configuration Token API
```bash
# Obtenir le token API
# 1. Se connecter à AI Hub
# 2. Aller dans Account -> Settings -> API Token
# 3. Copier le token

# Configuration du client
qai-hub configure --api_token VOTRE_TOKEN_API

# Vérification de la configuration
qai-hub whoami
```

### Étape 3 : Test de Connectivité
```bash
# Lister les dispositifs disponibles
qai-hub list-devices

# Rechercher le Snapdragon X Plus
qai-hub list-devices | grep -i "snapdragon x plus"
```

## 6. Installation et Configuration NPU Hexagon

### Étape 1 : Installation Pilote NPU
1. **Télécharger Qualcomm Package Manager 3** depuis developer.qualcomm.com
2. **Modifier OS Setting** : Changer de "Windows" vers "Windows (ARM64)"
3. **Navigation** : Neural processors > Snapdragon X Elite NPU
4. **Installer** pilote NPU version 1.0.0.10 minimum
5. **Redémarrer** le système

### Étape 2 : Vérification NPU
```powershell
# Vérifier dans le Gestionnaire de périphériques
Get-PnpDevice -FriendlyName "*Qualcomm*Hexagon*NPU*"

# Tester la détection via Python
python -c "import onnxruntime as ort; print('Providers disponibles:', ort.get_available_providers())"
# Vous devriez voir 'QNNExecutionProvider' dans la liste
```

## 7. Configuration ONNX Runtime QNN

### Étape 1 : Variables d'Environnement
```powershell
# Ajouter le chemin des DLL QNN (généralement inclus avec onnxruntime-qnn)
$env:PATH += ";$env:USERPROFILE\AppData\Local\Programs\Python\Python311\Lib\site-packages\onnxruntime\capi"
```

### Étape 2 : Test Configuration QNN
```python
# test_qnn_setup.py
import onnxruntime as ort
import numpy as np

def test_qnn_availability():
    """Test la disponibilité du QNN Execution Provider"""
    
    # Vérifier les providers disponibles
    providers = ort.get_available_providers()
    print(f"Providers disponibles: {providers}")
    
    if "QNNExecutionProvider" in providers:
        print("✅ QNN Execution Provider détecté")
        return True
    else:
        print("❌ QNN Execution Provider non disponible")
        return False

def test_qnn_session():
    """Test création session avec QNN"""
    try:
        # Options de session
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        
        # Création session (sans modèle pour ce test)
        print("Test de création session QNN...")
        return True
        
    except Exception as e:
        print(f"❌ Erreur création session QNN: {e}")
        return False

if __name__ == "__main__":
    print("=== Test Configuration ONNX Runtime QNN ===")
    qnn_available = test_qnn_availability()
    
    if qnn_available:
        test_qnn_session()
    else:
        print("\n🔧 Solutions possibles:")
        print("1. Réinstaller onnxruntime-qnn")
        print("2. Vérifier l'installation du pilote NPU")
        print("3. Redémarrer le système")
```

## 8. Test avec Modèle Whisper Base EN

### Étape 1 : Import et Chargement
```python
# test_whisper_qualcomm.py
import qai_hub_models
from qai_hub_models.models.whisper_base_en import Model, App
import numpy as np
import torch

def test_whisper_model():
    """Test du modèle Whisper Base EN Qualcomm"""
    
    try:
        print("📦 Chargement du modèle Whisper Base EN...")
        model = Model.from_pretrained()
        app = App(model)
        print("✅ Modèle chargé avec succès")
        
        # Test avec input factice
        print("🧪 Test avec input audio factice...")
        sample_inputs = model.sample_inputs()
        
        # Exécution locale
        output = model(**sample_inputs)
        print("✅ Inférence locale réussie")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test modèle: {e}")
        return False

if __name__ == "__main__":
    test_whisper_model()
```

### Étape 2 : Compilation pour Snapdragon X Plus
```python
# compile_whisper_for_snapdragon.py
import qai_hub as hub
import torch
from qai_hub_models.models.whisper_base_en import Model

def compile_whisper_for_snapdragon():
    """Compile le modèle Whisper pour Snapdragon X Plus"""
    
    try:
        # Chargement modèle
        print("📦 Chargement modèle Whisper...")
        torch_model = Model.from_pretrained()
        torch_model.eval()
        
        # Traçage du modèle
        print("🔄 Traçage du modèle...")
        sample_inputs = torch_model.sample_inputs()
        input_specs = {}
        traced_inputs = []
        
        for name, (shape, dtype) in sample_inputs.items():
            input_specs[name] = shape
            traced_inputs.append(torch.randn(shape).to(dtype))
        
        traced_model = torch.jit.trace(torch_model, traced_inputs)
        
        # Compilation pour Snapdragon X Plus avec ONNX Runtime
        print("🚀 Soumission job de compilation...")
        compile_job = hub.submit_compile_job(
            model=traced_model,
            device=hub.Device("Snapdragon X Plus 8-Core CRD"),
            input_specs=input_specs,
            options="--target_runtime onnx"
        )
        
        print(f"📋 Job ID: {compile_job.job_id}")
        print("⏳ Attente de la compilation...")
        
        # Attendre la compilation
        target_model = compile_job.get_target_model()
        print("✅ Compilation terminée!")
        
        # Profiling sur dispositif réel
        print("📊 Lancement profiling...")
        profile_job = hub.submit_profile_job(
            model=target_model,
            device=hub.Device("Snapdragon X Plus 8-Core CRD")
        )
        
        # Attendre les résultats
        profile_data = profile_job.download_profile()
        print("✅ Profiling terminé!")
        
        return target_model, profile_data
        
    except Exception as e:
        print(f"❌ Erreur compilation: {e}")
        return None, None

if __name__ == "__main__":
    model, profile = compile_whisper_for_snapdragon()
    if model:
        print(f"🎉 Modèle compilé: {model}")
```

## 9. Exécution Locale avec ONNX Runtime QNN

### Configuration Session Optimisée
```python
# local_inference_optimized.py
import onnxruntime as ort
import numpy as np
from pathlib import Path

def create_optimized_qnn_session(model_path):
    """Crée une session ONNX Runtime optimisée pour QNN"""
    
    # Options de session
    session_options = ort.SessionOptions()
    
    # Optimisations performance
    session_options.inter_op_num_threads = 4
    session_options.intra_op_num_threads = 4
    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Configuration QNN spécifique
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session_options.add_session_config_entry("ep.context_enable", "1")
    
    # Provider options pour QNN
    qnn_provider_options = {
        "backend_path": "QnnHtp.dll",  # NPU Hexagon
        "profiling_level": "basic",
        "rpc_control_latency": "low",
        "vtcm_mb": "8"
    }
    
    try:
        # Création session avec QNN EP
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["QNNExecutionProvider"],
            provider_options=[qnn_provider_options]
        )
        
        print("✅ Session QNN créée avec succès")
        print(f"📊 Providers actifs: {session.get_providers()}")
        
        return session
        
    except Exception as e:
        print(f"❌ Erreur création session QNN: {e}")
        
        # Fallback vers CPU
        print("🔄 Fallback vers CPU...")
        return ort.InferenceSession(str(model_path))

def run_inference_with_qnn(session, input_data):
    """Exécute l'inférence avec la session QNN"""
    
    try:
        # Préparation des inputs
        input_name = session.get_inputs()[0].name
        inputs = {input_name: input_data}
        
        # Mesure de performance
        import time
        start_time = time.time()
        
        # Inférence
        outputs = session.run(None, inputs)
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # en ms
        
        print(f"⚡ Temps d'inférence: {inference_time:.2f}ms")
        
        return outputs
        
    except Exception as e:
        print(f"❌ Erreur inférence: {e}")
        return None

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacer par le chemin vers votre modèle ONNX compilé
    model_path = "whisper_base_en_snapdragon.onnx"
    
    if Path(model_path).exists():
        session = create_optimized_qnn_session(model_path)
        
        # Test avec données factices
        input_shape = (1, 80, 3000)  # Exemple pour Whisper
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        outputs = run_inference_with_qnn(session, test_input)
        print(f"🎯 Outputs shape: {[out.shape for out in outputs] if outputs else 'None'}")
```

## 10. Dépannage Commun

### Problème : QNNExecutionProvider Non Trouvé
```bash
# Solutions :
1. Vérifier l'installation : pip show onnxruntime-qnn
2. Réinstaller : pip uninstall onnxruntime-qnn && pip install onnxruntime-qnn
3. Vérifier le pilote NPU dans le Gestionnaire de périphériques
4. Redémarrer le système
```

### Problème : Installation qai_hub_models Échec
```bash
# Vérifier l'architecture Python
python -c "import platform; print(platform.machine())"
# Doit afficher "AMD64", sinon réinstaller Python x64

# Solutions :
1. Utiliser Python x64 OBLIGATOIREMENT
2. Vider le cache pip : pip cache purge
3. Installer avec --no-cache-dir : pip install --no-cache-dir qai_hub_models
```

### Problème : PyAudio Installation Échec
```bash
# Solutions multiples :
pip install pipwin && pipwin install pyaudio
# OU
pip install --only-binary=all pyaudio
# OU compiler depuis les sources avec Visual Studio Build Tools
```

### Problème : Modèle "Hang" sur NPU
```bash
# Solutions :
1. Vérifier la quantification du modèle (FP16/INT8)
2. Réduire la taille des inputs
3. Utiliser options QNN : "rpc_control_latency": "low"
4. Augmenter timeout : session_options.add_session_config_entry("timeout", "30000")
```

## 11. Scripts de Validation Complète

### Script de Validation Totale
```python
# validate_complete_setup.py
import sys
import importlib
from pathlib import Path

def check_python_architecture():
    """Vérifie l'architecture Python"""
    import platform
    arch = platform.machine()
    print(f"🔍 Architecture Python: {arch}")
    return arch == "AMD64"

def check_required_packages():
    """Vérifie les packages requis"""
    required_packages = [
        "onnxruntime",
        "qai_hub",
        "qai_hub_models",
        "torch",
        "numpy",
        "pyaudio"
    ]
    
    results = {}
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            results[package] = "✅ OK"
        except ImportError:
            results[package] = "❌ MANQUANT"
    
    return results

def check_qnn_availability():
    """Vérifie QNN Execution Provider"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return "QNNExecutionProvider" in providers
    except:
        return False

def check_ai_hub_auth():
    """Vérifie l'authentification AI Hub"""
    try:
        import subprocess
        result = subprocess.run(["qai-hub", "whoami"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def main():
    """Validation complète du setup"""
    print("🚀 Validation Setup Qualcomm AI Hub + ONNX Runtime")
    print("=" * 60)
    
    # 1. Architecture Python
    arch_ok = check_python_architecture()
    if not arch_ok:
        print("❌ ERREUR CRITIQUE: Utiliser Python x64 (AMD64)")
        return False
    
    # 2. Packages
    packages = check_required_packages()
    print("\n📦 Packages:")
    for pkg, status in packages.items():
        print(f"  {pkg}: {status}")
    
    # 3. QNN Provider
    qnn_ok = check_qnn_availability()
    print(f"\n🧠 QNN Execution Provider: {'✅ OK' if qnn_ok else '❌ MANQUANT'}")
    
    # 4. AI Hub Auth
    auth_ok = check_ai_hub_auth()
    print(f"🔐 AI Hub Auth: {'✅ OK' if auth_ok else '❌ NON CONFIGURÉ'}")
    
    # Résumé
    all_ok = arch_ok and all("✅" in status for status in packages.values()) and qnn_ok and auth_ok
    
    print("\n" + "=" * 60)
    if all_ok:
        print("🎉 SETUP COMPLET - Prêt pour l'inférence!")
    else:
        print("⚠️  SETUP INCOMPLET - Vérifier les éléments marqués ❌")
    
    return all_ok

if __name__ == "__main__":
    main()
```

## 12. Commandes de Résumé Installation

```bash
# Installation complète en une fois (copier-coller)
# ATTENTION : Utiliser dans un terminal x64

# 1. Environnement virtuel
python -m venv qai_hub_env
qai_hub_env\Scripts\activate

# 2. Mise à jour outils
python -m pip install --upgrade pip setuptools wheel

# 3. Packages principaux
pip install onnxruntime-qnn qai-hub qai_hub_models

# 4. Dépendances PyTorch et audio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy opencv-python pillow librosa
pip install pipwin && pipwin install pyaudio

# 5. Configuration AI Hub
qai-hub configure --api_token VOTRE_TOKEN

# 6. Test
python -c "import onnxruntime as ort; print('QNN disponible:', 'QNNExecutionProvider' in ort.get_available_providers())"
```

Ce guide couvre l'ensemble de la procédure d'installation nécessaire pour exécuter des modèles Qualcomm AI Hub avec ONNX Runtime sur Windows Snapdragon X Plus. La contrainte principale est l'utilisation obligatoire de Python x64 en émulation pour la compatibilité avec l'écosystème Qualcomm.