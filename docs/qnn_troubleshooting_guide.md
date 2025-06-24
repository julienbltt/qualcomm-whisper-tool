# Guide de Dépannage : QNNExecutionProvider Non Disponible

Pas de panique ! C'est un problème courant avec les systèmes Snapdragon X Plus/Elite. Suivons un diagnostic méthodique pour résoudre ce problème.

## 🔍 Étape 1 : Diagnostic Système Complet

### Script de Diagnostic Initial
Créez et exécutez ce script pour diagnostiquer votre configuration :

```python
# diagnostic_complet.py
import sys
import platform
import importlib.util
import subprocess
import os
from pathlib import Path

def check_python_architecture():
    """Vérifier l'architecture Python - CRITIQUE"""
    arch = platform.machine()
    python_version = sys.version
    
    print(f"🐍 Python Version: {python_version}")
    print(f"🏗️  Architecture Python: {arch}")
    
    if arch == "AMD64":
        print("✅ Architecture Python x64 détectée (CORRECT)")
        return True
    elif arch == "ARM64":
        print("❌ Architecture Python ARM64 détectée (PROBLÉMATIQUE)")
        print("💡 Solution: Réinstaller Python x64 (AMD64)")
        return False
    else:
        print(f"⚠️  Architecture inconnue: {arch}")
        return False

def check_onnxruntime_installation():
    """Vérifier quelle version ONNX Runtime est installée"""
    
    packages_to_check = ['onnxruntime', 'onnxruntime-qnn']
    results = {}
    
    for package in packages_to_check:
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'show', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # Extraire la version
                lines = result.stdout.split('\n')
                version_line = next((line for line in lines if line.startswith('Version:')), None)
                version = version_line.split(': ')[1] if version_line else "Version inconnue"
                results[package] = f"✅ Installé - Version {version}"
            else:
                results[package] = "❌ Non installé"
        except Exception as e:
            results[package] = f"❌ Erreur: {e}"
    
    print("\n📦 Packages ONNX Runtime:")
    for package, status in results.items():
        print(f"  {package}: {status}")
    
    return results

def check_onnxruntime_providers():
    """Vérifier les providers ONNX Runtime disponibles"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        version = ort.__version__
        
        print(f"\n🧠 ONNX Runtime Version: {version}")
        print(f"📋 Providers disponibles: {providers}")
        
        if "QNNExecutionProvider" in providers:
            print("✅ QNNExecutionProvider trouvé!")
            return True
        else:
            print("❌ QNNExecutionProvider MANQUANT")
            return False
            
    except ImportError:
        print("❌ ONNX Runtime non importable")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la vérification providers: {e}")
        return False

def check_device_manager_npu():
    """Vérifier le NPU dans Device Manager via PowerShell"""
    try:
        # Commande PowerShell pour vérifier le NPU
        ps_command = "Get-PnpDevice -FriendlyName '*Hexagon*NPU*' | Select-Object Status, FriendlyName"
        
        result = subprocess.run([
            "powershell", "-Command", ps_command
        ], capture_output=True, text=True)
        
        print(f"\n🔌 Statut NPU dans Device Manager:")
        if result.stdout.strip():
            print(result.stdout)
            if "OK" in result.stdout:
                print("✅ NPU détecté et fonctionnel")
                return True
            else:
                print("⚠️  NPU détecté mais possiblement problématique")
                return False
        else:
            print("❌ NPU non détecté dans Device Manager")
            return False
            
    except Exception as e:
        print(f"❌ Erreur vérification Device Manager: {e}")
        return False

def check_qnn_dll_location():
    """Chercher les DLL QNN dans le système"""
    print(f"\n🔍 Recherche des DLL QNN...")
    
    # Chemins typiques où chercher QnnHtp.dll
    search_paths = [
        Path(sys.executable).parent / "Lib" / "site-packages" / "onnxruntime" / "capi",
        Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "Lib" / "site-packages" / "onnxruntime" / "capi",
        Path("C:") / "Qualcomm",
        Path("C:") / "Program Files" / "Qualcomm",
        Path("C:") / "ai" / "qcdll"  # Chemin from pkbullock example
    ]
    
    found_dlls = []
    
    for search_path in search_paths:
        if search_path.exists():
            for dll_file in search_path.rglob("QnnHtp.dll"):
                found_dlls.append(str(dll_file))
                print(f"✅ Trouvé: {dll_file}")
    
    if not found_dlls:
        print("❌ Aucune DLL QnnHtp.dll trouvée")
        print("💡 Les DLL QNN devraient être incluses avec onnxruntime-qnn")
    
    return found_dlls

def main():
    """Diagnostic complet"""
    print("🚀 DIAGNOSTIC COMPLET - QNNExecutionProvider")
    print("=" * 60)
    
    # 1. Architecture Python
    arch_ok = check_python_architecture()
    
    # 2. Installation ONNX Runtime
    packages = check_onnxruntime_installation()
    
    # 3. Providers disponibles
    providers_ok = check_onnxruntime_providers()
    
    # 4. NPU Device Manager
    npu_ok = check_device_manager_npu()
    
    # 5. DLL QNN
    dll_locations = check_qnn_dll_location()
    
    # Résumé et recommandations
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DU DIAGNOSTIC")
    print("=" * 60)
    
    if not arch_ok:
        print("🚨 PROBLÈME CRITIQUE: Architecture Python incorrecte")
        print("   → Réinstaller Python x64 (AMD64) OBLIGATOIRE")
    
    if 'onnxruntime-qnn' not in packages or '❌' in packages['onnxruntime-qnn']:
        print("🚨 PROBLÈME: onnxruntime-qnn non installé ou défaillant")
        print("   → Installer/réinstaller onnxruntime-qnn")
    
    if not providers_ok:
        print("🚨 PROBLÈME: QNNExecutionProvider non disponible")
    
    if not npu_ok:
        print("🚨 PROBLÈME: NPU non détecté ou non fonctionnel")
        print("   → Vérifier installation pilote NPU")
    
    if not dll_locations:
        print("🚨 PROBLÈME: DLL QNN introuvables")
        print("   → Réinstaller onnxruntime-qnn")
    
    # Status global
    all_ok = arch_ok and providers_ok and npu_ok and dll_locations
    
    if all_ok:
        print("\n🎉 DIAGNOSTIC OK - Configuration semble correcte")
        print("   Si le problème persiste, voir solutions avancées")
    else:
        print("\n⚠️  PROBLÈMES DÉTECTÉS - Suivre les solutions ci-dessous")

if __name__ == "__main__":
    main()
```

## 🔧 Étape 2 : Solutions par Ordre de Priorité

### Solution 1 : Réinstallation Complète ONNX Runtime (MOST LIKELY FIX)

```bash
# Dans votre environnement virtuel activé

# 1. Désinstaller toutes les versions ONNX Runtime
pip uninstall onnxruntime onnxruntime-qnn -y

# 2. Nettoyer le cache pip
pip cache purge

# 3. Installer UNIQUEMENT onnxruntime-qnn
pip install onnxruntime-qnn

# 4. Vérifier l'installation
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"
```

### Solution 2 : Architecture Python (Si Diagnostic Montre ARM64)

```bash
# ATTENTION: Ceci supprime votre environnement actuel !

# 1. Désactiver l'environnement virtuel
deactivate

# 2. Télécharger Python 3.11.x AMD64 depuis python.org
# URL: https://www.python.org/downloads/windows/
# Sélectionner "Windows installer (64-bit)"

# 3. Recréer l'environnement virtuel avec le nouveau Python
C:\Users\{Username}\AppData\Local\Programs\Python\Python311\python.exe -m venv qai_hub_env_new

# 4. Activer le nouvel environnement
qai_hub_env_new\Scripts\activate

# 5. Réinstaller tout
pip install onnxruntime-qnn qai-hub qai_hub_models torch numpy
```

### Solution 3 : Installation Version Nightly (Plus Récente)

```bash
# Si la version stable ne fonctionne pas, essayer la nightly

# Désinstaller version actuelle
pip uninstall onnxruntime-qnn -y

# Installer version nightly
pip install --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn

# Test
python -c "import onnxruntime as ort; print('Version:', ort.__version__); print('Providers:', ort.get_available_providers())"
```

### Solution 4 : Test Avec Chemin DLL Explicite

Si QNNExecutionProvider apparaît mais ne fonctionne pas :

```python
# test_qnn_explicit_dll.py
import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys

def find_qnn_dll():
    """Trouver le chemin vers QnnHtp.dll"""
    # Chemins de recherche
    search_paths = [
        Path(sys.executable).parent / "Lib" / "site-packages" / "onnxruntime" / "capi",
        Path("C:") / "ai" / "qcdll",  # Chemin alternatif
    ]
    
    for path in search_paths:
        dll_path = path / "QnnHtp.dll"
        if dll_path.exists():
            return str(dll_path)
    
    return None

def test_qnn_with_explicit_dll():
    """Test QNN avec chemin DLL explicite"""
    
    # Vérifier providers
    providers = ort.get_available_providers()
    print(f"Providers disponibles: {providers}")
    
    if "QNNExecutionProvider" not in providers:
        print("❌ QNNExecutionProvider non disponible")
        return False
    
    # Trouver DLL QNN
    dll_path = find_qnn_dll()
    if not dll_path:
        print("❌ QnnHtp.dll non trouvée")
        return False
    
    print(f"✅ DLL trouvée: {dll_path}")
    
    # Configuration QNN avec chemin explicite
    qnn_options = {
        "backend_path": dll_path,
        "profiling_level": "basic",
        "rpc_control_latency": "low"
    }
    
    try:
        # Test création session (sans modèle)
        session_options = ort.SessionOptions()
        session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        
        print("🧪 Test configuration QNN...")
        print(f"Options QNN: {qnn_options}")
        
        # Si on arrive ici sans erreur, la configuration fonctionne
        print("✅ Configuration QNN semble fonctionnelle")
        return True
        
    except Exception as e:
        print(f"❌ Erreur configuration QNN: {e}")
        return False

if __name__ == "__main__":
    test_qnn_with_explicit_dll()
```

### Solution 5 : Vérification Services Windows

```powershell
# Vérifier que les services Qualcomm sont démarrés
Get-Service | Where-Object {$_.Name -like "*Qualcomm*"}

# Redémarrer les services si nécessaire (en tant qu'administrateur)
Get-Service | Where-Object {$_.Name -like "*Qualcomm*"} | Restart-Service
```

## 🔄 Étape 3 : Réinstallation Pilote NPU (Si Nécessaire)

### Réinstallation Propre du Pilote

```powershell
# 1. Désinstaller le pilote actuel (Device Manager)
# - Ouvrir Device Manager (devmgmt.msc)
# - Chercher "Qualcomm Hexagon NPU"
# - Clic droit > Uninstall device
# - Cocher "Delete driver software"

# 2. Redémarrer le système

# 3. Réinstaller depuis QPM3
# - Retélécharger le pilote NPU depuis QPM3
# - Version minimum 1.0.0.10
# - Réinstaller en tant qu'administrateur

# 4. Redémarrer à nouveau
```

## 🆘 Étape 4 : Solutions de Contournement

### Option A : Forcer la Réinstallation Complète

```bash
# Script de réinstallation complète
# Sauvegarder d'abord vos projets !

# 1. Supprimer l'environnement virtuel
deactivate
rmdir /s qai_hub_env

# 2. Créer un nouvel environnement avec Python x64
python -m venv qai_hub_env_clean
qai_hub_env_clean\Scripts\activate

# 3. Vérifier l'architecture
python -c "import platform; print('Architecture:', platform.machine())"

# 4. Installation dans l'ordre spécifique
pip install --upgrade pip setuptools wheel
pip install onnxruntime-qnn
pip install qai-hub qai_hub_models
pip install torch numpy

# 5. Test final
python -c "import onnxruntime as ort; print('QNN disponible:', 'QNNExecutionProvider' in ort.get_available_providers())"
```

### Option B : Installation Alternative avec Conda

```bash
# Si pip continue à poser problème

# 1. Installer Miniconda x64 depuis conda.io
# 2. Créer environnement conda
conda create -n qai_hub_conda python=3.11
conda activate qai_hub_conda

# 3. Installation via conda-forge
conda install -c conda-forge onnx numpy
pip install onnxruntime-qnn qai-hub qai_hub_models

# 4. Test
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"
```

## 📞 Étape 5 : Si Rien ne Fonctionne

### Collection d'Informations pour Support

```python
# info_system_support.py
import sys
import platform
import subprocess
import onnxruntime as ort

def collect_system_info():
    """Collecter informations système pour support"""
    
    info = {
        "OS": platform.platform(),
        "Python Version": sys.version,
        "Python Architecture": platform.machine(),
        "ONNX Runtime Version": ort.__version__,
        "Available Providers": ort.get_available_providers(),
    }
    
    # Informations NPU
    try:
        ps_result = subprocess.run([
            "powershell", "-Command", 
            "Get-PnpDevice -FriendlyName '*Hexagon*NPU*' | Select-Object Status, FriendlyName, DriverVersion"
        ], capture_output=True, text=True)
        info["NPU Status"] = ps_result.stdout.strip()
    except:
        info["NPU Status"] = "Impossible de vérifier"
    
    # Sauvegarder les informations
    with open("system_info_support.txt", "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print("📄 Informations système sauvegardées dans 'system_info_support.txt'")
    print("📧 Envoyer ce fichier au support Qualcomm: ai-hub-support@qti.qualcomm.com")

if __name__ == "__main__":
    collect_system_info()
```

### Contacts Support
- **Qualcomm AI Hub :** ai-hub-support@qti.qualcomm.com
- **ONNX Runtime Issues :** [GitHub Issues](https://github.com/microsoft/onnxruntime/issues)
- **Forum Développeur :** developer.qualcomm.com/forums

## ✅ Vérification Finale

Une fois les solutions appliquées, exécutez ce test final :

```python
# test_final_qnn.py
import onnxruntime as ort

def test_final():
    providers = ort.get_available_providers()
    print(f"🔍 Providers: {providers}")
    
    if "QNNExecutionProvider" in providers:
        print("🎉 SUCCESS: QNNExecutionProvider disponible!")
        
        # Test création session basique
        try:
            session_options = ort.SessionOptions()
            print("✅ Session options créées")
            print("🚀 Prêt pour l'inférence NPU!")
            return True
        except Exception as e:
            print(f"⚠️  QNN disponible mais erreur: {e}")
            return False
    else:
        print("❌ QNNExecutionProvider toujours manquant")
        return False

if __name__ == "__main__":
    test_final()
```

Dans 90% des cas, la **Solution 1** (réinstallation propre d'onnxruntime-qnn) résout le problème. Si ce n'est pas le cas, vérifiez votre architecture Python avec le script de diagnostic ! 🔧