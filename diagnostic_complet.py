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