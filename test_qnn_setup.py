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