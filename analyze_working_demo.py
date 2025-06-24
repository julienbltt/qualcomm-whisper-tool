# analyze_working_demo.py
"""
Analyse du demo Whisper qui fonctionne pour extraire la vraie API
et créer un module réutilisable basé sur ce qui marche vraiment
"""

import inspect
import importlib
import sys
from pathlib import Path

def find_demo_source_code():
    """Trouve et analyse le code source du demo qui fonctionne"""
    
    print("🔍 ANALYSE DU DEMO WHISPER QUI FONCTIONNE")
    print("=" * 50)
    
    try:
        # Importer le module demo
        import qai_hub_models.models.whisper_base_en.demo as demo_module
        
        print("✅ Module demo importé avec succès")
        
        # Obtenir le code source
        try:
            source_code = inspect.getsource(demo_module)
            print(f"✅ Code source récupéré ({len(source_code)} caractères)")
            
            # Sauvegarder le code source
            with open("whisper_demo_source.py", "w", encoding="utf-8") as f:
                f.write(source_code)
            print("💾 Code source sauvegardé dans 'whisper_demo_source.py'")
            
        except Exception as e:
            print(f"⚠️  Impossible de récupérer le code source: {e}")
        
        # Analyser les fonctions disponibles
        print(f"\n📋 Fonctions dans le module demo:")
        demo_functions = []
        for name, obj in inspect.getmembers(demo_module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                demo_functions.append(name)
                print(f"  🔧 {name}")
                
                # Obtenir la signature
                try:
                    sig = inspect.signature(obj)
                    print(f"    Signature: {name}{sig}")
                except:
                    print(f"    Signature: Non disponible")
        
        return demo_module, demo_functions
        
    except Exception as e:
        print(f"❌ Erreur import demo: {e}")
        return None, []

def analyze_demo_main_function(demo_module):
    """Analyse la fonction principale du demo"""
    
    print(f"\n🎯 ANALYSE FONCTION PRINCIPALE")
    print("=" * 35)
    
    # Chercher la fonction main ou équivalente
    main_functions = ['main', 'demo', 'run', 'run_demo']
    
    for func_name in main_functions:
        if hasattr(demo_module, func_name):
            func = getattr(demo_module, func_name)
            print(f"✅ Fonction trouvée: {func_name}")
            
            try:
                # Analyser les paramètres
                sig = inspect.signature(func)
                print(f"📋 Signature: {func_name}{sig}")
                
                # Obtenir le code source de cette fonction
                func_source = inspect.getsource(func)
                print(f"📝 Code source ({len(func_source)} caractères):")
                print("-" * 40)
                print(func_source)
                print("-" * 40)
                
                return func
                
            except Exception as e:
                print(f"⚠️  Erreur analyse {func_name}: {e}")
    
    print("❌ Aucune fonction principale trouvée")
    return None

def extract_working_api_pattern(demo_module):
    """Extrait le pattern API qui fonctionne"""
    
    print(f"\n🔬 EXTRACTION PATTERN API")
    print("=" * 30)
    
    try:
        # Chercher les imports dans le module
        source_file = inspect.getfile(demo_module)
        print(f"📁 Fichier source: {source_file}")
        
        # Lire le fichier source directement
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("📝 Contenu du fichier demo:")
        print("=" * 40)
        print(content)
        print("=" * 40)
        
        # Analyser les imports
        lines = content.split('\n')
        imports = []
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                imports.append(line.strip())
        
        print(f"\n📦 Imports utilisés:")
        for imp in imports:
            print(f"  {imp}")
        
        return content
        
    except Exception as e:
        print(f"❌ Erreur extraction pattern: {e}")
        return None

def create_working_module_template(api_pattern):
    """Crée un template de module basé sur le pattern qui fonctionne"""
    
    print(f"\n🏗️  CRÉATION MODULE TEMPLATE")
    print("=" * 35)
    
    if not api_pattern:
        print("❌ Pas de pattern API disponible")
        return
    
    # Analyser le pattern pour créer un template
    template = f'''# whisper_stt_working.py
"""
Module STT Whisper basé sur le demo officiel qui fonctionne
Extrait du code source du demo: qai_hub_models.models.whisper_base_en.demo
"""

{api_pattern}

class WorkingWhisperSTT:
    """Module STT basé sur l'API qui fonctionne vraiment"""
    
    def __init__(self):
        # À adapter selon ce qui fonctionne dans le demo
        pass
    
    def transcribe_audio(self, audio_data):
        """Transcription basée sur le demo qui fonctionne"""
        # À implémenter selon l'API du demo
        pass

# Template d'intégration
class WhisperIntegration:
    def __init__(self, wake_word_detector=None):
        self.whisper_stt = WorkingWhisperSTT()
        self.wake_word_detector = wake_word_detector
        
        if wake_word_detector:
            self._setup_integration()
    
    def _setup_integration(self):
        def on_wake_word(word, confidence):
            print(f"Wake word détecté: {{word}}")
            # Activer STT ici
            self.activate_stt()
        
        if hasattr(self.wake_word_detector, 'register_callback'):
            self.wake_word_detector.register_callback('trigger', on_wake_word)
    
    def activate_stt(self):
        print("🎤 STT activé - Utilisation API qui fonctionne")
        # Implémenter avec l'API du demo

if __name__ == "__main__":
    # Test basé sur ce qui fonctionne
    integration = WhisperIntegration()
    integration.activate_stt()
'''
    
    # Sauvegarder le template
    with open("whisper_stt_working_template.py", "w", encoding="utf-8") as f:
        f.write(template)
    
    print("✅ Template créé: whisper_stt_working_template.py")
    print("💡 Personnalisez ce template avec l'API qui fonctionne")

def run_demo_directly():
    """Execute le demo directement pour voir ce qu'il fait"""
    
    print(f"\n🚀 EXÉCUTION DEMO DIRECT")
    print("=" * 30)
    
    try:
        import subprocess
        import sys
        
        print("📦 Lancement demo avec verbose...")
        
        # Lancer avec capture détaillée
        result = subprocess.run([
            sys.executable, "-m", 
            "qai_hub_models.models.whisper_base_en.demo",
            "--help"  # Voir les options
        ], capture_output=True, text=True, timeout=30)
        
        print("📤 Sortie stdout:")
        print(result.stdout)
        
        if result.stderr:
            print("📤 Sortie stderr:")
            print(result.stderr)
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur exécution demo: {e}")
        return False

def main():
    """Analyse complète du demo qui fonctionne"""
    
    print("🎯 INGÉNIERIE INVERSE DU DEMO WHISPER")
    print("=" * 45)
    print("Objectif: Comprendre pourquoi le demo fonctionne")
    print("         et extraire l'API correcte\n")
    
    # 1. Analyser le code source
    demo_module, functions = find_demo_source_code()
    
    if demo_module:
        # 2. Analyser la fonction principale
        main_func = analyze_demo_main_function(demo_module)
        
        # 3. Extraire le pattern API
        api_pattern = extract_working_api_pattern(demo_module)
        
        # 4. Créer un template basé sur ce qui fonctionne
        create_working_module_template(api_pattern)
        
        print(f"\n✅ ANALYSE TERMINÉE")
        print("📁 Fichiers créés:")
        print("  - whisper_demo_source.py (code source complet)")
        print("  - whisper_stt_working_template.py (template)")
        
        print(f"\n💡 PROCHAINES ÉTAPES:")
        print("1. Examiner whisper_demo_source.py")
        print("2. Identifier l'API qui fonctionne")
        print("3. Adapter whisper_stt_working_template.py")
        print("4. Intégrer dans votre système existant")
    
    # 5. Exécuter le demo pour voir ce qu'il fait
    run_demo_directly()

if __name__ == "__main__":
    main()