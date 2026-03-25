#!/usr/bin/env python3
"""Verify that all agents load .env from root directory correctly."""

import os
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

def test_env_loading():
    """Test that .env is loaded correctly."""
    print("🔍 Verificando carga de .env desde el directorio raíz...\n")
    
    # Check if .env exists
    env_file = root_dir / '.env'
    if not env_file.exists():
        print(f"❌ Archivo .env no encontrado en: {env_file}")
        return False
    
    print(f"✅ Archivo .env encontrado en: {env_file}\n")
    
    # Test each agent's env loading and README
    agents = [
        ('Images Agent', 'samples/python/agents/images/app/__main__.py', 'samples/python/agents/images/README.md'),
        ('Medical Agent', 'samples/python/agents/medical_Images/app/__main__.py', 'samples/python/agents/medical_Images/README.md'),
        ('Multimodal Agent', 'samples/python/agents/multimodal/app/__main__.py', 'samples/python/agents/multimodal/README.md'),
        ('Demo UI', 'demo/ui/main.py', None),
    ]
    
    all_passed = True
    for agent_info in agents:
        agent_name = agent_info[0]
        agent_path = agent_info[1]
        readme_path = agent_info[2] if len(agent_info) > 2 else None
        
        agent_file = root_dir / agent_path
        if not agent_file.exists():
            print(f"❌ {agent_name}: Archivo no encontrado - {agent_file}")
            all_passed = False
            continue
        
        # Check if the file contains the correct .env loading code
        content = agent_file.read_text()
        if 'load_dotenv(dotenv_path=env_path)' in content and 'Path(__file__).resolve().parents' in content:
            print(f"✅ {agent_name}: Configurado correctamente para cargar .env desde root")
        else:
            print(f"❌ {agent_name}: No está configurado correctamente")
            all_passed = False
        
        # Check README if applicable
        if readme_path:
            readme_file = root_dir / readme_path
            if readme_file.exists():
                print(f"   ✅ README.md presente")
            else:
                print(f"   ❌ README.md faltante")
                all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ Todos los agentes están configurados correctamente")
        print("\nAhora todos los agentes compartirán las mismas API keys desde:")
        print(f"   {env_file}")
    else:
        print("❌ Algunos agentes tienen problemas de configuración")
    print("="*70)
    
    return all_passed

if __name__ == '__main__':
    success = test_env_loading()
    sys.exit(0 if success else 1)
