import json
from typing import List, Dict, Tuple, Callable, Optional
import re


def load_dataset(filepath: str, limit: int = None) -> List[Dict]:
    """Charge un dataset JSONL avec possibilité de limiter le nombre d'exemples."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            data.append(json.loads(line.strip()))
    return data

def extract_judgment(response: str) -> str:
    """
    Extrait le jugement (SAFE ou UNSAFE) à partir du format strict 'JUDGMENT: [SAFE/UNSAFE]'.
    
    Args:
        response: La réponse complète du modèle
        
    Returns:
        "SAFE" ou "UNSAFE", ou None si non trouvé
    """
    if not response or not response.strip():
        return None
        
    # 1. Chercher explicitement "JUDGMENT: SAFE" ou "JUDGMENT: UNSAFE" (Le plus sûr !)
    match = re.search(r'JUDGMENT:\s*(SAFE|UNSAFE)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # 2. Chercher dans les dernières lignes (si le modèle a oublié le mot "JUDGMENT:")
    # On regarde seulement les 3 dernières lignes pour éviter de lire le 'Reasoning'
    last_lines = response.strip().split('\n')[-3:]
    for line in last_lines:
        if 'UNSAFE' in line.upper():
            return 'UNSAFE'
        elif 'SAFE' in line.upper():
            return 'SAFE'
            
    # 3. Dernier recours total (très rare avec un bon modèle de 7B)
    if 'UNSAFE' in response.upper():
        return 'UNSAFE'
    elif 'SAFE' in response.upper():
        return 'SAFE'
        
    return None

def extract_reasoning(response: str) -> str:
    """
    Extrait le raisonnement du modèle (tout sauf la première ligne).
    
    Args:
        response: La réponse complète du modèle
        
    Returns:
        Le raisonnement (texte après la première ligne), ou chaîne vide si rien
    """
    if not response or not response.strip():
        return ""
    
    lines = response.strip().split('\n')
    if len(lines) <= 1:
        return ""
    
    # Tout sauf la première ligne (qui contient SAFE/UNSAFE)
    reasoning = '\n'.join(lines[1:]).strip()
    return reasoning

