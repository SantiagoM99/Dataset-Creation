# src/mesh_downloader.py
"""
MeshDownloader - Descarga y procesa el archivo oficial de MeSH

Esta clase se encarga de:
1. Descargar el archivo desc2025.xml desde NCBI
2. Parsear el XML y extraer términos MeSH, tree numbers y nombres
3. Generar mapeos optimizados para búsquedas rápidas
4. Guardar todo en un archivo JSON optimizado

Uso:
    downloader = MeshDownloader()
    mesh_data = downloader.setup_mesh_data()
"""

import requests
import xml.etree.ElementTree as ET
import json
import os
from typing import Dict, List, Set

class MeshDownloader:
    """
    Descarga y procesa datos oficiales de MeSH desde NCBI
    
    Attributes:
        mesh_url (str): URL del archivo oficial MeSH XML
        mesh_file (str): Nombre del archivo XML local
        mesh_data_file (str): Nombre del archivo JSON procesado
    """
    
    def __init__(self, mesh_data_dir: str = "mesh/data"):
        """
        Inicializa el downloader de MeSH
        
        Args:
            mesh_data_dir (str): Directorio donde guardar los archivos de MeSH
        """
        self.mesh_url = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2025.xml"
        self.mesh_data_dir = mesh_data_dir
        self.mesh_file = os.path.join(mesh_data_dir, "desc2025.xml")
        self.mesh_data_file = os.path.join(mesh_data_dir, "mesh_data.json")
        
        # Crear directorio si no existe
        os.makedirs(mesh_data_dir, exist_ok=True)
        
    def download_mesh_file(self) -> None:
        """
        Descarga el archivo oficial de MeSH si no existe
        
        El archivo es aproximadamente 200MB y contiene todos los términos
        MeSH oficiales con sus tree numbers y metadata.
        """
        if os.path.exists(self.mesh_file):
            print(f"✅ Archivo {self.mesh_file} ya existe. Saltando descarga.")
            return
        
        print("📥 Descargando archivo MeSH oficial...")
        print(f"🔗 URL: {self.mesh_url}")
        print("⏳ Esto puede tomar varios minutos...")
        
        try:
            response = requests.get(self.mesh_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(self.mesh_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r📊 Descargando: {percent:.1f}%", end='', flush=True)
            
            print(f"\n✅ Descarga completa: {self.mesh_file}")
            
        except Exception as e:
            print(f"❌ Error descargando archivo: {e}")
            raise
    
    def parse_mesh_xml(self) -> Dict:
        """
        Parsea el XML de MeSH y extrae información relevante
        
        Extrae tres tipos de mapeos principales:
        - ui_to_tree_numbers: MeSH UI -> Lista de tree numbers
        - tree_to_name: Tree number -> Nombre de categoría  
        - ui_to_name: MeSH UI -> Nombre del término
        
        Returns:
            Dict: Datos procesados de MeSH listos para uso
        """
        print("🔄 Parseando archivo MeSH XML...")
        print("⏳ Esto puede tomar varios minutos...")
        
        # Estructura para almacenar datos
        mesh_data = {
            'ui_to_tree_numbers': {},  # D004285 -> ["B01.050.150.900.649.313"]
            'tree_to_name': {},        # "B01" -> "Eukaryota"
            'ui_to_name': {}           # D004285 -> "Dogs"
        }
        
        try:
            # Parsear XML
            tree = ET.parse(self.mesh_file)
            root = tree.getroot()
            
            descriptor_count = 0
            
            print("📖 Extrayendo términos MeSH...")
            for descriptor_record in root.findall('.//DescriptorRecord'):
                descriptor_count += 1
                if descriptor_count % 1000 == 0:
                    print(f"📊 Procesados {descriptor_count} descriptores...")
                
                # Obtener UI (Unique Identifier)
                descriptor_ui_elem = descriptor_record.find('.//DescriptorUI')
                if descriptor_ui_elem is None:
                    continue
                descriptor_ui = descriptor_ui_elem.text
                
                # Obtener nombre del descriptor
                descriptor_name_elem = descriptor_record.find('.//DescriptorName/String')
                if descriptor_name_elem is not None:
                    descriptor_name = descriptor_name_elem.text
                    mesh_data['ui_to_name'][descriptor_ui] = descriptor_name
                
                # Obtener Tree Numbers
                tree_numbers = []
                tree_number_list = descriptor_record.find('.//TreeNumberList')
                if tree_number_list is not None:
                    for tree_number_elem in tree_number_list.findall('.//TreeNumber'):
                        if tree_number_elem.text:
                            tree_numbers.append(tree_number_elem.text)
                
                if tree_numbers:
                    mesh_data['ui_to_tree_numbers'][descriptor_ui] = tree_numbers
            
            print(f"✅ Parseados {descriptor_count} descriptores")
            
            # Generar mapeo de tree numbers a nombres de categorías
            print("🗂️  Generando mapeo de categorías...")
            self._generate_tree_to_name_mapping(mesh_data)
            
            # Guardar datos procesados
            print(f"💾 Guardando datos procesados en {self.mesh_data_file}...")
            with open(self.mesh_data_file, 'w', encoding='utf-8') as f:
                json.dump(mesh_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Datos guardados exitosamente")
            return mesh_data
            
        except Exception as e:
            print(f"❌ Error parseando XML: {e}")
            raise
    
    def _generate_tree_to_name_mapping(self, mesh_data: Dict) -> None:
        """
        Genera el mapeo de tree numbers a nombres usando los descriptores
        
        Crea mapeos para todos los niveles del árbol jerárquico de MeSH,
        desde categorías principales (A, B, C...) hasta subcategorías específicas.
        
        Args:
            mesh_data (Dict): Datos de MeSH a procesar
        """
        # Mapeo manual de categorías principales (fijas en MeSH)
        main_categories = {
            'A': 'Anatomy',
            'B': 'Organisms',
            'C': 'Diseases', 
            'D': 'Chemicals and Drugs',
            'E': 'Analytical, Diagnostic and Therapeutic Techniques and Equipment',
            'F': 'Psychiatry and Psychology',
            'G': 'Phenomena and Processes',
            'H': 'Disciplines and Occupations',
            'I': 'Anthropology, Education, Sociology and Social Phenomena',
            'J': 'Technology, Industry, Agriculture',
            'K': 'Humanities',
            'L': 'Information Science',
            'M': 'Named Groups',
            'N': 'Health Care',
            'V': 'Publication Characteristics',
            'Z': 'Geographicals'
        }
        
        # Agregar categorías principales
        for code, name in main_categories.items():
            mesh_data['tree_to_name'][code] = name
        
        # Generar todos los prefijos de tree numbers
        print("🌳 Generando jerarquía de tree numbers...")
        all_tree_numbers = set()
        for tree_list in mesh_data['ui_to_tree_numbers'].values():
            all_tree_numbers.update(tree_list)
        
        tree_prefixes = set()
        for tree_num in all_tree_numbers:
            parts = tree_num.split('.')
            for i in range(1, len(parts) + 1):
                prefix = '.'.join(parts[:i])
                tree_prefixes.add(prefix)
        
        # Mapear cada prefijo a su nombre correspondiente
        mapped_count = 0
        for prefix in sorted(tree_prefixes):
            if prefix not in mesh_data['tree_to_name']:
                # Buscar el descriptor que tenga exactamente este tree number
                best_match = None
                for ui, tree_numbers in mesh_data['ui_to_tree_numbers'].items():
                    if prefix in tree_numbers:
                        descriptor_name = mesh_data['ui_to_name'].get(ui)
                        if descriptor_name:
                            best_match = descriptor_name
                            break
                
                if best_match:
                    mesh_data['tree_to_name'][prefix] = best_match
                    mapped_count += 1
                else:
                    # Fallback: usar el nombre de la categoría padre + código
                    parent_code = prefix[0]
                    parent_name = main_categories.get(parent_code, f"Category_{parent_code}")
                    mesh_data['tree_to_name'][prefix] = f"{parent_name} ({prefix})"
        
        print(f"✅ Generados mapeos para {len(mesh_data['tree_to_name'])} tree numbers")
        print(f"📊 Términos únicos mapeados: {mapped_count}")
    
    def load_mesh_data(self) -> Dict:
        """
        Carga los datos de MeSH procesados desde el archivo JSON
        
        Returns:
            Dict: Datos de MeSH si existen, None si no se encuentran
        """
        if os.path.exists(self.mesh_data_file):
            print(f"📖 Cargando datos MeSH desde {self.mesh_data_file}")
            try:
                with open(self.mesh_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print("✅ Datos MeSH cargados exitosamente")
                return data
            except Exception as e:
                print(f"❌ Error cargando datos: {e}")
                return None
        return None
    
    def setup_mesh_data(self) -> Dict:
        """
        Configura los datos de MeSH: descarga y procesa si no existen
        
        Este es el método principal que debes usar. Se encarga de:
        1. Verificar si ya existen datos procesados
        2. Si no existen, descargar y procesar el XML
        3. Devolver los datos listos para uso
        
        Returns:
            Dict: Datos de MeSH listos para procesamiento
        """
        # Verificar si ya existen datos procesados
        existing_data = self.load_mesh_data()
        if existing_data:
            return existing_data
        
        # Si no existen, procesarlos desde cero
        print("🚀 Configurando datos MeSH por primera vez...")
        print("⏳ Este proceso puede tomar 10-15 minutos...")
        
        self.download_mesh_file()
        return self.parse_mesh_xml()
    
    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas de los datos MeSH procesados
        
        Returns:
            Dict: Estadísticas de los datos (términos, categorías, etc.)
        """
        data = self.load_mesh_data()
        if not data:
            return {"error": "No hay datos MeSH disponibles"}
        
        return {
            "total_mesh_terms": len(data['ui_to_name']),
            "terms_with_tree_numbers": len(data['ui_to_tree_numbers']),
            "total_categories": len(data['tree_to_name']),
            "main_categories": len([k for k in data['tree_to_name'].keys() if len(k) == 1]),
            "level2_categories": len([k for k in data['tree_to_name'].keys() if '.' not in k and len(k) > 1])
        }