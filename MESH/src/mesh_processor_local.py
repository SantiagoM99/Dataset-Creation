# src/mesh_processor_local.py
"""
MeshProcessorLocal - Procesa artículos científicos y extrae categorías MeSH

Esta clase se encarga de:
1. Cargar datos MeSH preprocesados desde mesh_data.json
2. Procesar artículos individuales o datasets completos
3. Extraer categorías de nivel 1 y 2 solo de major subjects
4. Generar DataFrames con categorías MeSH y abstracts en español

Uso:
    processor = MeshProcessorLocal()
    df = processor.process_json_file('mi_dataset.json')
"""

import json
import os
import pandas as pd
from typing import Dict, List, Set

class MeshProcessorLocal:
    """
    Procesador local de artículos científicos usando datos MeSH precargados
    
    Utiliza datos MeSH preprocesados para extraer categorías de manera
    eficiente sin necesidad de APIs externas.
    
    Attributes:
        mesh_data (Dict): Datos MeSH precargados (ui_to_tree_numbers, etc.)
        mesh_data_file (str): Ruta al archivo mesh_data.json
    """
    
    def __init__(self, mesh_data_dir: str = "mesh/data"):
        """
        Inicializa el procesador MeSH local
        
        Args:
            mesh_data_dir (str): Directorio donde están los datos MeSH
            
        Raises:
            Exception: Si no se pueden cargar los datos MeSH
        """
        self.mesh_data_file = os.path.join(mesh_data_dir, "mesh_data.json")
        self.mesh_data = self._load_mesh_data()
        
        if not self.mesh_data:
            raise Exception(f"❌ No se pudieron cargar los datos de MeSH desde {self.mesh_data_file}")
        
        print("✅ MeSH data cargado exitosamente:")
        print(f"  📊 {len(self.mesh_data['ui_to_tree_numbers'])} términos con tree numbers")
        print(f"  🗂️  {len(self.mesh_data['tree_to_name'])} categorías mapeadas")
        print(f"  🏷️  {len(self.mesh_data['ui_to_name'])} nombres de términos")
    
    def _load_mesh_data(self) -> Dict:
        """
        Carga los datos de MeSH desde el archivo JSON
        
        Returns:
            Dict: Datos MeSH si existen, None si hay error
        """
        if not os.path.exists(self.mesh_data_file):
            print(f"❌ Archivo {self.mesh_data_file} no encontrado")
            print("💡 Ejecuta primero: python mesh/src/run_mesh_downloader.py")
            return None
        
        try:
            with open(self.mesh_data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error cargando {self.mesh_data_file}: {e}")
            return None
    
    def get_mesh_tree_numbers(self, mesh_ui: str) -> List[str]:
        """
        Obtiene los Tree Numbers de un MeSH UI desde datos locales
        
        Args:
            mesh_ui (str): MeSH Unique Identifier (ej: "D004285")
            
        Returns:
            List[str]: Lista de tree numbers (ej: ["B01.050.150.900.649.313"])
        """
        return self.mesh_data['ui_to_tree_numbers'].get(mesh_ui, [])
    
    def get_category_name(self, tree_code: str) -> str:
        """
        Obtiene el nombre de una categoría desde datos locales
        
        Args:
            tree_code (str): Código de tree (ej: "B", "B01", "B01.050")
            
        Returns:
            str: Nombre de la categoría (ej: "Organisms", "Eukaryota")
        """
        return self.mesh_data['tree_to_name'].get(tree_code, f"Unknown_{tree_code}")
    
    def extract_tree_levels(self, tree_number: str) -> tuple:
        """
        Extrae el primer y segundo nivel de un Tree Number
        
        Args:
            tree_number (str): Tree number completo (ej: "B01.050.150.900")
            
        Returns:
            tuple: (nivel1, nivel2) (ej: ("B", "B01"))
        """
        parts = tree_number.split('.')
        level1 = parts[0][0] if parts[0] else ''  # Primera letra
        level2 = parts[0] if len(parts) > 0 else ''  # Primer segmento
        return level1, level2
    
    def get_mesh_categories(self, mesh_ui: str) -> Dict[str, Set[str]]:
        """
        Obtiene las categorías de primer y segundo nivel para un MeSH UI
        
        Args:
            mesh_ui (str): MeSH Unique Identifier
            
        Returns:
            Dict[str, Set[str]]: Diccionario con level1_codes y level2_codes
        """
        tree_numbers = self.get_mesh_tree_numbers(mesh_ui)
        
        level1_codes = set()
        level2_codes = set()
        
        for tree_num in tree_numbers:
            l1, l2 = self.extract_tree_levels(tree_num)
            if l1:
                level1_codes.add(l1)
            if l2:
                level2_codes.add(l2)
        
        return {
            'level1_codes': level1_codes,
            'level2_codes': level2_codes
        }
    
    def process_article_mesh(self, article: Dict, debug: bool = False) -> Dict:
        """
        Procesa los MeSH terms de un artículo individual
        
        Solo procesa términos marcados como major_topic=True para obtener
        las categorías principales del artículo.
        
        Args:
            article (Dict): Datos del artículo con mesh_terms
            debug (bool): Si mostrar información de debugging
            
        Returns:
            Dict: Categorías extraídas (level1_codes, level1_names, etc.)
        """
        level1_categories = set()
        level2_categories = set()
        
        mesh_terms = article.get('mesh_terms', [])
        
        if debug:
            print(f"\n=== DEBUGGING ARTÍCULO {article.get('pmid')} ===")
            print(f"📊 Total MeSH terms: {len(mesh_terms)}")
        
        major_count = 0
        processed_uis = set()  # Evitar duplicados
        
        for mesh_term in mesh_terms:
            descriptor = mesh_term.get('descriptor', {})
            
            if debug:
                print(f"MeSH: {descriptor.get('name', 'N/A')} ({descriptor.get('ui', 'N/A')}) - Major: {descriptor.get('major_topic', 'N/A')}")
            
            # Solo procesar major subjects únicos
            if descriptor.get('major_topic', False):
                mesh_ui = descriptor.get('ui')
                if mesh_ui and mesh_ui not in processed_uis:
                    processed_uis.add(mesh_ui)
                    major_count += 1
                    
                    if debug:
                        print(f"  🔍 Procesando MeSH UI: {mesh_ui}")
                    
                    categories = self.get_mesh_categories(mesh_ui)
                    
                    if debug:
                        print(f"  📂 Categorías encontradas: L1={categories['level1_codes']}, L2={categories['level2_codes']}")
                    
                    level1_categories.update(categories['level1_codes'])
                    level2_categories.update(categories['level2_codes'])
        
        if debug:
            print(f"✅ Major subjects únicos procesados: {major_count}")
            print(f"📊 Categorías finales L1: {level1_categories}")
            print(f"📊 Categorías finales L2: {level2_categories}")
        
        # Obtener nombres de categorías
        level1_names = []
        for code in sorted(level1_categories):
            name = self.get_category_name(code)
            level1_names.append(name)
            if debug:
                print(f"  🏷️  L1 {code} -> {name}")
        
        level2_names = []
        for code in sorted(level2_categories):
            name = self.get_category_name(code)
            level2_names.append(name)
            if debug:
                print(f"  🏷️  L2 {code} -> {name}")
        
        return {
            'level1_codes': sorted(list(level1_categories)),
            'level1_names': level1_names,
            'level2_codes': sorted(list(level2_categories)),
            'level2_names': level2_names,
            'major_subjects_count': major_count
        }
    
    def process_dataset(self, articles: List[Dict], debug: bool = False) -> pd.DataFrame:
        """
        Procesa una lista de artículos y devuelve DataFrame con categorías
        
        Args:
            articles (List[Dict]): Lista de artículos a procesar
            debug (bool): Si mostrar debugging para los primeros 3 artículos
            
        Returns:
            pd.DataFrame: DataFrame con categorías MeSH y abstracts en español
        """
        results = []
        
        print(f"🚀 Iniciando procesamiento de {len(articles)} artículos...")
        
        for i, article in enumerate(articles):
            if (i + 1) % 100 == 0 or i < 10:
                print(f"📊 Procesando artículo {i+1}/{len(articles)}...")
            
            # Debug detallado solo para los primeros 3 artículos
            article_debug = debug and i < 3
            
            categories = self.process_article_mesh(article, debug=article_debug)
            
            # Extraer abstract en español
            spanish_abstract = ""
            abstract_info = article.get('abstract', {})
            if abstract_info and isinstance(abstract_info, dict):
                spanish_translation = abstract_info.get('spanish_translation', {})
                if spanish_translation and spanish_translation.get('success', False):
                    spanish_abstract = spanish_translation.get('text', '')
            
            result = {
                'pmid': article.get('pmid'),
                'title': article.get('title', ''),
                'year': article.get('publication_info', {}).get('year'),
                'spanish_abstract': spanish_abstract,
                'level1_codes': categories['level1_codes'],
                'level1_names': categories['level1_names'],
                'level2_codes': categories['level2_codes'],
                'level2_names': categories['level2_names'],
                'major_subjects_count': categories['major_subjects_count'],
                'level1_categories_str': '; '.join(categories['level1_names']),
                'level2_categories_str': '; '.join(categories['level2_names'])
            }
            
            results.append(result)
        
        df = pd.DataFrame(results)
        print(f"✅ Procesamiento completado: {len(df)} artículos")
        
        return df
    
    def process_json_file(self, json_file_path: str, debug: bool = False) -> pd.DataFrame:
        """
        Procesa un archivo JSON con artículos científicos
        
        Args:
            json_file_path (str): Ruta al archivo JSON con artículos
            debug (bool): Si mostrar información de debugging
            
        Returns:
            pd.DataFrame: DataFrame con categorías MeSH y abstracts
            
        Raises:
            FileNotFoundError: Si no se encuentra el archivo
            Exception: Si hay errores procesando el archivo
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"❌ Archivo no encontrado: {json_file_path}")
        
        print(f"📖 Cargando artículos desde: {json_file_path}")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            raise Exception(f"❌ Error cargando JSON: {e}")
        
        articles = data.get('articles', [])
        if not articles:
            raise Exception("❌ No se encontraron artículos en el archivo JSON")
        
        print(f"✅ Cargados {len(articles)} artículos")
        
        return self.process_dataset(articles, debug=debug)
    
    def save_results(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Guarda los resultados en un archivo CSV
        
        Args:
            df (pd.DataFrame): DataFrame con resultados
            output_path (str): Ruta donde guardar el archivo
        """
        try:
            # Crear directorio si no existe
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"💾 Resultados guardados en: {output_path}")
            
        except Exception as e:
            print(f"❌ Error guardando archivo: {e}")
            raise
    
    def get_processing_stats(self, df: pd.DataFrame) -> Dict:
        """
        Obtiene estadísticas del procesamiento realizado
        
        Args:
            df (pd.DataFrame): DataFrame procesado
            
        Returns:
            Dict: Estadísticas del procesamiento
        """
        # Artículos con categorías
        has_categories = df['level1_categories_str'].str.len() > 0
        articles_with_categories = has_categories.sum()
        
        # Artículos con abstract en español
        has_spanish_abstract = df['spanish_abstract'].str.len() > 0
        articles_with_spanish = has_spanish_abstract.sum()
        
        # Categorías más frecuentes
        all_l1_categories = []
        for categories_str in df['level1_categories_str']:
            if categories_str:
                all_l1_categories.extend(categories_str.split('; '))
        
        from collections import Counter
        category_counts = Counter(all_l1_categories)
        
        return {
            'total_articles': len(df),
            'articles_with_categories': articles_with_categories,
            'articles_with_spanish_abstract': articles_with_spanish,
            'coverage_percentage': (articles_with_categories / len(df)) * 100,
            'spanish_abstract_percentage': (articles_with_spanish / len(df)) * 100,
            'unique_level1_categories': len(category_counts),
            'most_common_categories': dict(category_counts.most_common(10))
        }