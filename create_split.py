"""
Script for taking the dataset and creating both traditional and multilabel train/test splits
with advanced text preprocessing options.

This script reads a dataset from a specified path, applies text preprocessing to title and
spanish_abstract columns, and creates two types of splits:
1. Traditional split: One label per article (using first category)
2. Multilabel split: Binary columns for each category

Usage:
    python create_split.py <input_path> <output_path> [options]
    
Examples:
    python create_split.py dataset/translated_dataset.csv dataset/splits/
    python create_split.py dataset/translated_dataset.csv dataset/splits/ --text-processing lemmatize
    python create_split.py dataset/translated_dataset.csv dataset/splits/ --text-processing stem --extra-cleaning --min-length 3
"""
import argparse
import pandas as pd
import numpy as np
import os
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Text processing libraries
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Descargando recursos de NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Optional: spaCy for advanced lemmatization
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        print("Modelo de spaCy no encontrado. Ejecuta: python -m spacy download es_core_news_sm")
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    print("spaCy no instalado. Usa: pip install spacy")
    SPACY_AVAILABLE = False
    nlp = None


class TextPreprocessor:
    """Clase para manejar el preprocesamiento de texto en español"""
    
    def __init__(self, processing_type='basic', remove_stopwords=True, min_length=2, extra_cleaning=False):
        self.processing_type = processing_type
        self.remove_stopwords = remove_stopwords
        self.min_length = min_length
        self.extra_cleaning = extra_cleaning
        
        # Configurar herramientas según el tipo de procesamiento
        self.spanish_stopwords = set(stopwords.words('spanish'))
        self.stemmer = SnowballStemmer('spanish')
        
        # Añadir stopwords adicionales comunes en textos académicos
        additional_stopwords = {
            'además', 'asimismo', 'así', 'embargo', 'tanto', 'través', 'debido', 
            'respecto', 'relación', 'partir', 'acuerdo', 'dentro', 'fuera'
        }
        self.spanish_stopwords.update(additional_stopwords)
        
        print(f"Inicializando preprocessor de texto:")
        print(f"   - Tipo de procesamiento: {processing_type}")
        print(f"   - Remover stopwords: {remove_stopwords}")
        print(f"   - Longitud mínima: {min_length}")
        print(f"   - Limpieza extra: {extra_cleaning}")
        
    def clean_text(self, text):
        """Limpieza básica del texto"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convertir a string y lowercase
        text = str(text).lower()
        
        if self.extra_cleaning:
            # Limpieza más agresiva
            # Remover URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            # Remover emails
            text = re.sub(r'\S+@\S+', '', text)
            # Remover números (opcional - mantener solo si no son importantes)
            # text = re.sub(r'\d+', '', text)
            # Remover caracteres especiales extra
            text = re.sub(r'[^\w\s]', ' ', text)
        else:
            # Limpieza básica - remover puntuación pero mantener espacios
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_filter(self, text):
        """Tokenizar y filtrar palabras"""
        if not text:
            return []
        
        # Tokenizar
        try:
            tokens = word_tokenize(text, language='spanish')
        except:
            # Fallback a split simple si word_tokenize falla
            tokens = text.split()
        
        # Filtrar tokens
        filtered_tokens = []
        for token in tokens:
            # Filtrar por longitud mínima
            if len(token) < self.min_length:
                continue
                
            # Remover stopwords si está activado
            if self.remove_stopwords and token in self.spanish_stopwords:
                continue
                
            # Mantener solo tokens que contengan al menos una letra
            if not re.search(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]', token):
                continue
                
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def apply_processing(self, tokens):
        """Aplicar el tipo de procesamiento especificado"""
        if not tokens:
            return []
            
        if self.processing_type == 'stem':
            return [self.stemmer.stem(token) for token in tokens]
        
        elif self.processing_type == 'lemmatize':
            if SPACY_AVAILABLE and nlp:
                # Usar spaCy para lemmatización
                text_to_process = ' '.join(tokens)
                doc = nlp(text_to_process)
                return [token.lemma_ for token in doc if not token.is_space]
            else:
                print("spaCy no disponible, usando stemming como fallback")
                return [self.stemmer.stem(token) for token in tokens]
        
        elif self.processing_type == 'both':
            # Aplicar tanto stemming como lemmatización (experimental)
            if SPACY_AVAILABLE and nlp:
                text_to_process = ' '.join(tokens)
                doc = nlp(text_to_process)
                lemmatized = [token.lemma_ for token in doc if not token.is_space]
                return [self.stemmer.stem(token) for token in lemmatized]
            else:
                return [self.stemmer.stem(token) for token in tokens]
        
        else:  # 'basic' or any other value
            return tokens
    
    def preprocess_text(self, text):
        """Pipeline completo de preprocesamiento"""
        # 1. Limpieza básica
        cleaned_text = self.clean_text(text)
        
        # 2. Tokenización y filtrado
        tokens = self.tokenize_and_filter(cleaned_text)
        
        # 3. Aplicar procesamiento específico
        processed_tokens = self.apply_processing(tokens)
        
        # 4. Unir tokens en texto final
        return ' '.join(processed_tokens)


def preprocess_dataframe_text(df, preprocessor, text_columns=['title', 'spanish_abstract']):
    """Aplicar preprocesamiento a columnas de texto del dataframe"""
    df_processed = df.copy()
    
    print(f"\n🔤 Aplicando preprocesamiento de texto...")
    
    for column in text_columns:
        if column in df_processed.columns:
            print(f"   Procesando columna: {column}")
            
            # Mostrar estadísticas antes
            non_empty_before = df_processed[column].notna().sum()
            avg_length_before = df_processed[column].str.len().mean()
            
            # Aplicar preprocesamiento
            df_processed[f'{column}_processed'] = df_processed[column].apply(
                preprocessor.preprocess_text
            )
            
            # Mostrar estadísticas después
            non_empty_after = (df_processed[f'{column}_processed'] != '').sum()
            avg_length_after = df_processed[f'{column}_processed'].str.len().mean()
            
            print(f"      - Antes: {non_empty_before} textos, {avg_length_before:.1f} chars promedio")
            print(f"      - Después: {non_empty_after} textos, {avg_length_after:.1f} chars promedio")
        else:
            print(f"   ⚠️  Columna '{column}' no encontrada")
    
    return df_processed


def create_traditional_split(df, output_path):
    """Crea split tradicional usando solo la primera categoría"""
    
    # Usar solo la primera categoría para clasificación tradicional
    df_traditional = df.copy()
    df_traditional['single_category'] = df_traditional['level1_categories_str'].str.split(';').str[0].str.strip()
    
    # Mostrar distribución
    category_counts = df_traditional['single_category'].value_counts()
    print("\nDistribución de categorías principales:")
    print(category_counts)
    
    # Split sin stratify (para evitar errores con categorías raras)
    train_df, test_df = train_test_split(df_traditional, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Guardar splits tradicionales
    train_df.to_csv(os.path.join(output_path, 'train_traditional.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'val_traditional.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test_traditional.csv'), index=False)
    
    print(f"\nSplit tradicional guardado:")
    print(f"   - train_traditional.csv: {len(train_df)} artículos")
    print(f"   - val_traditional.csv: {len(val_df)} artículos")
    print(f"   - test_traditional.csv: {len(test_df)} artículos")
    
    return train_df, val_df, test_df


def create_multilabel_split(df, output_path):
    """Crea split multilabel con columnas binarias para cada categoría"""
    
    # Convertir string a lista de categorías
    df_multilabel = df.copy()
    df_multilabel['categories_list'] = df_multilabel['level1_categories_str'].str.split(';').apply(
        lambda x: [cat.strip() for cat in x] if x else []
    )
    
    # Crear matriz binaria multilabel
    mlb = MultiLabelBinarizer()
    multilabel_matrix = mlb.fit_transform(df_multilabel['categories_list'])

    # Crear nombres de columnas limpias
    category_columns = []
    for cat in mlb.classes_:
        # Limpiar nombres para columnas válidas
        clean_name = cat.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '').replace('-', '_')
        category_columns.append(f"category_{clean_name}")
    
    # Crear DataFrame con columnas binarias
    multilabel_df = pd.DataFrame(multilabel_matrix, columns=category_columns, index=df_multilabel.index)
    
    # Combinar con datos originales (sin categories_list)
    df_final = pd.concat([df_multilabel.drop(['categories_list'], axis=1), multilabel_df], axis=1)
    
    # Mostrar estadísticas multilabel
    print(f"\nEstadísticas multilabel:")
    print(f"   - Categorías únicas: {len(mlb.classes_)}")
    print(f"   - Forma de matriz: {multilabel_matrix.shape}")
    
    # Mostrar distribución por categoría (top 10)
    print(f"\nTop 10 categorías por frecuencia:")
    category_counts = [(cat, multilabel_matrix[:, i].sum()) for i, cat in enumerate(mlb.classes_)]
    category_counts.sort(key=lambda x: x[1], reverse=True)
    
    for cat, count in category_counts[:10]:
        percentage = (count / len(df_multilabel)) * 100
        print(f"   - {cat}: {count} artículos ({percentage:.1f}%)")
    
    # Split multilabel (sin stratify porque es complejo para multilabel)
    train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Guardar splits multilabel
    train_df.to_csv(os.path.join(output_path, 'train_multilabel.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'val_multilabel.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test_multilabel.csv'), index=False)
    
    print(f"\nSplit multilabel guardado:")
    print(f"   - train_multilabel.csv: {len(train_df)} artículos")
    print(f"   - val_multilabel.csv: {len(val_df)} artículos")
    print(f"   - test_multilabel.csv: {len(test_df)} artículos")
    
    # Guardar información de las categorías
    categories_info = {
        'categories': list(mlb.classes_),
        'column_names': category_columns,
        'total_samples': len(df_multilabel),
        'category_counts': {cat: int(multilabel_matrix[:, i].sum()) for i, cat in enumerate(mlb.classes_)}
    }
    
    with open(os.path.join(output_path, 'multilabel_info.json'), 'w', encoding='utf-8') as f:
        json.dump(categories_info, f, indent=2, ensure_ascii=False)
    
    print(f"   - multilabel_info.json: información de categorías")
    
    return train_df, val_df, test_df, mlb.classes_


def analyze_splits(traditional_splits, multilabel_splits, multilabel_categories, preprocessor=None):
    """Analiza y compara ambos tipos de splits"""
    print("\n" + "="*80)
    print("ANÁLISIS COMPARATIVO DE SPLITS")
    print("="*80)
    
    train_trad, val_trad, test_trad = traditional_splits
    train_multi, val_multi, test_multi, _ = multilabel_splits
    
    print(f"\n📊 TAMAÑOS DE DATASETS:")
    print(f"   Train: {len(train_trad)}, Val: {len(val_trad)}, Test: {len(test_trad)}")
    
    print(f"\n🏷️  INFORMACIÓN DE ETIQUETAS:")
    print(f"   Traditional: 1 etiqueta por artículo (single_category)")
    print(f"   Multilabel: {len(multilabel_categories)} categorías binarias")
    
    if preprocessor:
        print(f"\n🔤 PROCESAMIENTO DE TEXTO APLICADO:")
        print(f"   Tipo: {preprocessor.processing_type}")
        print(f"   Stopwords removidas: {preprocessor.remove_stopwords}")
        print(f"   Longitud mínima: {preprocessor.min_length}")
        print(f"   Limpieza extra: {preprocessor.extra_cleaning}")


def create_split(input_path, output_path, text_processing='basic', min_length=2, extra_cleaning=False, remove_stopwords=True):
    """Función principal que crea ambos tipos de splits con preprocesamiento de texto"""
    
    # Leer el conjunto de datos
    print(f"📁 Cargando dataset desde: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Total artículos cargados: {len(df)}")
    
    # Filtrar filas sin etiquetas
    print(f"\n🔍 Filtrando artículos sin categorías...")
    has_categories = df['level1_categories_str'].notna() & (df['level1_categories_str'].str.len() > 0)
    df_filtered = df[has_categories]
    articles_with_categories = len(df_filtered)
    articles_without_categories = len(df) - articles_with_categories
    
    print(f"   Con categorías: {articles_with_categories}")
    print(f"   Sin categorías: {articles_without_categories}")
    
    # Inicializar preprocessor de texto
    preprocessor = TextPreprocessor(
        processing_type=text_processing,
        remove_stopwords=remove_stopwords,
        min_length=min_length,
        extra_cleaning=extra_cleaning
    )
    
    # Aplicar preprocesamiento de texto
    df_processed = preprocess_dataframe_text(df_filtered, preprocessor)
    
    # Crear splits tradicionales
    print(f"\n📝 Creando splits tradicionales...")
    traditional_splits = create_traditional_split(df_processed, output_path)
    
    # Crear splits multilabel  
    print(f"\n🏷️  Creando splits multilabel...")
    multilabel_splits = create_multilabel_split(df_processed, output_path)
    
    # Guardar información del preprocessor
    preprocessing_info = {
        'text_processing_type': text_processing,
        'remove_stopwords': remove_stopwords,
        'min_length': min_length,
        'extra_cleaning': extra_cleaning,
        'processed_columns': ['title_processed', 'spanish_abstract_processed'],
        'original_columns': ['title', 'spanish_abstract']
    }
    
    with open(os.path.join(output_path, 'preprocessing_info.json'), 'w', encoding='utf-8') as f:
        json.dump(preprocessing_info, f, indent=2, ensure_ascii=False)
    
    # Análisis comparativo
    analyze_splits(traditional_splits, multilabel_splits, multilabel_splits[3], preprocessor)
    
    print(f"\n✅ Ambos splits creados exitosamente en {output_path}")
    print(f"   - Columnas originales mantenidas")
    print(f"   - Columnas procesadas: title_processed, spanish_abstract_processed")
    print(f"   - Información de preprocesamiento guardada en preprocessing_info.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crear splits tradicionales y multilabel del dataset con preprocesamiento de texto.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s dataset.csv output/ 
  %(prog)s dataset.csv output/ --text-processing stem
  %(prog)s dataset.csv output/ --text-processing lemmatize --extra-cleaning
  %(prog)s dataset.csv output/ --text-processing both --min-length 3 --no-stopwords
        """
    )
    
    parser.add_argument('input_path', type=str, 
                       help='Ruta al archivo CSV de entrada del dataset.')
    parser.add_argument('output_path', type=str, 
                       help='Ruta al directorio donde se guardarán los splits.')
    
    # Argumentos de preprocesamiento de texto
    parser.add_argument('--text-processing', 
                       choices=['basic', 'stem', 'lemmatize', 'both'], 
                       default='basic',
                       help='Tipo de procesamiento de texto: basic (solo limpieza), stem (stemming), lemmatize (lemmatización), both (ambos)')
    
    parser.add_argument('--min-length', type=int, default=2,
                       help='Longitud mínima de palabras a mantener (default: 2)')
    
    parser.add_argument('--extra-cleaning', action='store_true',
                       help='Aplicar limpieza extra (URLs, emails, etc.)')
    
    parser.add_argument('--no-stopwords', action='store_true',
                       help='NO remover stopwords (por defecto SÍ se remueven)')

    args = parser.parse_args()

    # Asegurarse de que el directorio de salida existe
    os.makedirs(args.output_path, exist_ok=True)

    # Procesar argumentos
    remove_stopwords = not args.no_stopwords  # Invertir lógica

    create_split(
        input_path=args.input_path,
        output_path=args.output_path,
        text_processing=args.text_processing,
        min_length=args.min_length,
        extra_cleaning=args.extra_cleaning,
        remove_stopwords=remove_stopwords
    )