"""
Script for taking the dataset and creating both traditional and multilabel train/test splits.

This script reads a dataset from a specified path, creates two types of splits:
1. Traditional split: One label per article (using first category)
2. Multilabel split: Binary columns for each category

Usage:
    python create_split.py <input_path> <output_path>
Example:
    python create_split.py dataset/translated_dataset.csv dataset/splits/
"""
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

def create_traditional_split(df, output_path):
    """
    Crea split tradicional usando solo la primera categoría
    """
    
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
    
    
    return train_df, val_df, test_df

def create_multilabel_split(df, output_path):
    """
    Crea split multilabel con columnas binarias para cada categoría
    """
    
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
    
    # Mostrar distribución por categoría
    print(f"\nDistribución por categoría:")
    for i, cat in enumerate(mlb.classes_):
        count = multilabel_matrix[:, i].sum()
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
    
    import json
    with open(os.path.join(output_path, 'multilabel_info.json'), 'w', encoding='utf-8') as f:
        json.dump(categories_info, f, indent=2, ensure_ascii=False)
    
    print(f"   - multilabel_info.json: información de categorías")
    
    return train_df, val_df, test_df, mlb.classes_

def analyze_splits(traditional_splits, multilabel_splits, multilabel_categories):
    """
    Analiza y compara ambos tipos de splits
    """
    print("\n" + "="*80)
    print("ANÁLISIS COMPARATIVO DE SPLITS")
    print("="*80)
    
    train_trad, val_trad, test_trad = traditional_splits
    train_multi, val_multi, test_multi, _ = multilabel_splits
    
    print(f"\n TAMAÑOS DE DATASETS:")
    print(f"   Train: {len(train_trad)}, Val: {len(val_trad)}, Test: {len(test_trad)}")
    
    print(f"\n INFORMACIÓN DE ETIQUETAS:")
    print(f"   Traditional: 1 etiqueta por artículo (single_category)")
    print(f"   Multilabel: {len(multilabel_categories)} categorías binarias")
    

def create_split(input_path, output_path):
    """
    Función principal que crea ambos tipos de splits
    """
    
    # Leer el conjunto de datos
    print(f"Cargando dataset desde: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Total artículos cargados: {len(df)}")
    
    # Filtrar filas sin etiquetas
    print(f"\n🔍 Filtrando artículos sin categorías...")
    has_categories = df['level1_categories_str'].str.len() > 0
    df_filtered = df[has_categories]
    articles_with_categories = len(df_filtered)
    articles_without_categories = len(df) - articles_with_categories
    
    # Crear splits tradicionales
    traditional_splits = create_traditional_split(df_filtered, output_path)
    
    # Crear splits multilabel  
    multilabel_splits = create_multilabel_split(df_filtered, output_path)
    
    # Análisis comparativo
    analyze_splits(traditional_splits, multilabel_splits, multilabel_splits[3])
    
    print(f"\nAmbos splits creados exitosamente en {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crear splits tradicionales y multilabel del dataset.")
    parser.add_argument('input_path', type=str, help='Ruta al archivo CSV de entrada del dataset.')
    parser.add_argument('output_path', type=str, help='Ruta al directorio donde se guardarán los splits.')

    args = parser.parse_args()

    # Asegurarse de que el directorio de salida existe
    os.makedirs(args.output_path, exist_ok=True)

    create_split(args.input_path, args.output_path)