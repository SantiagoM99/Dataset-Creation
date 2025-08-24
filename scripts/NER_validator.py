#!/usr/bin/env python3
"""
Validador de archivos NERsuite para verificar integridad de etiquetado NER
Detecta y reporta violaciones de las reglas de etiquetado BIO

Uso:
python validate_nersuite.py --input ./data --output validation_report.json
python validate_nersuite.py --input ./data --fix --backup
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import csv

@dataclass
class ValidationError:
    """Representa un error de validación"""
    line_number: int
    token: str
    label: str
    error_type: str
    description: str
    suggested_fix: Optional[str] = None

@dataclass
class FileValidationResult:
    """Resultado de validación para un archivo"""
    file_path: Path
    total_lines: int
    total_errors: int
    errors: List[ValidationError] = field(default_factory=list)
    entity_counts: Dict[str, int] = field(default_factory=dict)
    is_valid: bool = True

class NERSuiteValidator:
    """Validador de archivos NERsuite"""
    
    ENCODINGS = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']
    
    def __init__(self, auto_fix: bool = False):
        self.auto_fix = auto_fix
        self.total_files_processed = 0
        self.total_errors_found = 0
        self.error_statistics = defaultdict(int)
    
    def read_file(self, file_path: Path) -> List[Tuple[str, str]]:
        """Lee archivo NERsuite y retorna lista de (token, label)"""
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                tokens_labels = []
                for line in lines:
                    if '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            # Asumiendo formato: label\ttoken o token\tlabel
                            # Detectar automáticamente el formato
                            if len(parts) >= 4:
                                # Formato completo NERsuite
                                label = parts[0]
                                token = parts[3]
                            else:
                                # Formato simplificado
                                token = parts[0]
                                label = parts[1]
                            tokens_labels.append((token, label))
                
                return tokens_labels
                
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError(f"No se pudo decodificar {file_path}")
    
    def validate_file(self, file_path: Path) -> FileValidationResult:
        """Valida un archivo NERsuite"""
        result = FileValidationResult(
            file_path=file_path,
            total_lines=0,
            total_errors=0
        )
        
        try:
            tokens_labels = self.read_file(file_path)
            result.total_lines = len(tokens_labels)
            
            # Variables de estado
            current_entity_type = None
            last_label = None
            
            for i, (token, label) in enumerate(tokens_labels, 1):
                # Validar cada línea
                errors = self.validate_line(
                    i, token, label, last_label, current_entity_type
                )
                
                # Añadir errores al resultado
                for error in errors:
                    result.errors.append(error)
                    result.total_errors += 1
                    self.error_statistics[error.error_type] += 1
                
                # Actualizar estado
                if label.startswith('B-'):
                    entity_type = label[2:]
                    current_entity_type = entity_type
                    result.entity_counts[entity_type] = result.entity_counts.get(entity_type, 0) + 1
                elif label.startswith('I-'):
                    entity_type = label[2:]
                    if current_entity_type != entity_type:
                        current_entity_type = None
                elif label == 'O':
                    current_entity_type = None
                
                last_label = label
            
            result.is_valid = (result.total_errors == 0)
            
        except Exception as e:
            result.errors.append(ValidationError(
                line_number=0,
                token="",
                label="",
                error_type="FILE_ERROR",
                description=f"Error leyendo archivo: {e}"
            ))
            result.is_valid = False
        
        return result
    
    def validate_line(
        self, 
        line_num: int, 
        token: str, 
        label: str, 
        last_label: Optional[str],
        current_entity: Optional[str]
    ) -> List[ValidationError]:
        """Valida una línea individual"""
        errors = []
        
        # Regla 1: Formato de etiqueta válido
        if label not in ['O'] and not label.startswith(('B-', 'I-')):
            errors.append(ValidationError(
                line_number=line_num,
                token=token,
                label=label,
                error_type="INVALID_LABEL_FORMAT",
                description=f"Etiqueta '{label}' no sigue formato BIO",
                suggested_fix="O"
            ))
            return errors
        
        # Regla 2: I- no puede aparecer sin B- previo
        if label.startswith('I-'):
            entity_type = label[2:]
            
            # Caso 1: Primera línea es I-
            if last_label is None:
                errors.append(ValidationError(
                    line_number=line_num,
                    token=token,
                    label=label,
                    error_type="ORPHAN_I_TAG",
                    description=f"I-{entity_type} sin B-{entity_type} previo (primera línea)",
                    suggested_fix=f"B-{entity_type}"
                ))
            
            # Caso 2: I- después de O
            elif last_label == 'O':
                errors.append(ValidationError(
                    line_number=line_num,
                    token=token,
                    label=label,
                    error_type="ORPHAN_I_TAG",
                    description=f"I-{entity_type} sin B-{entity_type} previo (después de O)",
                    suggested_fix=f"B-{entity_type}"
                ))
            
            # Caso 3: I- después de B- o I- diferente
            elif last_label.startswith(('B-', 'I-')):
                last_entity_type = last_label[2:]
                if last_entity_type != entity_type:
                    errors.append(ValidationError(
                        line_number=line_num,
                        token=token,
                        label=label,
                        error_type="ENTITY_TYPE_MISMATCH",
                        description=f"I-{entity_type} después de {last_label}",
                        suggested_fix=f"B-{entity_type}"
                    ))
        
        # Regla 3: Verificar consistencia de tipos de entidad
        if label.startswith(('B-', 'I-')):
            entity_type = label[2:]
            if not entity_type:
                errors.append(ValidationError(
                    line_number=line_num,
                    token=token,
                    label=label,
                    error_type="EMPTY_ENTITY_TYPE",
                    description=f"Etiqueta '{label}' sin tipo de entidad",
                    suggested_fix="O"
                ))
        
        return errors
    
    def fix_file(self, file_path: Path, validation_result: FileValidationResult, backup: bool = True) -> bool:
        """Intenta corregir automáticamente errores en el archivo"""
        if not validation_result.errors:
            return True
        
        try:
            # Crear backup si se solicita
            if backup:
                backup_path = file_path.with_suffix('.nersuite.backup')
                import shutil
                shutil.copy2(file_path, backup_path)
                print(f"📁 Backup creado: {backup_path}")
            
            # Leer archivo original
            tokens_labels = self.read_file(file_path)
            
            # Aplicar correcciones sugeridas
            corrections_applied = 0
            error_dict = {err.line_number: err for err in validation_result.errors}
            
            fixed_tokens_labels = []
            for i, (token, label) in enumerate(tokens_labels, 1):
                if i in error_dict and error_dict[i].suggested_fix:
                    fixed_tokens_labels.append((token, error_dict[i].suggested_fix))
                    corrections_applied += 1
                else:
                    fixed_tokens_labels.append((token, label))
            
            # Guardar archivo corregido
            with open(file_path, 'w', encoding='utf-8') as f:
                for token, label in fixed_tokens_labels:
                    f.write(f"{token}\t{label}\n")
            
            print(f"✅ {corrections_applied} correcciones aplicadas a {file_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error corrigiendo {file_path.name}: {e}")
            return False
    
    def validate_directory(self, directory: Path, fix_errors: bool = False, backup: bool = True) -> Dict:
        """Valida todos los archivos .nersuite en un directorio"""
        results = []
        
        # Encontrar todos los archivos .nersuite
        nersuite_files = list(directory.glob("*.nersuite"))
        
        if not nersuite_files:
            print(f"⚠️ No se encontraron archivos .nersuite en {directory}")
            return {"status": "no_files", "results": []}
        
        print(f"🔍 Validando {len(nersuite_files)} archivos...")
        
        for file_path in nersuite_files:
            self.total_files_processed += 1
            
            # Validar archivo
            result = self.validate_file(file_path)
            self.total_errors_found += result.total_errors
            
            # Intentar corregir si se solicita
            if fix_errors and result.errors and self.auto_fix:
                self.fix_file(file_path, result, backup)
                # Re-validar después de corrección
                result = self.validate_file(file_path)
            
            results.append(result)
            
            # Mostrar progreso
            if result.is_valid:
                print(f"✅ {file_path.name}: OK")
            else:
                print(f"❌ {file_path.name}: {result.total_errors} errores")
        
        return {
            "total_files": self.total_files_processed,
            "total_errors": self.total_errors_found,
            "error_statistics": dict(self.error_statistics),
            "results": results
        }
    
    def generate_report(self, validation_results: Dict, output_path: Path):
        """Genera reporte detallado de validación"""
        
        # Preparar datos para JSON
        report_data = {
            "timestamp": "XD",
            "summary": {
                
                "total_files_processed": validation_results["total_files"],
                "total_errors_found": validation_results["total_errors"],
                "files_with_errors": sum(1 for r in validation_results["results"] if not r.is_valid),
                "error_type_distribution": validation_results["error_statistics"]
            },
            "files": []
        }
        
        # Añadir detalles de cada archivo
        for result in validation_results["results"]:
            file_info = {
                "file": str(result.file_path),
                "valid": result.is_valid,
                "total_lines": result.total_lines,
                "total_errors": result.total_errors,
                "entity_counts": result.entity_counts,
                "errors": [
                    {
                        "line": err.line_number,
                        "token": err.token,
                        "label": err.label,
                        "type": err.error_type,
                        "description": err.description,
                        "suggested_fix": err.suggested_fix
                    }
                    for err in result.errors
                ]
            }
            report_data["files"].append(file_info)
        
        # Guardar JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Reporte JSON guardado: {json_path}")
        
        # Generar reporte CSV de errores
        csv_path = output_path.with_suffix('.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Archivo', 'Línea', 'Token', 'Etiqueta', 'Tipo Error', 'Descripción', 'Corrección Sugerida'])
            
            for result in validation_results["results"]:
                for err in result.errors:
                    writer.writerow([
                        result.file_path.name,
                        err.line_number,
                        err.token,
                        err.label,
                        err.error_type,
                        err.description,
                        err.suggested_fix or 'N/A'
                    ])
        
        print(f"📊 Reporte CSV guardado: {csv_path}")
        
        # Mostrar resumen en consola
        print("\n" + "="*50)
        print("RESUMEN DE VALIDACIÓN")
        print("="*50)
        print(f"📁 Archivos procesados: {validation_results['total_files']}")
        print(f"❌ Total de errores: {validation_results['total_errors']}")
        print(f"📊 Archivos con errores: {sum(1 for r in validation_results['results'] if not r.is_valid)}")
        
        if validation_results["error_statistics"]:
            print("\n📈 Distribución de errores:")
            for error_type, count in sorted(validation_results["error_statistics"].items(), key=lambda x: x[1], reverse=True):
                print(f"   - {error_type}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Validador de archivos NERsuite")
    parser.add_argument('--input', required=True, help='Directorio con archivos .nersuite')
    parser.add_argument('--output', default='validation_report', help='Archivo de reporte (sin extensión)')
    parser.add_argument('--fix', action='store_true', help='Intentar corregir errores automáticamente')
    parser.add_argument('--backup', action='store_true', help='Crear backup antes de corregir')
    parser.add_argument('--verbose', action='store_true', help='Mostrar detalles de cada error')
    
    args = parser.parse_args()
    
    # Configurar validador
    validator = NERSuiteValidator(auto_fix=args.fix)
    
    # Validar directorio
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ El directorio {input_dir} no existe")
        return
    
    # Ejecutar validación
    results = validator.validate_directory(input_dir, fix_errors=args.fix, backup=args.backup)
    
    # Generar reporte
    output_path = Path(args.output)
    validator.generate_report(results, output_path)
    
    # Mostrar errores detallados si verbose
    if args.verbose and results["total_errors"] > 0:
        print("\n" + "="*50)
        print("ERRORES DETALLADOS")
        print("="*50)
        
        for result in results["results"]:
            if result.errors:
                print(f"\n📁 {result.file_path.name}:")
                for err in result.errors[:10]:  # Mostrar máximo 10 errores por archivo
                    print(f"   Línea {err.line_number}: {err.token}\t{err.label}")
                    print(f"   → {err.description}")
                    if err.suggested_fix:
                        print(f"   → Sugerencia: {err.suggested_fix}")
                
                if len(result.errors) > 10:
                    print(f"   ... y {len(result.errors) - 10} errores más")

if __name__ == "__main__":
    main()