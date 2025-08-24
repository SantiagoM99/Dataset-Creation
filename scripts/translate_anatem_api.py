#!/usr/bin/env python3
"""
Traductor mejorado de dataset AnatEM usando OpenAI Batch API
Versión optimizada con clases, división inteligente de documentos y batch processing

Características:
- División inteligente de documentos largos
- Uso de Batch API para reducir costos (50% descuento)
- Procesamiento en stream para archivos grandes
- Reintentos automáticos para archivos fallidos
- Arquitectura modular con clases

Uso:
python anatem_translator.py batch --input ./data --output ./translated
python anatem_translator.py stream --input ./data --output ./translated --file archivo.nersuite
python anatem_translator.py retry --batch-id batch_xyz123
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
import logging
from dotenv import load_dotenv

# ========================================
# CONFIGURACIÓN
# ========================================
@dataclass
class Config:
    """Configuración central del traductor"""
    api_key: str = os.getenv("OPENAI_API_KEY")
    model: str = "gpt-5"
    batch_size: int = 100
    max_lines_per_chunk: int = 25
    max_tokens_per_request: int = 8000
    temperature: float = 0.1
    output_dir: str = "translated_output"
    log_file: str = "translation.log"
    
    # Límites de OpenAI Batch API
    max_requests_per_batch: int = 50000
    max_batch_input_file_size: int = 100 * 1024 * 1024  # 100 MB

# ========================================
# LOGGER
# ========================================
def setup_logger(log_file: str) -> logging.Logger:
    """Configura el sistema de logging"""
    logger = logging.getLogger("AnatEMTranslator")
    logger.setLevel(logging.INFO)
    
    # Archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formato
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ========================================
# CLASES PRINCIPALES
# ========================================
@dataclass
class Token:
    """Representa un token con su etiqueta"""
    text: str
    label: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"text": self.text, "label": self.label}

@dataclass
class DocumentChunk:
    """Representa un fragmento de documento"""
    tokens: List[Token]
    source_file: str
    chunk_index: int
    total_chunks: int
    
    def to_text(self) -> str:
        """Convierte a formato texto para traducción"""
        return "\n".join([f"{t.text}\t{t.label}" for t in self.tokens])
    
    def estimate_tokens(self) -> int:
        """Estima tokens para API de OpenAI"""
        # Aproximación: 1 token ≈ 4 caracteres
        text = self.to_text()
        return len(text) // 4

class NERSuiteParser:
    """Parser para archivos NERsuite"""
    
    ENCODINGS = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']
    
    @classmethod
    def read_file(cls, file_path: Path) -> List[Token]:
        """Lee archivo NERsuite y retorna lista de tokens"""
        for encoding in cls.ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                tokens = []
                for line in lines:
                    if '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 4:
                            label = parts[0]  # Primera columna: etiqueta NER
                            text = parts[3]   # Cuarta columna: token real
                            tokens.append(Token(text, label))
                
                return tokens
                
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError(f"No se pudo decodificar {file_path}")
    
    @classmethod
    def is_valid_file(cls, file_path: Path) -> bool:
        """Verifica si el archivo es válido para procesar"""
        if file_path.name.startswith(('._', '.', 'PMC')):
            return False
        
        if file_path.suffix != '.nersuite':
            return False
        
        if file_path.stat().st_size == 0:
            return False
        
        try:
            with open(file_path, 'rb') as f:
                first_bytes = f.read(50)
            
            if b'Mac OS X' in first_bytes or b'\x00\x05\x16\x07' in first_bytes:
                return False
                
        except:
            return False
        
        return True

class DocumentSplitter:
    """Divide documentos largos en chunks manejables"""
    
    def __init__(self, max_lines: int = 500):
        self.max_lines = max_lines
    
    def split_document(self, tokens: List[Token], source_file: str) -> List[DocumentChunk]:
        """Divide documento en chunks respetando límites y puntos naturales de corte"""
        if len(tokens) <= self.max_lines:
            return [DocumentChunk(tokens, source_file, 0, 1)]
        
        chunks = []
        current_chunk = []
        chunk_index = 0
        
        for i, token in enumerate(tokens):
            current_chunk.append(token)
            
            # Verificar si debemos cortar
            should_split = False
            
            # Si alcanzamos el límite de líneas
            if len(current_chunk) >= self.max_lines:
                # Buscar punto natural de corte (. o ,) en las próximas 50 líneas
                for j in range(i, min(i + 50, len(tokens))):
                    if tokens[j].text in ['.', ',', ';', '!', '?']:
                        # Agregar tokens hasta el punto de corte
                        for k in range(i + 1, j + 1):
                            if k < len(tokens):
                                current_chunk.append(tokens[k])
                        should_split = True
                        break
                
                # Si no encontramos punto natural, cortar de todos modos
                if not should_split and len(current_chunk) >= self.max_lines + 50:
                    should_split = True
            
            if should_split or i == len(tokens) - 1:
                if current_chunk:
                    chunks.append(DocumentChunk(
                        tokens=current_chunk.copy(),
                        source_file=source_file,
                        chunk_index=chunk_index,
                        total_chunks=-1  # Se actualizará después
                    ))
                    chunk_index += 1
                    current_chunk = []
        
        # Actualizar total de chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks

class TranslationPromptBuilder:
    """Construye prompts optimizados para traducción médica"""
    
    @staticmethod
    def build_prompt(chunk: DocumentChunk) -> str:
        """Crea prompt especializado para traducción médica"""
        input_text = chunk.to_text()
        
        chunk_info = ""
        if chunk.total_chunks > 1:
            chunk_info = f"\nNOTA: Este es el fragmento {chunk.chunk_index + 1} de {chunk.total_chunks} del documento."
        
        prompt = f"""Eres un experto etiquetador médico especializado en anatomía. Tu tarea es traducir este dataset de entidades anatómicas del inglés al español manteniendo las etiquetas originales en inglés.

REGLAS FUNDAMENTALES DE NER:
- B-[etiqueta]: Marca el INICIO de una entidad anatómica
- I-[etiqueta]: Marca la CONTINUACIÓN de la misma entidad
- O: Marca tokens que NO son entidades anatómicas
- NUNCA puede haber I- sin un B- previo de la misma etiqueta

INSTRUCCIONES CRÍTICAS:
1. Traduce SOLO las palabras, MANTÉN etiquetas en inglés
2. El número de etiquetas B- debe ser EXACTAMENTE igual al original
3. Si inviertes orden de palabras, ajusta etiquetas coherentemente
4. Si añades preposiciones/artículos para naturalidad:
   - Dentro de entidad: usar I- de la misma etiqueta
   - Fuera de entidad: usar O
5. No añadas etiquetas a términos que no tenían previamente
6. Es posible que haya textos donde hay ninguna etiqueta, no es necesario crear nuevas etiquetas.


EJEMPLOS CLAVE:

1. Inversión con preposición añadida:
ENTRADA:
cell\tB-Organism_substance
lysate\tI-Organism_substance
experiment\tO

SALIDA:
lisado\tB-Organism_substance
celular\tI-Organism_substance
experimento\tO

2. Adición de preposición dentro de entidad:
ENTRADA:
blood\tB-Organism_substance
samples\tI-Organism_substance

SALIDA:
muestras\tB-Organism_substance
de\tI-Organism_substance
sangre\tI-Organism_substance

3. Inversión de términos:
ENTRADA:
ventricular\tB-Multi-tissue_structure
fibrillation\tO

SALIDA:
fibrilación	O
ventricular	B-Multi-tissue_structure

VERIFICACIÓN: (OBLIGATORIO)
No añadir etiquetas en términos que no tenían previamente dicha etiqueta.

TEXTO A TRADUCIR:
{input_text}

TRADUCCIÓN:"""
        
        return prompt

class OpenAIBatchTranslator:
    """Maneja traducciones usando OpenAI Batch API"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.logger = logger
        self.prompt_builder = TranslationPromptBuilder()
    
    def create_batch_request(self, chunk: DocumentChunk, request_id: str) -> Dict[str, Any]:
        """Crea una request para el batch API"""
        prompt = self.prompt_builder.build_prompt(chunk)
        
        return {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Eres un experto médico en terminología anatómica especializado en traducción científica precisa."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens_per_request
            }
        }
    
    def create_batch_file(self, chunks: List[DocumentChunk], output_path: Path) -> Path:
        """Crea archivo JSONL para batch processing"""
        batch_file = output_path / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                request_id = f"{chunk.source_file}__chunk_{chunk.chunk_index}"
                request = self.create_batch_request(chunk, request_id)
                f.write(json.dumps(request) + '\n')
        
        self.logger.info(f"Archivo batch creado: {batch_file} ({len(chunks)} requests)")
        return batch_file
    
    def upload_batch_file(self, file_path: Path) -> str:
        """Sube archivo al servicio de OpenAI"""
        with open(file_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        self.logger.info(f"Archivo subido con ID: {response.id}")
        return response.id
    
    def create_batch_job(self, file_id: str, description: str = None) -> str:
        """Crea un batch job en OpenAI"""
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": description or f"AnatEM translation {datetime.now()}",
                "model": self.config.model
            }
        )
        
        self.logger.info(f"Batch job creado con ID: {batch.id}")
        return batch.id
    
    def check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Verifica el estado de un batch"""
        batch = self.client.batches.retrieve(batch_id)
        
        # Convertir request_counts a diccionario si existe
        request_counts_dict = {}
        if batch.request_counts:
            request_counts_dict = {
                "total": batch.request_counts.total if hasattr(batch.request_counts, 'total') else 0,
                "completed": batch.request_counts.completed if hasattr(batch.request_counts, 'completed') else 0,
                "failed": batch.request_counts.failed if hasattr(batch.request_counts, 'failed') else 0
            }
        
        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "request_counts": request_counts_dict,
            "errors": batch.errors
        }
    
    def wait_for_batch(self, batch_id: str, check_interval: int = 60) -> bool:
        """Espera a que termine un batch"""
        self.logger.info(f"Esperando batch {batch_id}...")
        
        with tqdm(desc="Procesando batch", unit="checks") as pbar:
            while True:
                try:
                    batch = self.client.batches.retrieve(batch_id)
                    
                    if batch.status == "completed":
                        self.logger.info("✅ Batch completado exitosamente")
                        return True
                    elif batch.status == "failed":
                        self.logger.error(f"❌ Batch falló: {batch.errors}")
                        return False
                    elif batch.status == "cancelled":
                        self.logger.warning("⚠️ Batch cancelado")
                        return False
                    
                    # Actualizar barra de progreso
                    completed = 0
                    total = 0
                    if batch.request_counts:
                        completed = getattr(batch.request_counts, 'completed', 0)
                        total = getattr(batch.request_counts, 'total', 0)
                    
                    pbar.set_postfix({
                        "status": batch.status,
                        "completed": completed,
                        "total": total,
                        "progress": f"{completed}/{total}" if total > 0 else "0/0"
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Error verificando batch: {e}")
                    pbar.update(1)
                
                time.sleep(check_interval)
    
    def retrieve_batch_results(self, batch_id: str, output_dir: Path) -> Dict[str, List[Token]]:
        """Recupera y procesa los resultados del batch"""
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.output_file_id is None:
            self.logger.error("No hay archivo de salida disponible")
            return {}
        
        # Descargar archivo de resultados
        content = self.client.files.content(batch.output_file_id)
        results_file = output_dir / f"results_{batch_id}.jsonl"
        
        with open(results_file, 'wb') as f:
            f.write(content.content)
        
        # Procesar resultados
        translations = {}
        
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                custom_id = result["custom_id"]
                
                if result["response"]["status_code"] == 200:
                    content = result["response"]["body"]["choices"][0]["message"]["content"]
                    translated_tokens = self.parse_translation(content)
                    translations[custom_id] = translated_tokens
                else:
                    self.logger.error(f"Error en {custom_id}: {result['response']}")
        
        return translations
    
    def parse_translation(self, response: str) -> List[Token]:
        """Parsea respuesta de traducción"""
        if not response:
            return []
        
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        tokens = []
        
        for line in lines:
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    tokens.append(Token(parts[0], parts[1]))
        
        return tokens

class StreamTranslator:
    """Traductor en modo stream para archivos individuales"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.logger = logger
        self.prompt_builder = TranslationPromptBuilder()
    
    def translate_chunk_stream(self, chunk: DocumentChunk) -> List[Token]:
        """Traduce un chunk en modo stream"""
        prompt = self.prompt_builder.build_prompt(chunk)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un experto médico en terminología anatómica."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_per_request,
                stream=True
            )
            
            # Recolectar respuesta en stream
            full_response = ""
            for chunk_response in stream:
                if chunk_response.choices[0].delta.content is not None:
                    full_response += chunk_response.choices[0].delta.content
            
            # Parsear respuesta
            return self.parse_translation(full_response)
            
        except Exception as e:
            self.logger.error(f"Error en traducción stream: {e}")
            return []
    
    def parse_translation(self, response: str) -> List[Token]:
        """Parsea respuesta de traducción"""
        if not response:
            return []
        
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        tokens = []
        
        for line in lines:
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    tokens.append(Token(parts[0], parts[1]))
        
        return tokens

class TranslationManager:
    """Gestor principal del proceso de traducción"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(config.log_file)
        self.parser = NERSuiteParser()
        self.splitter = DocumentSplitter(config.max_lines_per_chunk)
        self.batch_translator = OpenAIBatchTranslator(config, self.logger)
        self.stream_translator = StreamTranslator(config, self.logger)

    def process_stream_mode(self, input_file: Path, output_dir: Path):
        """Procesa un archivo en modo stream"""
        self.logger.info(f"🔄 Procesando archivo en modo stream: {input_file.name}")
        
        try:
            # Leer archivo
            tokens = self.parser.read_file(input_file)
            
            # Dividir en chunks
            chunks = self.splitter.split_document(tokens, input_file.name)
            
            self.logger.info(f"📦 Archivo dividido en {len(chunks)} chunks")
            
            # Traducir cada chunk
            all_translations = []
            for chunk in tqdm(chunks, desc="Traduciendo chunks"):
                translated_tokens = self.stream_translator.translate_chunk_stream(chunk)
                all_translations.extend(translated_tokens)
            
            # Guardar traducción
            output_file = output_dir / input_file.name
            self.save_file(all_translations, output_file)
            
            self.logger.info(f"✅ Archivo traducido guardado: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error procesando {input_file.name}: {e}")
    
    def process_batch_mode(self, input_dir: Path, output_dir: Path):
        """Procesa archivos en modo batch"""
        self.logger.info("🚀 Iniciando procesamiento en modo batch")
        
        # Encontrar archivos válidos
        all_files = list(input_dir.glob("*.nersuite"))
        valid_files = [f for f in all_files if self.parser.is_valid_file(f)]
        
        self.logger.info(f"📊 Archivos encontrados: {len(all_files)}")
        self.logger.info(f"📊 Archivos válidos: {len(valid_files)}")
        
        if not valid_files:
            self.logger.error("No se encontraron archivos válidos")
            return
        
        # Procesar archivos y crear chunks
        all_chunks = []
        for file_path in tqdm(valid_files, desc="Preparando archivos"):
            try:
                tokens = self.parser.read_file(file_path)
                chunks = self.splitter.split_document(tokens, file_path.name)
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error(f"Error procesando {file_path.name}: {e}")
        
        self.logger.info(f"📦 Total de chunks creados: {len(all_chunks)}")
        
        # Dividir en batches si es necesario
        batches = [all_chunks[i:i + self.config.batch_size] 
                  for i in range(0, len(all_chunks), self.config.batch_size)]
        
        self.logger.info(f"📚 Batches a procesar: {len(batches)}")
        
        # Procesar cada batch
        for i, batch_chunks in enumerate(batches):
            self.logger.info(f"\n🔄 Procesando batch {i+1}/{len(batches)}")
            
            # Crear archivo batch
            batch_file = self.batch_translator.create_batch_file(batch_chunks, output_dir)
            
            # Subir archivo
            file_id = self.batch_translator.upload_batch_file(batch_file)
            
            # Crear batch job
            batch_id = self.batch_translator.create_batch_job(
                file_id, 
                f"AnatEM Batch {i+1}/{len(batches)}"
            )
            
            # Guardar información del batch
            batch_info = {
                "batch_id": batch_id,
                "file_id": file_id,
                "chunks": len(batch_chunks),
                "files": list(set(c.source_file for c in batch_chunks)),
                "created_at": datetime.now().isoformat()
            }
            
            batch_info_file = output_dir / f"batch_info_{batch_id}.json"
            with open(batch_info_file, 'w') as f:
                json.dump(batch_info, f, indent=2)
            
            self.logger.info(f"📝 Batch ID guardado: {batch_id}")
            
            # Esperar y recuperar resultados
            if self.batch_translator.wait_for_batch(batch_id):
                translations = self.batch_translator.retrieve_batch_results(batch_id, output_dir)
                self.save_translations(translations, output_dir)
            else:
                self.logger.error(f"Batch {batch_id} no completado exitosamente")
    
    def _process_single_batch(self, chunks: List[DocumentChunk], output_dir: Path, batch_number: int):
        """Procesa un único batch de chunks"""
        try:
            # Crear archivo batch
            batch_file = self.batch_translator.create_batch_file(chunks, output_dir)
            
            # Subir archivo
            file_id = self.batch_translator.upload_batch_file(batch_file)
            
            # Crear batch job
            batch_id = self.batch_translator.create_batch_job(
                file_id, 
                f"AnatEM Batch - {len(chunks)} chunks"
            )
            
            # Guardar información del batch
            batch_info = {
                "batch_id": batch_id,
                "file_id": file_id,
                "chunks": len(chunks),
                "files": list(set(c.source_file for c in chunks)),
                "created_at": datetime.now().isoformat()
            }
            
            batch_info_file = output_dir / f"batch_info_{batch_id}.json"
            with open(batch_info_file, 'w') as f:
                json.dump(batch_info, f, indent=2)
            
            self.logger.info(f"📝 Batch ID guardado: {batch_id}")
            self.logger.info(f"📊 Chunks en batch: {len(chunks)}")
            self.logger.info(f"📊 Archivos únicos: {len(batch_info['files'])}")
            
            # Esperar y recuperar resultados
            if self.batch_translator.wait_for_batch(batch_id):
                translations = self.batch_translator.retrieve_batch_results(batch_id, output_dir)
                self.save_translations(translations, output_dir)
            else:
                self.logger.error(f"Batch {batch_id} no completado exitosamente")
                
        except Exception as e:
            self.logger.error(f"Error procesando batch: {e}")    
        
    def retry_failed_batch(self, batch_id: str, output_dir: Path):
        """Reintenta procesar un batch fallido o cancelado"""
        self.logger.info(f"🔄 Reintentando batch: {batch_id}")
        
        # Verificar estado actual
        batch = self.batch_translator.client.batches.retrieve(batch_id)
        
        if batch.status == "completed":
            self.logger.info("Batch ya completado, recuperando resultados...")
            translations = self.batch_translator.retrieve_batch_results(batch_id, output_dir)
            self.save_translations(translations, output_dir)
            
        elif batch.status in ["failed", "cancelled", "expired"]:
            self.logger.warning(f"⚠️ Batch en estado '{batch.status}', recreando desde archivo local...")
            
            # Buscar archivo JSONL original en el directorio
            found_file = None
            
            # Buscar por ID del batch en el nombre
            for jsonl_file in output_dir.glob("batch_*.jsonl"):
                with open(jsonl_file, 'r') as f:
                    first_line = f.readline()
                    if batch_id in first_line or batch_id[:8] in jsonl_file.name:
                        found_file = jsonl_file
                        break
            
            # Si no se encuentra, buscar el archivo más reciente
            if not found_file:
                jsonl_files = sorted(output_dir.glob("batch_*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
                if jsonl_files:
                    self.logger.info(f"Usando archivo JSONL más reciente: {jsonl_files[0].name}")
                    found_file = jsonl_files[0]
            
            if found_file:
                self.logger.info(f"📂 Archivo encontrado: {found_file.name}")
                
                # Re-subir el archivo
                new_file_id = self.batch_translator.upload_batch_file(found_file)
                
                # Crear nuevo batch
                new_batch_id = self.batch_translator.create_batch_job(
                    new_file_id,
                    f"Retry of {batch_id[:8]}"
                )
                
                self.logger.info(f"✅ Nuevo batch creado: {new_batch_id}")
                
                # Guardar información del nuevo batch
                retry_info = {
                    "batch_id": new_batch_id,
                    "original_batch_id": batch_id,
                    "file_id": new_file_id,
                    "jsonl_file": str(found_file),
                    "created_at": datetime.now().isoformat()
                }
                
                retry_info_file = output_dir / f"batch_info_{new_batch_id}.json"
                with open(retry_info_file, 'w') as f:
                    json.dump(retry_info, f, indent=2)
                
                # Esperar y procesar
                if self.batch_translator.wait_for_batch(new_batch_id):
                    translations = self.batch_translator.retrieve_batch_results(new_batch_id, output_dir)
                    self.save_translations(translations, output_dir)
                else:
                    self.logger.error(f"❌ Nuevo batch {new_batch_id} falló")
            else:
                self.logger.error(f"❌ No se encontró archivo JSONL para el batch {batch_id}")
                self.logger.info(f"💡 Archivos disponibles en {output_dir}:")
                for f in output_dir.glob("*.jsonl"):
                    self.logger.info(f"   - {f.name}")
                    
        elif batch.status in ["validating", "in_progress", "finalizing"]:
            self.logger.info(f"Batch en progreso, estado: {batch.status}")
            if self.batch_translator.wait_for_batch(batch_id):
                translations = self.batch_translator.retrieve_batch_results(batch_id, output_dir)
                self.save_translations(translations, output_dir)
    
    def save_translations(self, translations: Dict[str, List[Token]], output_dir: Path):
        """Guarda las traducciones agrupadas por archivo"""
        # Agrupar por archivo fuente
        file_translations = {}
        
        for chunk_id, tokens in translations.items():
            # Extraer nombre de archivo del chunk_id
            file_name = chunk_id.split("__chunk_")[0]
            
            if file_name not in file_translations:
                file_translations[file_name] = []
            
            file_translations[file_name].extend(tokens)
        
        # Guardar cada archivo
        for file_name, tokens in file_translations.items():
            output_file = output_dir / file_name
            self.save_file(tokens, output_file)
            self.logger.info(f"✅ Guardado: {output_file}")
    
    def save_file(self, tokens: List[Token], output_file: Path):
        """Guarda tokens en archivo"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for token in tokens:
                f.write(f"{token.text}\t{token.label}\n")

def main():
    parser = argparse.ArgumentParser(description="Traductor mejorado de AnatEM con Batch API")
    
    subparsers = parser.add_subparsers(dest='mode', help='Modo de operación')
    
    # Modo batch
    batch_parser = subparsers.add_parser('batch', help='Procesar en modo batch')
    batch_parser.add_argument('--input', required=True, help='Directorio de entrada')
    batch_parser.add_argument('--output', required=True, help='Directorio de salida')
    batch_parser.add_argument('--api-key', help='API key de OpenAI')
    batch_parser.add_argument('--model', default='gpt-4o', help='Modelo a usar')
    batch_parser.add_argument('--max-lines', type=int, default=200, help='Máximo líneas por chunk')
    
    # Modo stream
    stream_parser = subparsers.add_parser('stream', help='Procesar archivo en stream')
    stream_parser.add_argument('--input', required=True, help='Directorio de entrada')
    stream_parser.add_argument('--output', required=True, help='Directorio de salida')
    stream_parser.add_argument('--file', required=True, help='Archivo específico')
    stream_parser.add_argument('--api-key', help='API key de OpenAI')
    stream_parser.add_argument('--model',default=200, help='Modelo a usar')
    
    # Modo retry
    retry_parser = subparsers.add_parser('retry', help='Reintentar batch fallido')
    retry_parser.add_argument('--batch-id', required=True, help='ID del batch')
    retry_parser.add_argument('--output', required=True, help='Directorio de salida')
    retry_parser.add_argument('--api-key', help='API key de OpenAI')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # Crear configuración
    config = Config()
    
    if hasattr(args, 'api_key') and args.api_key:
        config.api_key = args.api_key
    
    if hasattr(args, 'model') and args.model:
        config.model = args.model
    
    if hasattr(args, 'max_lines') and args.max_lines:
        config.max_lines_per_chunk = args.max_lines
    
    # Crear manager
    manager = TranslationManager(config)
    
    # Ejecutar según modo
    if args.mode == 'batch':
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        manager.process_batch_mode(input_dir, output_dir)
        
    elif args.mode == 'stream':
        input_dir = Path(args.input)
        input_file = input_dir / args.file
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_file.exists():
            print(f"❌ Archivo no encontrado: {input_file}")
            return
        
        manager.process_stream_mode(input_file, output_dir)
        
    elif args.mode == 'retry':
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        manager.retry_failed_batch(args.batch_id, output_dir)

if __name__ == "__main__":
    main()